import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pandas as pd
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import warnings
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict

import asyncio

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from alpaca_trade_api.entity import Order as AlpacaOrder

import ta  # We only use ta.volatility for the ATR calculation
from pytz import timezone as pytz_timezone
from datetime import timezone as dt_timezone

import nest_asyncio
nest_asyncio.apply()

# ============================
# Dynamic Config Reload Setup
# ============================
last_config_load_time = None
config_cache = None
CONFIG_PATH = "config.json"

def load_config(force_reload=False):
    """
    Dynamically load the config.json file. If 'force_reload' is True
    or the file has changed, reload it from disk.
    """
    global last_config_load_time, config_cache
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError("config.json file not found.")
    current_mtime = os.path.getmtime(CONFIG_PATH)
    if force_reload or last_config_load_time is None or current_mtime != last_config_load_time:
        with open(CONFIG_PATH, "r") as config_file:
            config_cache = json.load(config_file)
        last_config_load_time = current_mtime
        logger.info("Configuration reloaded from config.json.")
    return config_cache

# ============================
# AlpacaTrader Class
# ============================

class AlpacaTrader:
    def __init__(self, config: dict, logger: logging.Logger):
        self.logger = logger
        self.alpaca_rest_client = REST(
            key_id=config.get("API_KEY"),
            secret_key=config.get("API_SECRET"),
            base_url=config.get("BASE_URL", "https://paper-api.alpaca.markets"),
            api_version='v2'
        )

        nyc = pytz_timezone("America/New_York")
        now = datetime.now(nyc)
        try:
            calendars = self.alpaca_rest_client.get_calendar(
                start=now.strftime("%Y-%m-%d"), end=now.strftime("%Y-%m-%d")
            )
            if not calendars:
                self.logger.error("No calendar data returned from Alpaca API.")
                self.market_open = self.market_close = None
                return
            calendar = calendars[0]

            if now.date() >= calendar.date.date():
                self.market_open = now.replace(
                    hour=calendar.open.hour,
                    minute=calendar.open.minute,
                    second=0,
                    microsecond=0
                )
                self.market_close = now.replace(
                    hour=calendar.close.hour,
                    minute=calendar.close.minute,
                    second=0,
                    microsecond=0
                )
            else:
                self.market_open = self.market_close = None
        except Exception as e:
            self.logger.error(f"Error fetching market calendar: {e}", exc_info=True)
            self.market_open = self.market_close = None

    async def get_position(self, symbol: str) -> Optional[AlpacaOrder]:
        """
        Return the position object if a position exists for 'symbol', otherwise None.
        """
        try:
            pos = self.alpaca_rest_client.get_position(symbol)
            if float(pos.qty) == 0:
                return None
            return pos
        except tradeapi.rest.APIError as api_err:
            if 'Position does not exist for asset' in str(api_err):
                return None
            self.logger.error(f"APIError retrieving position for {symbol}: {api_err}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving position for {symbol}: {e}", exc_info=True)
            return None

    async def submit_order(self, **kwargs) -> Optional[AlpacaOrder]:
        """
        Submit an order to Alpaca and return the Order object if successful.
        """
        try:
            order = self.alpaca_rest_client.submit_order(**kwargs)
            self.logger.info(f"Order submitted: {order}")
            return order
        except tradeapi.rest.APIError as api_err:
            self.logger.error(f"APIError submitting order: {api_err}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}", exc_info=True)
            return None

    async def close_position_direct(self, symbol: str):
        """
        Close the position in 'symbol' by first canceling any open orders and then
        calling Alpaca's close_position endpoint.
        """
        try:
            orders = self.alpaca_rest_client.list_orders(
                status='open',
                symbols=[symbol],
                limit=50
            )
            if orders:
                self.logger.info(f"Found {len(orders)} open orders for {symbol}. Canceling them.")
                for order in orders:
                    try:
                        self.alpaca_rest_client.cancel_order(order.id)
                        self.logger.info(f"Canceled order {order.id} for {symbol}.")
                    except tradeapi.rest.APIError as e:
                        self.logger.error(f"Error canceling order {order.id} for {symbol}: {e}", exc_info=True)
                    except Exception as e:
                        self.logger.error(f"Unexpected error canceling order {order.id}: {e}", exc_info=True)
                # Give some time for cancellation to process
                await asyncio.sleep(2)

            # Now attempt to close the position
            try:
                self.alpaca_rest_client.close_position(symbol)
                self.logger.info(f"Close position request sent for {symbol}")
                return True
            except tradeapi.rest.APIError as e:
                self.logger.error(f"APIError closing position for {symbol}: {e}", exc_info=True)
                return False
            except Exception as e:
                self.logger.error(f"Error closing position for {symbol}: {e}", exc_info=True)
                return False

        except Exception as e:
            self.logger.error(f"Error preparing to close position for {symbol}: {e}", exc_info=True)
            return False

    async def get_latest_quote(self, symbol: str):
        """
        Return a mid-price if bid/ask exist, else last_price from the latest quote.
        """
        try:
            quote = self.alpaca_rest_client.get_latest_quote(symbol)
            if not quote:
                self.logger.warning(f"No quote data available for {symbol}.")
                return None
            if quote.bid_price is not None and quote.ask_price is not None:
                latest_price = float(quote.bid_price + quote.ask_price) / 2
            elif quote.last_price is not None:
                latest_price = float(quote.last_price)
            else:
                self.logger.warning(f"Incomplete quote data for {symbol}.")
                return None
            return latest_price
        except tradeapi.rest.APIError as api_err:
            self.logger.error(f"APIError fetching latest quote for {symbol}: {api_err}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching latest quote for {symbol}: {e}", exc_info=True)
            return None

    async def list_positions(self) -> List[AlpacaOrder]:
        """
        List all current positions in the Alpaca account.
        """
        try:
            positions = self.alpaca_rest_client.list_positions()
            return positions
        except Exception as e:
            self.logger.error(f"Error listing positions: {e}", exc_info=True)
            return []

    async def is_shortable(self, symbol: str) -> bool:
        """
        Determine if the given symbol can be shorted.
        """
        try:
            asset = self.alpaca_rest_client.get_asset(symbol)
            return asset.shortable
        except Exception as e:
            self.logger.error(f"Error checking if {symbol} is shortable: {e}", exc_info=True)
            return False

    async def get_account(self):
        """
        Return the Alpaca account details.
        """
        try:
            account = self.alpaca_rest_client.get_account()
            return account
        except Exception as e:
            self.logger.error(f"Error fetching account information: {e}", exc_info=True)
            return None

    async def run(self):
        """Placeholder: can be used for event-based streaming logic if needed."""
        pass

    async def close(self):
        """Placeholder: gracefully close streams or finalize tasks if needed."""
        pass

# ============================
# Initialize Logging
# ============================
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config.json not found.")

with open(CONFIG_PATH, "r") as f:
    initial_config = json.load(f)

LOG_FILE = initial_config.get("LOG_FILE", "trading_bot.log")
LOG_LEVEL = getattr(logging, initial_config.get("LOG_LEVEL", "INFO").upper(), logging.INFO)

logger = logging.getLogger('TradingBot')
logger.setLevel(LOG_LEVEL)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10**6, backupCount=5)
file_formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

config = load_config(force_reload=True)

API_KEY = config.get("API_KEY")
API_SECRET = config.get("API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("API_KEY or API_SECRET not found in config.json.")

trader = AlpacaTrader(config=config, logger=logger)

SYMBOL_LIST_FILE = config.get("SYMBOL_LIST_FILE", "list.txt")

class SymbolListScanner:
    def __init__(self, symbol_file: str, logger: logging.Logger):
        self.symbol_file = symbol_file
        self.logger = logger

    async def scan(self) -> List[str]:
        """
        Read symbols from the provided file (one symbol per line).
        """
        try:
            if not os.path.exists(self.symbol_file):
                self.logger.error(f"Symbol file {self.symbol_file} does not exist.")
                return []
            with open(self.symbol_file, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(symbols)} symbols from {self.symbol_file}.")
            return symbols
        except Exception as e:
            self.logger.error(f"Error reading symbol file {self.symbol_file}: {e}", exc_info=True)
            return []

symbol_scanner = SymbolListScanner(
    symbol_file=SYMBOL_LIST_FILE,
    logger=logger
)

# For recording trades
trade_log = []
analysis_reports = []

# Ensure directories exist
for directory in ['cached_data', 'analysis_reports']:
    os.makedirs(directory, exist_ok=True)

# ============================
# Data Fetching
# ============================

def fetch_historical_data(symbol, timeframe='15Min', limit_days=365):
    """
    Fetch historical OHLCV data from Alpaca for the given symbol and timeframe.
    """
    end_time = datetime.now(dt_timezone.utc)
    start_time = end_time - timedelta(days=limit_days)
    try:
        if timeframe == '15Min':
            bars = trader.alpaca_rest_client.get_bars(
                symbol,
                TimeFrame(15, TimeFrameUnit.Minute),
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                adjustment='raw'
            ).df
        else:
            # For demonstration, fallback to the same 15Min if another timeframe was requested.
            bars = trader.alpaca_rest_client.get_bars(
                symbol,
                TimeFrame(15, TimeFrameUnit.Minute),
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                adjustment='raw'
            ).df

        if bars.empty:
            logger.warning(f"No data fetched for {symbol}.")
            return pd.DataFrame()

        bars = bars.reset_index()
        data = bars[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return data
    except tradeapi.rest.APIError as api_err:
        logger.error(f"APIError fetching data for {symbol}: {api_err}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()

# ============================
# Custom SuperTrend Calculation
# ============================

def compute_supertrend(data: pd.DataFrame, atr_period: int, multiplier: float) -> pd.DataFrame:
    """
    Manual SuperTrend calculation that returns the 'supertrend' and
    'supertrend_direction' (1 for bullish, -1 for bearish) columns.
    """
    df = data.copy()
    if df.empty:
        return df

    # 1) Compute ATR using ta.volatility
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=atr_period
    )
    df['atr'] = atr_indicator.average_true_range()

    # 2) Calculate Basic Upper/Lower Bands
    #    mid = (high + low) / 2
    df['basic_upperband'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
    df['basic_lowerband'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])

    final_upperband = np.zeros(len(df))
    final_lowerband = np.zeros(len(df))

    # 3) Initialize first values
    final_upperband[0] = df['basic_upperband'].iloc[0]
    final_lowerband[0] = df['basic_lowerband'].iloc[0]

    # 4) Compute the "final" bands
    for i in range(1, len(df)):
        # Upper Band
        if df['close'].iloc[i-1] <= final_upperband[i-1]:
            final_upperband[i] = min(df['basic_upperband'].iloc[i], final_upperband[i-1])
        else:
            final_upperband[i] = df['basic_upperband'].iloc[i]

        # Lower Band
        if df['close'].iloc[i-1] >= final_lowerband[i-1]:
            final_lowerband[i] = max(df['basic_lowerband'].iloc[i], final_lowerband[i-1])
        else:
            final_lowerband[i] = df['basic_lowerband'].iloc[i]

    df['final_upperband'] = final_upperband
    df['final_lowerband'] = final_lowerband

    # 5) Determine SuperTrend and direction
    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df))

    # For the 1st row, pick direction arbitrarily based on close vs. final_upperband
    if df['close'].iloc[0] <= df['final_upperband'].iloc[0]:
        supertrend[0] = df['final_upperband'].iloc[0]
        direction[0] = -1  # could also choose +1
    else:
        supertrend[0] = df['final_lowerband'].iloc[0]
        direction[0] = +1

    for i in range(1, len(df)):
        # If close <= final_upperband => we are below upper band => downtrend
        if df['close'].iloc[i] <= df['final_upperband'].iloc[i]:
            supertrend[i] = df['final_upperband'].iloc[i]
            direction[i] = -1
        else:
            # Otherwise, uptrend => supertrend is final_lowerband
            supertrend[i] = df['final_lowerband'].iloc[i]
            direction[i] = +1

    df['supertrend'] = supertrend
    df['supertrend_direction'] = direction

    return df

# ============================
# Trade Logging
# ============================

def record_trade(trade_details):
    """
    Append trade details to 'trade_log' and also write to 'trade_log.json'.
    """
    try:
        trade_log.append(trade_details)
        with open('trade_log.json', 'a') as f:
            json.dump(trade_details, f)
            f.write('\n')
        logger.info(f"Trade recorded: {trade_details}")
        analysis_reports.append(trade_details)
    except Exception as e:
        logger.error(f"Error recording trade: {e}", exc_info=True)

# ============================
# Position Management
# ============================

async def close_position(symbol, trader: AlpacaTrader):
    """
    Close any existing position in the given symbol.
    """
    try:
        position = await trader.get_position(symbol)
        if not position:
            logger.info(f"No open position to close for {symbol}.")
            return
        success = await trader.close_position_direct(symbol)
        if success:
            # Wait a moment and verify the position is closed
            await asyncio.sleep(2)
            pos = await trader.get_position(symbol)
            if pos:
                logger.warning(f"Position for {symbol} still not closed after request.")
            else:
                logger.info(f"Position for {symbol} successfully closed.")
                trade_details = {
                    "timestamp": datetime.now(dt_timezone.utc).isoformat(),
                    "symbol": symbol,
                    "side": f"CLOSE_{position.side.upper()}",
                    "qty": abs(int(float(position.qty))),
                    "filled_avg_price": float(position.avg_entry_price),
                    "filled_qty": abs(int(float(position.qty))),
                    "exit_reason": "SuperTrendSignal"
                }
                record_trade(trade_details)
    except Exception as e:
        logger.error(f"Error closing position for {symbol}: {e}", exc_info=True)

async def enter_position(
    symbol, side, current_price, stop_loss_price, take_profit_price, 
    total_capital, trader: AlpacaTrader
):
    """
    Submit a bracket order (market entry, stop-loss, and take-profit).
    Position size is determined by both risk-per-trade and max allocation constraints.
    """
    cfg = load_config()
    risk_per_trade = cfg.get("RISK_PER_TRADE", 0.01) * total_capital
    risk_per_share = abs(current_price - stop_loss_price)
    if risk_per_share == 0:
        logger.warning(f"Risk per share is 0 for {symbol}, skipping trade.")
        return
    qty_by_risk = int(risk_per_trade / risk_per_share)

    max_allocation_per_trade = cfg.get("MAX_ALLOCATION_PER_TRADE", 0.05) * total_capital
    qty_by_allocation = int(max_allocation_per_trade / current_price)
    qty = min(qty_by_risk, qty_by_allocation)
    if qty <= 0:
        logger.info(f"Calculated qty is 0 for {symbol}, skipping trade.")
        return

    # Check total allocation limit
    total_allocated = sum([abs(float(pos.market_value)) for pos in await trader.list_positions()])
    if total_allocated + (qty * current_price) > cfg.get("MAX_TOTAL_ALLOCATION", 0.5) * total_capital:
        logger.warning(f"Cannot enter position for {symbol}: Allocation limit reached.")
        return

    # Round stop and limit prices to a suitable price increment
    def round_price(price, increment):
        decimals = len(str(increment).split('.')[1]) if '.' in str(increment) else 2
        return round(round(price / increment) * increment, decimals)

    price_increment = 0.01 if current_price >= 1.0 else 0.0001
    stop_loss_price = round_price(stop_loss_price, price_increment)
    take_profit_price = round_price(take_profit_price, price_increment)

    order = await trader.submit_order(
        symbol=symbol,
        qty=qty,
        side=side.lower(),
        type='market',
        time_in_force='gtc',
        order_class='bracket',
        take_profit=dict(limit_price=take_profit_price),
        stop_loss=dict(stop_price=stop_loss_price)
    )
    if order is None:
        logger.error(f"Failed to submit {side.upper()} order for {symbol}.")
        return

    logger.info(f"Submitted {side.upper()} order for {symbol} with SL={stop_loss_price}, TP={take_profit_price}")

    # Wait until order is filled, canceled, or rejected
    while True:
        updated_order = trader.alpaca_rest_client.get_order(order.id)
        if updated_order.status in ["filled", "canceled", "rejected"]:
            break
        await asyncio.sleep(1)

    filled_avg_price = float(updated_order.filled_avg_price or 0.0)
    filled_qty = float(updated_order.filled_qty or 0.0)

    trade_details = {
        "timestamp": datetime.now(dt_timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side.upper(),
        "qty": qty,
        "filled_avg_price": filled_avg_price,
        "filled_qty": filled_qty,
        "entry_price": current_price,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "exit_reason": None  # Will be updated upon closure
    }
    record_trade(trade_details)

# ============================
# Risk Management
# ============================

def check_max_drawdown(account):
    """
    Check if the account's drawdown from the last_equity is above MAX_DRAWDOWN_LIMIT.
    If so, stop trading (in practice, you might systematically reduce risk or alert).
    """
    cfg = load_config()
    try:
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        # Avoid division by zero if last_equity is 0
        if last_equity == 0:
            return False

        drawdown = (last_equity - equity) / last_equity
        if drawdown >= cfg.get("MAX_DRAWDOWN_LIMIT", 0.2):
            logger.critical(f"Maximum drawdown limit reached: {drawdown:.2%}. Stopping trading.")
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking max drawdown: {e}", exc_info=True)
        return False

# ============================
# SuperTrend Signal Logic
# ============================

async def process_symbol(symbol, trader: AlpacaTrader):
    """
    Fetch fresh data for 'symbol', compute SuperTrend, and decide whether to go
    long, short, or exit positions.
    """
    cfg = load_config()
    st_period = cfg.get("SUPER_TREND_PERIOD", 10)
    st_multiplier = cfg.get("SUPER_TREND_MULTIPLIER", 3.0)

    # 1) Fetch historical data
    data = fetch_historical_data(symbol, timeframe='15Min', limit_days=cfg.get("LOOKBACK_DAYS", 180))
    if data.empty or len(data) < st_period:
        logger.warning(f"Not enough data or empty data for {symbol}. Skipping.")
        return

    # 2) Compute SuperTrend (manually)
    data = compute_supertrend(data, st_period, st_multiplier)
    if data.empty or 'supertrend_direction' not in data.columns:
        logger.warning(f"SuperTrend computation failed for {symbol}.")
        return

    # 3) Evaluate the last bar for signals
    last_row = data.iloc[-1]
    st_direction = last_row['supertrend_direction']  # +1 = bullish, -1 = bearish
    current_price = await trader.get_latest_quote(symbol)
    if current_price is None:
        logger.warning(f"No price data for {symbol}.")
        return

    # Check account status & risk
    account = await trader.get_account()
    if not account:
        return
    if check_max_drawdown(account):
        return

    position = await trader.get_position(symbol)
    total_capital = float(account.equity)

    # Adjust these as needed for your risk appetite
    sl_buffer = cfg.get("SUPER_TREND_SL_BUFFER", 0.02)  # Extra offset from ST line
    tp_multiplier = cfg.get("SUPER_TREND_TP_MULTIPLIER", 2.0)  # R:R ratio

    supertrend_line = float(last_row['supertrend'])
    if st_direction == 1.0:
        # BULLISH signal
        # If currently short, close
        if position and position.side == 'short':
            await close_position(symbol, trader)

        # Enter long if no position or was short
        if not position or position.side == 'short':
            stop_loss_price = supertrend_line * (1 - sl_buffer)
            risk_per_share = current_price - stop_loss_price
            take_profit_price = current_price + (risk_per_share * tp_multiplier)
            await enter_position(
                symbol, "BUY", current_price, stop_loss_price, 
                take_profit_price, total_capital, trader
            )

    else:
        # BEARISH signal
        # If currently long, close
        if position and position.side == 'long':
            await close_position(symbol, trader)

        # Enter short if no position or was long, and symbol is shortable
        if (not position or position.side == 'long') and await trader.is_shortable(symbol):
            stop_loss_price = supertrend_line * (1 + sl_buffer)
            risk_per_share = stop_loss_price - current_price
            take_profit_price = current_price - (risk_per_share * tp_multiplier)
            await enter_position(
                symbol, "SELL", current_price, stop_loss_price, 
                take_profit_price, total_capital, trader
            )

# ============================
# Reporting / Analysis
# ============================

def perform_analysis(trade_log: List[Dict]):
    """
    Placeholder for performing trade performance analysis or generating reports.
    """
    pass

# ============================
# Main Symbol Processing
# ============================

async def process_all_symbols(trader: AlpacaTrader, active_symbols: set):
    """
    Scan the symbol list, add/remove symbols, and run SuperTrend logic on each active symbol.
    """
    try:
        cfg = load_config()
        desired_symbols = set(await symbol_scanner.scan())
        logger.info(f"Desired symbols: {desired_symbols}")

        # 1) Handle newly added symbols
        new_symbols = desired_symbols - active_symbols
        if new_symbols:
            logger.info(f"New symbols detected: {new_symbols}")
            for symbol in new_symbols:
                active_symbols.add(symbol)

        # 2) Handle removed symbols
        removed_symbols = active_symbols - desired_symbols
        if removed_symbols:
            for symbol in removed_symbols:
                position = await trader.get_position(symbol)
                if position:
                    await close_position(symbol, trader)
                active_symbols.remove(symbol)

        # 3) Process all active symbols (SuperTrend signals + trades)
        tasks = [process_symbol(symbol, trader) for symbol in active_symbols]
        if tasks:
            await asyncio.gather(*tasks)

        # Perform any analysis or reporting
        perform_analysis(trade_log)

    except Exception as e:
        logger.error(f"Error in process_all_symbols: {e}", exc_info=True)

# ============================
# Main Loop
# ============================

async def main():
    global active_symbols
    await trader.run()

    active_symbols = set()

    # Initial symbol load
    symbols_to_activate = await symbol_scanner.scan()
    if not symbols_to_activate:
        logger.warning("No symbols for trading in the list.")
    else:
        for s in symbols_to_activate:
            active_symbols.add(s)

    # Main loop: continuously process symbols
    while True:
        try:
            account = await trader.get_account()
            if not account:
                await asyncio.sleep(60)
                continue

            await process_all_symbols(trader, active_symbols)
            await asyncio.sleep(60)  # run every 1 minute (adjust as desired)

        except Exception as e:
            logger.error(f"Error in trading bot loop: {e}", exc_info=True)
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down trading bot gracefully.")
        try:
            asyncio.run(trader.close())
        except Exception as e:
            logger.error(f"Error during trader.close(): {e}", exc_info=True)
        exit()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        try:
            asyncio.run(trader.close())
        except Exception as close_e:
            logger.error(f"Error during trader.close() after unhandled exception: {close_e}", exc_info=True)
        exit(1)

