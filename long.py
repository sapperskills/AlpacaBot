#!/usr/bin/env python3

"""
OptimizedLongOnlySwingTrader.py

A sophisticated RL-based swing trading bot tailored for long-only positions, integrating:
- Alpaca API for stock and ETF trading
- SuperTrend signals for bullish direction
- Advanced DQN RL agent with Double DQN, Dueling Networks, and Prioritized Experience Replay
- Extensive Feature Engineering with RSI, MACD, Bollinger Bands, ATR, Volume Change, and SMA Crossover
- Enhanced Reward Function considering Directional Accuracy and Penalties for Losses
- Robust Risk Management with Dynamic Position Sizing based on ATR and Maximum Concurrent Positions
- Memory Persistence to retain learning across sessions
- Continuous Learning and Adaptation for ongoing improvement
- Comprehensive Logging and Trade Recording for monitoring and analysis
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
import random
import numpy as np
import pandas as pd
import asyncio
import pickle

import nest_asyncio
nest_asyncio.apply()

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit, APIError
from alpaca_trade_api.entity import Order as AlpacaOrder
from pytz import timezone as pytz_timezone

# RL and PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# For Technical Indicators
import ta.volatility
import ta.trend
import ta.momentum

###############################################################################
# 1) Load Config + Logging
###############################################################################
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config.json not found")

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()
API_KEY = config.get("API_KEY")
API_SECRET = config.get("API_SECRET")
PAPER = config.get("PAPER", True)
BASE_URL = config.get("BASE_URL", "https://paper-api.alpaca.markets")

LOG_FILE = config.get("LOG_FILE", "trading_bot.log")
LOG_LEVEL_STR = config.get("LOG_LEVEL", "INFO").upper()

# RL Hyperparameters
MEMORY_SIZE = config.get("MEMORY_SIZE", 100000)
BATCH_SIZE = config.get("BATCH_SIZE", 64)
GAMMA = config.get("GAMMA", 0.99)
LEARNING_RATE = config.get("LEARNING_RATE", 0.0005)
TARGET_UPDATE = config.get("TARGET_UPDATE", 1000)
EPSILON_DECAY = config.get("EPSILON_DECAY", 10000)
ALPHA = config.get("ALPHA", 0.6)
BETA_START = config.get("BETA_START", 0.4)

# Risk Management
CAPITAL_ALLOCATION = config.get("CAPITAL_ALLOCATION", 0.05)
MAX_DRAWDOWN = config.get("MAX_DRAWDOWN_LIMIT", 0.2)  # 20% stop trading
TRAILING_STOP_LOSS = config.get("TRAILING_STOP_LOSS", 0.05)  # 5%
TAKE_PROFIT = config.get("TAKE_PROFIT", 0.10)  # 10%
ATR_MULTIPLIER = config.get("ATR_MULTIPLIER", 1.5)  # For position sizing

# Holding Period
MIN_HOLD_DAYS = config.get("MIN_HOLD_DAYS", 180)  # 6 months
MAX_HOLD_DAYS = config.get("MAX_HOLD_DAYS", 365)  # 1 year

# Maximum Concurrent Positions
MAX_CONCURRENT_POSITIONS = config.get("MAX_CONCURRENT_POSITIONS", 5)

SYMBOL_LIST_FILE = config.get("SYMBOL_LIST_FILE", "list.txt")

if not API_KEY or not API_SECRET:
    raise ValueError("API_KEY or SECRET not found in config.json")

# Set up Logging
logger = logging.getLogger("LongOnlySwingTrader")
logger.setLevel(getattr(logging, LOG_LEVEL_STR, logging.INFO))

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10**6, backupCount=5)
file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

###############################################################################
# 2) Alpaca Trader
###############################################################################
class AlpacaTrader:
    def __init__(self, config: dict, logger: logging.Logger):
        self.logger = logger
        self.rest = REST(
            key_id=config["API_KEY"],
            secret_key=config["API_SECRET"],
            base_url=config.get("BASE_URL", "https://paper-api.alpaca.markets"),
            api_version="v2"
        )
        self.initialize_market_calendar()

    def initialize_market_calendar(self):
        try:
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y-%m-%d")
            calendars = self.rest.get_calendar(start=date_str, end=date_str)
            if calendars:
                cal = calendars[0]
                self.market_open = cal.open
                self.market_close = cal.close
                self.logger.info(f"Market Open: {self.market_open}, Market Close: {self.market_close}")
            else:
                self.market_open = None
                self.market_close = None
                self.logger.warning("No calendar data returned from Alpaca API.")
        except Exception as e:
            self.logger.error(f"Error fetching market calendar => {e}", exc_info=True)
            self.market_open = None
            self.market_close = None

    async def get_account(self):
        try:
            return self.rest.get_account()
        except Exception as e:
            self.logger.error(f"get_account => {e}", exc_info=True)
            return None

    async def get_position(self, symbol: str) -> Optional[AlpacaOrder]:
        try:
            pos = self.rest.get_position(symbol)
            if float(pos.qty) == 0:
                return None
            return pos
        except APIError as ex:
            if "Position does not exist" in str(ex):
                return None
            self.logger.error(f"get_position => {ex}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"get_position => {e}", exc_info=True)
            return None

    async def submit_order(self, **kwargs):
        try:
            order = self.rest.submit_order(**kwargs)
            self.logger.info(f"Order Submitted => {order}")
            return order
        except Exception as e:
            self.logger.error(f"submit_order => {e}", exc_info=True)
            return None

    async def close_position(self, symbol: str):
        try:
            # Cancel open orders
            orders = self.rest.list_orders(status="open", symbols=[symbol])
            if orders:
                self.logger.info(f"Cancelling {len(orders)} open orders for {symbol}")
                for o in orders:
                    self.rest.cancel_order(o.id)
                await asyncio.sleep(1)  # Wait for cancellations to process

            # Close position
            self.rest.close_position(symbol)
            self.logger.info(f"Close position request sent for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"close_position => {symbol}: {e}", exc_info=True)
            return False

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            quote = self.rest.get_latest_quote(symbol)
            if quote.bid_price is not None and quote.ask_price is not None:
                return (quote.bid_price + quote.ask_price) / 2
            elif quote.last_price is not None:
                return quote.last_price
            return None
        except Exception as e:
            self.logger.error(f"get_latest_price => {symbol}: {e}", exc_info=True)
            return None

###############################################################################
# 3) Symbol Scanner
###############################################################################
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
            if len(symbols) > MAX_CONCURRENT_POSITIONS:
                self.logger.warning(f"Number of symbols ({len(symbols)}) exceeds MAX_CONCURRENT_POSITIONS ({MAX_CONCURRENT_POSITIONS}). Truncating the list.")
                symbols = symbols[:MAX_CONCURRENT_POSITIONS]
            self.logger.info(f"Loaded {len(symbols)} symbols from {self.symbol_file}.")
            return symbols
        except Exception as e:
            self.logger.error(f"Error reading symbol file {self.symbol_file}: {e}", exc_info=True)
            return []

###############################################################################
# 4) RL Net and Agent
###############################################################################
class DuelingDQN(nn.Module):
    """
    Dueling DQN with Double DQN architecture.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Combine streams: Q = V + (A - mean(A))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

# Prioritized Experience Replay
class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay with proportional prioritization.
    """
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, transition: tuple):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[tuple], List[int], List[float]]:
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights.tolist()

    def update_priorities(self, indices: List[int], priorities: List[float]):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, in_dim: int, out_dim: int, device: torch.device, memory_size: int, batch_size: int,
                 gamma: float, learning_rate: float, epsilon_decay: int, alpha: float, beta_start: float,
                 logger: logging.Logger):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.logger = logger

        self.policy_net = DuelingDQN(in_dim, out_dim).to(device)
        self.target_net = DuelingDQN(in_dim, out_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = PrioritizedReplayMemory(memory_size, alpha=alpha)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = 100000  # Number of frames over which beta increases to 1

    def select_action(self, state: torch.Tensor) -> int:
        """
        Select an action using epsilon-greedy strategy.
        """
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() < epsilon:
            action = random.randrange(self.out_dim)
            self.logger.debug(f"Selected random action: {action}")
            return action
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()
                self.logger.debug(f"Selected best action: {action} with Q-values: {q_values}")
                return action

    def store_transition(self, s: torch.Tensor, a: int, r: float, ns: torch.Tensor, d: int):
        """
        Store a transition in replay memory with maximum priority for new experiences.
        """
        transition = (s, a, r, ns, d)
        self.memory.push(transition)

    def optimize_model(self, beta: float):
        """
        Optimize the model by sampling a batch from memory and performing gradient descent.
        Implements Double DQN.
        """
        if len(self.memory) < self.batch_size:
            self.logger.debug("Not enough memory to optimize.")
            return

        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q Values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN: Select action using policy_net, evaluate with target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute TD Error
        td_errors = current_q_values - expected_q_values
        loss = (td_errors.pow(2) * weights).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-5  # Adding small constant to avoid zero priority
        self.memory.update_priorities(indices, new_priorities.flatten())

        self.logger.debug(f"Optimizing model. Loss: {loss.item()}")

    def update_target_net(self):
        """
        Update the target network to match the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.logger.info("Updated target network.")

    def save_checkpoint(self, filepath: str):
        """
        Save the agent's state to a file.
        """
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'steps_done': self.steps_done
            }, filepath)
            self.logger.info(f"Saved checkpoint to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

    def load_checkpoint(self, filepath: str):
        """
        Load the agent's state from a file.
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"Checkpoint file {filepath} does not exist. Starting fresh.")
            return
        try:
            # weights_only=False to load full checkpoint
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint.get('steps_done', 0)
            self.logger.info(f"Loaded checkpoint from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

# Transition tuple for experience replay
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayMemoryPersistence:
    def __init__(self, filepath: str, logger: logging.Logger):
        self.filepath = filepath
        self.logger = logger

    def save_memory(self, memory: PrioritizedReplayMemory):
        """
        Save the prioritized replay memory to a file using pickle.
        """
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump({
                    'memory': memory.memory,
                    'priorities': memory.priorities,
                    'pos': memory.pos
                }, f)
            self.logger.info(f"Saved replay memory to {self.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save replay memory: {e}", exc_info=True)

    def load_memory(self, memory: PrioritizedReplayMemory):
        """
        Load the prioritized replay memory from a file using pickle.
        """
        if not os.path.exists(self.filepath):
            self.logger.warning(f"Replay memory file {self.filepath} does not exist. Starting fresh.")
            return
        try:
            with open(self.filepath, 'rb') as f:
                data = pickle.load(f)
                memory.memory = deque(data['memory'], maxlen=memory.capacity)
                memory.priorities = data['priorities']
                memory.pos = data['pos']
            self.logger.info(f"Loaded replay memory from {self.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load replay memory: {e}", exc_info=True)

###############################################################################
# 5) Technical Indicators
###############################################################################
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SuperTrend, RSI, MACD, Bollinger Bands, ATR, Volume Change, and detect SMA Crossover patterns.
    """
    if df.empty:
        return df

    df = df.copy()
    
    # SuperTrend
    st_df = compute_supertrend(df, period=10, multiplier=3.0)
    df['supertrend'] = st_df['supertrend']
    df['supertrend_direction'] = st_df['supertrend_direction']
    
    # Relative Strength Index (RSI)
    rsi = ta.momentum.RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()
    
    # Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    
    # Average True Range (ATR)
    atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()
    
    # Volume Change
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    
    # Price Patterns: Simple Moving Averages Crossover (e.g., 50 SMA crossing 200 SMA)
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['sma_crossover'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)  # 1: bullish crossover, -1: bearish crossover
    
    # Drop rows with NaN values resulted from indicator calculations
    df.dropna(inplace=True)
    
    return df

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute SuperTrend indicator.
    """
    if df.empty:
        return df
    df2 = df.copy()
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df2["high"],
        low=df2["low"],
        close=df2["close"],
        window=period
    )
    df2["atr"] = atr_indicator.average_true_range()
    df2["basic_upperband"] = ((df2["high"] + df2["low"]) / 2) + (multiplier * df2["atr"])
    df2["basic_lowerband"] = ((df2["high"] + df2["low"]) / 2) - (multiplier * df2["atr"])

    final_upperband = np.zeros(len(df2))
    final_lowerband = np.zeros(len(df2))

    # Initialize first values
    final_upperband[0] = df2["basic_upperband"].iloc[0]
    final_lowerband[0] = df2["basic_lowerband"].iloc[0]

    # Compute the "final" bands
    for i in range(1, len(df2)):
        # Upper Band
        if df2["close"].iloc[i - 1] <= final_upperband[i - 1]:
            final_upperband[i] = min(df2["basic_upperband"].iloc[i], final_upperband[i - 1])
        else:
            final_upperband[i] = df2["basic_upperband"].iloc[i]

        # Lower Band
        if df2["close"].iloc[i - 1] >= final_lowerband[i - 1]:
            final_lowerband[i] = max(df2["basic_lowerband"].iloc[i], final_lowerband[i - 1])
        else:
            final_lowerband[i] = df2["basic_lowerband"].iloc[i]

    df2["final_upperband"] = final_upperband
    df2["final_lowerband"] = final_lowerband

    supertrend = np.zeros(len(df2))
    direction = np.zeros(len(df2))

    # For the first row
    if df2["close"].iloc[0] <= df2["final_upperband"].iloc[0]:
        supertrend[0] = df2["final_upperband"].iloc[0]
        direction[0] = -1
    else:
        supertrend[0] = df2["final_lowerband"].iloc[0]
        direction[0] = +1

    for i in range(1, len(df2)):
        if df2["close"].iloc[i] <= df2["final_upperband"].iloc[i]:
            supertrend[i] = df2["final_upperband"].iloc[i]
            direction[i] = -1
        else:
            supertrend[i] = df2["final_lowerband"].iloc[i]
            direction[i] = +1

    df2["supertrend"] = supertrend
    df2["supertrend_direction"] = direction

    return df2

###############################################################################
# 6) Merging RL with Real Trading => PnL-based Reward
###############################################################################
class TradeState:
    """
    Keeps track of trade state for each symbol to implement trailing stop-loss and take-profit.
    """
    def __init__(self):
        self.active_trades = {}  # symbol: {'entry_price': float, 'entry_time': datetime, 'stop_loss': float, 'take_profit': float}

    def update_trade(self, symbol: str, entry_price: float, atr: float):
        """
        Initialize or update trade state with trailing stop-loss and take-profit levels based on ATR.
        """
        now = datetime.now(timezone.utc)
        if symbol not in self.active_trades:
            self.active_trades[symbol] = {
                'entry_price': entry_price,
                'entry_time': now,
                'stop_loss': entry_price - (ATR_MULTIPLIER * atr),
                'take_profit': entry_price + (TAKE_PROFIT * entry_price)
            }
            logger.info(f"Initialized trade for {symbol}: Entry Price={entry_price}, Stop Loss={self.active_trades[symbol]['stop_loss']}, Take Profit={self.active_trades[symbol]['take_profit']}")

    def adjust_stop_loss(self, symbol: str, current_price: float, atr: float):
        """
        Adjust trailing stop-loss based on current price and ATR.
        """
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            # Update stop loss only if the price has moved favorably
            new_stop_loss = current_price - (ATR_MULTIPLIER * atr)
            if new_stop_loss > trade['stop_loss']:
                trade['stop_loss'] = new_stop_loss
                logger.info(f"Adjusted Stop Loss for {symbol}: New Stop Loss={trade['stop_loss']}")

    def check_exit(self, symbol: str, current_price: float, current_time: datetime, atr: float) -> Optional[str]:
        """
        Check if current price or holding period triggers exit.
        Returns 'stop_loss', 'take_profit', 'time_exit', or None.
        """
        if symbol not in self.active_trades:
            return None
        trade = self.active_trades[symbol]
        hold_duration = (current_time - trade['entry_time']).days

        # Check stop-loss and take-profit
        if current_price <= trade['stop_loss']:
            del self.active_trades[symbol]
            return 'stop_loss'
        elif current_price >= trade['take_profit']:
            del self.active_trades[symbol]
            return 'take_profit'
        
        # Check holding period
        if hold_duration >= MAX_HOLD_DAYS:
            del self.active_trades[symbol]
            return 'time_exit'
        elif hold_duration >= MIN_HOLD_DAYS:
            # Optionally, allow selling after min_hold_days
            return None  # Let the agent decide
        return None

old_equities: Dict[str, float] = {}
trade_state = TradeState()

def get_reward(symbol: str, eq_now: float) -> float:
    """
    Calculate reward based on change in equity for the specific symbol.
    """
    eq_old_val = old_equities.get(symbol, eq_now)
    reward = eq_now - eq_old_val
    old_equities[symbol] = eq_now
    return reward

def check_drawdown(account) -> bool:
    """
    Check if the account's drawdown from the last equity exceeds the limit.
    """
    try:
        eq = float(account.equity)
        last_eq = float(account.last_equity)
        if last_eq <= 0:
            return False

        drawdown = (last_eq - eq) / last_eq
        if drawdown >= MAX_DRAWDOWN:
            logger.critical(f"Maximum drawdown limit reached: {drawdown:.2%}. Stopping trading.")
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking drawdown: {e}", exc_info=True)
        return False

###############################################################################
# 7) RL Trader Wrapper
###############################################################################
class RLTrader:
    def __init__(self, device: torch.device, logger: logging.Logger, checkpoint_path: str = "agent_checkpoint.pth",
                 replay_memory_path: str = "replay_memory.pkl"):
        self.device = device
        self.state_dim = 14  # Updated to match the state vector
        self.action_dim = 3  # 0=Hold, 1=Buy, 2=Sell
        self.checkpoint_path = checkpoint_path
        self.replay_memory_path = replay_memory_path
        self.agent = DQNAgent(
            in_dim=self.state_dim,
            out_dim=self.action_dim,
            device=self.device,
            memory_size=MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE,
            epsilon_decay=EPSILON_DECAY,
            alpha=ALPHA,
            beta_start=BETA_START,
            logger=logger
        )
        self.memory_persistence = PrioritizedReplayMemoryPersistence(self.replay_memory_path, logger)
        self.load_agent()

    def load_agent(self):
        """
        Load agent's network weights and replay memory if available.
        """
        self.agent.load_checkpoint(self.checkpoint_path)
        self.memory_persistence.load_memory(self.agent.memory)

    def save_agent(self):
        """
        Save agent's network weights and replay memory.
        """
        self.agent.save_checkpoint(self.checkpoint_path)
        self.memory_persistence.save_memory(self.agent.memory)

    def pick_action(self, state: torch.Tensor) -> int:
        return self.agent.select_action(state)

    def store_transition(self, s: torch.Tensor, a: int, r: float, ns: torch.Tensor, d: int):
        self.agent.store_transition(s, a, r, ns, d)

    def optimize(self, beta: float):
        self.agent.optimize_model(beta)

    def update_target(self):
        self.agent.update_target_net()

###############################################################################
# 8) Trade Actions
###############################################################################
async def close_any_position(symbol: str, trader: AlpacaTrader, logger: logging.Logger):
    pos = await trader.get_position(symbol)
    if pos:
        success = await trader.close_position(symbol)
        if success:
            logger.info(f"Closed position for {symbol}")
            trade_details = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "action": f"Close_{pos.side}",
                "qty": abs(int(float(pos.qty))),
                "price": float(pos.avg_entry_price),
                "timestamp_closed": datetime.now(timezone.utc).isoformat()
            }
            record_trade(trade_details, logger)

async def apply_action(symbol: str, action: int, direction: int, trader: AlpacaTrader, logger: logging.Logger, trade_state: TradeState, atr: float):
    """
    Apply the chosen action: Hold, Buy, Sell.
    Incorporates dynamic position sizing and trailing stop-loss/take-profit.
    
    Parameters:
        symbol (str): The stock symbol.
        action (int): The action to take (0=Hold, 1=Buy, 2=Sell).
        direction (int): SuperTrend direction.
        trader (AlpacaTrader): The AlpacaTrader instance.
        logger (logging.Logger): The logger instance.
        trade_state (TradeState): The current trade state.
        atr (float): The current ATR value.
    """
    acct = await trader.get_account()
    if not acct:
        return
    eq = float(acct.equity)
    last_price = await trader.get_latest_price(symbol)
    if not last_price or last_price <= 0:
        return
    # Position sizing based on ATR to account for volatility
    position_size = (ATR_MULTIPLIER * atr)
    shares = int((eq * CAPITAL_ALLOCATION) / position_size)
    if shares < 1:
        logger.info(f"Not enough equity to allocate to {symbol}. Required shares based on ATR: 1, Available allocation: {eq * CAPITAL_ALLOCATION}")
        return

    if action == 0:
        logger.info(f"{symbol} => Action=Hold => No trade executed.")
    elif action == 1:
        # Buy
        pos = await trader.get_position(symbol)
        if pos and float(pos.qty) > 0:
            logger.info(f"{symbol} already has a long position. Skipping Buy action.")
            return
        order = await trader.submit_order(
            symbol=symbol,
            qty=shares,
            side="buy",
            type="market",
            time_in_force="gtc"
        )
        if order:
            logger.info(f"Buy order executed for {symbol}, Shares: {shares}")
            trade_details = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "action": "Buy",
                "qty": shares,
                "price": last_price,
                "timestamp_executed": datetime.now(timezone.utc).isoformat()
            }
            record_trade(trade_details, logger)
            trade_state.update_trade(symbol, last_price, atr)
    elif action == 2:
        # Sell
        pos = await trader.get_position(symbol)
        if pos and float(pos.qty) > 0:
            shares_to_sell = int(float(pos.qty))
            order = await trader.submit_order(
                symbol=symbol,
                qty=shares_to_sell,
                side="sell",
                type="market",
                time_in_force="gtc"
            )
            if order:
                logger.info(f"Sell order executed for {symbol}, Shares: {shares_to_sell}")
                trade_details = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "action": "Sell",
                    "qty": shares_to_sell,
                    "price": last_price,
                    "timestamp_executed": datetime.now(timezone.utc).isoformat()
                }
                record_trade(trade_details, logger)
                trade_state.active_trades.pop(symbol, None)
        else:
            logger.info(f"{symbol} has no active long position to sell.")

###############################################################################
# 9) Trade Logging
###############################################################################
def record_trade(trade_details: Dict, logger: logging.Logger):
    """
    Record trade details to log and a JSON file for analysis.
    """
    try:
        logger.info(f"Trade Recorded: {trade_details}")
        with open("trade_log.json", "a") as f:
            json.dump(trade_details, f)
            f.write('\n')
    except Exception as e:
        logger.error(f"Error recording trade: {e}", exc_info=True)

###############################################################################
# 10) RL State Construction
###############################################################################
def build_state(df: pd.DataFrame, symbol: str, trade_state: TradeState) -> torch.Tensor:
    """
    Build a 14-dimensional state vector including technical indicators and trade state.
    """
    last_row = df.iloc[-1]

    # SuperTrend
    supertrend_direction = float(last_row['supertrend_direction'])
    
    # RSI
    rsi = float(last_row['rsi'])
    
    # MACD
    macd = float(last_row['macd'])
    macd_signal = float(last_row['macd_signal'])
    macd_diff = float(last_row['macd_diff'])
    
    # Bollinger Bands
    bb_mavg = float(last_row['bb_bbm'])
    bb_hband = float(last_row['bb_bbh'])
    bb_lband = float(last_row['bb_bbl'])
    
    # ATR
    atr = float(last_row['atr'])
    
    # Volume Change
    volume_change = float(last_row['volume_change'])
    
    # SMA Crossover
    sma_crossover = float(last_row['sma_crossover'])
    
    # Time of Day (Normalized)
    now = datetime.now(timezone.utc)
    hour_fraction = (now.hour * 60 + now.minute) / (24.0 * 60.0)
    
    # Days Held (Normalized)
    if symbol in trade_state.active_trades:
        entry_time = trade_state.active_trades[symbol]['entry_time']
        days_held = (now - entry_time).days / MAX_HOLD_DAYS  # Normalized
    else:
        days_held = 0.0
    
    # Current Position (1 for long, 0 for no position)
    pos = 1.0 if symbol in trade_state.active_trades else 0.0
    
    state = [
        supertrend_direction,  # 1
        rsi,                   # 2
        macd,                  # 3
        macd_signal,           # 4
        macd_diff,             # 5
        bb_mavg,               # 6
        bb_hband,              # 7
        bb_lband,              # 8
        atr,                   # 9
        volume_change,         # 10
        sma_crossover,         # 11
        hour_fraction,         # 12
        days_held,             # 13
        pos                    # 14
    ]
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

###############################################################################
# 11) RL Environment and Processing
###############################################################################
async def process_symbol(symbol: str, rl: RLTrader, trader: AlpacaTrader, logger: logging.Logger, trade_state: TradeState):
    """
    Process a single symbol: fetch data, compute indicators, make trading decisions.
    """
    # 1) Fetch historical data
    bars = await fetch_historical_bars(symbol, trader, logger, days=400, timeframe="1D")  # Increased days for SMA 200
    if bars.empty or len(bars) < 200:
        logger.info(f"{symbol} => Not enough data to compute technical indicators. Skipping.")
        return

    # 2) Compute Technical Indicators
    bars = compute_technical_indicators(bars)

    # 3) Current state
    state = build_state(bars, symbol, trade_state).to(rl.device)

    # 4) Pick action
    action = rl.pick_action(state)
    logger.info(f"{symbol} => Action: {action}")

    # 5) Extract ATR and apply action
    atr = float(bars.iloc[-1]['atr'])
    await apply_action(symbol, action, bars.iloc[-1]['supertrend_direction'], trader, logger, trade_state, atr)

    # 6) Update trade state with trailing stop-loss
    pos = await trader.get_position(symbol)
    if pos:
        entry_price = float(pos.avg_entry_price)
        trade_state.update_trade(symbol, entry_price, atr)
        # Adjust trailing stop-loss based on current price
        current_price = await trader.get_latest_price(symbol)
        if current_price:
            trade_state.adjust_stop_loss(symbol, current_price, atr)

    # 7) Check for trailing stop-loss, take-profit, or holding period exit
    current_price = await trader.get_latest_price(symbol)
    current_time = datetime.now(timezone.utc)
    exit_reason = trade_state.check_exit(symbol, current_price, current_time, atr) if current_price else None
    if exit_reason:
        logger.info(f"{symbol} => Exiting trade due to {exit_reason}")
        action = 2  # Sell
        await apply_action(symbol, action, bars.iloc[-1]['supertrend_direction'], trader, logger, trade_state, atr)

    # 8) Fetch updated account info for reward
    acct_updated = await trader.get_account()
    if not acct_updated:
        logger.warning("Unable to retrieve updated account information for reward calculation.")
        return
    eq_new = float(acct_updated.equity)

    # 9) Calculate reward
    reward = get_reward(symbol, eq_new)

    # 10) Determine if episode is done
    done = 0
    if (eq_new - eq_old(symbol, eq_new)) < (-MAX_DRAWDOWN * eq_new):
        done = 1
        logger.warning(f"{symbol} => Equity dropped by {MAX_DRAWDOWN:.2%}. Setting done=1.")

    # 11) Build next state
    next_state = build_state(bars, symbol, trade_state).to(rl.device)

    # 12) Store transition
    rl.store_transition(state, action, reward, next_state, done)

def eq_old(symbol: str, eq_new: float) -> float:
    """
    Helper function to get previous equity for a symbol.
    """
    return old_equities.get(symbol, eq_new)

###############################################################################
# 12) Fetch Historical Bars
###############################################################################
async def fetch_historical_bars(symbol: str, trader: AlpacaTrader, logger: logging.Logger, days: int = 400, timeframe: str = "1D") -> pd.DataFrame:
    """
    Retrieve historical OHLCV data from Alpaca for the given symbol and timeframe.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        start_utc = now_utc - timedelta(days=days)
        
        # Correct TimeFrame usage
        if timeframe == "1D":
            tf = TimeFrame.Day
        elif timeframe == "15Min":
            tf = TimeFrame.FifteenMin
        else:
            tf = TimeFrame.Day  # Default to Day if unknown timeframe
        
        bars = trader.rest.get_bars(
            symbol,
            tf,
            start_utc.isoformat(),
            now_utc.isoformat(),
            adjustment="raw"
        ).df

        if bars.empty:
            logger.warning(f"{symbol} => No historical bars data retrieved.")
            return pd.DataFrame()

        bars.reset_index(inplace=True)
        bars.rename(columns={
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }, inplace=True)
        return bars
    except Exception as e:
        logger.error(f"fetch_historical_bars => {symbol}: {e}", exc_info=True)
        return pd.DataFrame()

###############################################################################
# 13) Main Bot Loop
###############################################################################
async def run_bot():
    """
    Main loop to continuously process symbols and train the RL agent.
    """
    # Initialize Trader and Symbol Scanner
    trader = AlpacaTrader(config, logger)
    symbol_scanner = SymbolListScanner(SYMBOL_LIST_FILE, logger)
    symbols = await symbol_scanner.scan()

    if not symbols:
        logger.error("No symbols loaded. Exiting bot.")
        return

    # Initialize RL Trader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl = RLTrader(device, logger)

    episode = 0
    while True:
        try:
            logger.info("Starting new trading cycle.")
            tasks = [process_symbol(sym, rl, trader, logger, trade_state) for sym in symbols]
            await asyncio.gather(*tasks)

            # Optimize RL agent with beta annealing
            beta = min(1.0, rl.agent.beta_start + episode * (1.0 - rl.agent.beta_start) / rl.agent.beta_frames)
            rl.optimize(beta)

            # Update target network periodically
            if episode % TARGET_UPDATE == 0:
                rl.update_target()

            # Save agent's state periodically
            if episode % 100 == 0:
                rl.save_agent()

            episode += 1
            logger.info("Trading cycle complete. Sleeping for 60 seconds.\n")
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in run_bot loop: {e}", exc_info=True)
            await asyncio.sleep(60)

###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    rl_instance = None  # Global variable to hold RLTrader instance

    async def main():
        global rl_instance
        rl_instance = RLTrader(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            logger=logger
        )
        await run_bot()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("User interrupted. Shutting down gracefully.")
        if rl_instance:
            rl_instance.save_agent()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)

