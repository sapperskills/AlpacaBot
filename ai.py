#!/usr/bin/env python3

"""
OptimizedLongShortSwingTrader.py

A sophisticated RL-based swing trading bot tailored for both long and short positions,
now enhanced with a GPT-based controller that can override RL actions
to focus on profits, detect patterns, and adapt the trading strategy dynamically.

Key Features:
- Uses Alpaca API for stock/ETF trading
- Integrates SuperTrend, RSI, MACD, Bollinger Bands, SMA Crossover, ATR, volume change
- Advanced DQN RL agent (Double DQN, Dueling, Prioritized Replay)
- GPT override for action decisions, plus optional hyperparameter updates
- Logging, memory persistence, trailing stops, multi-symbol approach
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

# Alpaca
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit, APIError
from alpaca_trade_api.entity import Order as AlpacaOrder
from pytz import timezone as pytz_timezone

# RL and PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# Technical Indicators
import ta.volatility
import ta.trend
import ta.momentum

# GPT / Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

###############################################################################
# 0) GPT Integration
###############################################################################
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

# Load GPT model & tokenizer once at the start
device_gpt = "cuda" if torch.cuda.is_available() else "cpu"
logger_gpt_init = logging.getLogger("GPTInit")
try:
    logger_gpt_init.info("Loading GPT model...")
    # Force float32, disable 8-bit/4-bit to avoid 'fp8' quantization errors
    gpt_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    gpt_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Force standard float32
        load_in_8bit=False,
        load_in_4bit=False
    )
    gpt_model.eval().to(device_gpt)
    logger_gpt_init.info(f"GPT model {MODEL_NAME} loaded successfully.")
except Exception as e:
    logger_gpt_init.warning(f"Failed to load GPT model: {e}")
    gpt_tokenizer = None
    gpt_model = None

def generate_text(prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """
    Generate text from the GPT model given a prompt.
    """
    if (gpt_tokenizer is None) or (gpt_model is None):
        return "ACTION: HOLD"  # fallback if model can't load
    inputs = gpt_tokenizer(prompt, return_tensors="pt").to(device_gpt)
    with torch.no_grad():
        outputs = gpt_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    return gpt_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

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

# Holding Period
MIN_HOLD_DAYS = config.get("MIN_HOLD_DAYS", 180)  # 6 months
MAX_HOLD_DAYS = config.get("MAX_HOLD_DAYS", 365)  # 1 year

SYMBOL_LIST_FILE = config.get("SYMBOL_LIST_FILE", "list.txt")

if not API_KEY or not API_SECRET:
    raise ValueError("API_KEY or SECRET not found in config.json")

# Set up Logging
logger = logging.getLogger("LongShortSwingTrader")
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
                # Changed to DEBUG instead of ERROR
                self.logger.debug(f"No active position for {symbol}")
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
            if len(symbols) != 10:
                self.logger.warning(
                    f"Symbol list should contain exactly 10 symbols (5 stocks and 5 ETFs). "
                    f"Current count: {len(symbols)}"
                )
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
    Dueling DQN with Double DQN architecture and Specialized Clusters.
    """
    def __init__(self, in_dim: int, out_dim: int, num_clusters: int):
        super(DuelingDQN, self).__init__()
        self.shared_feature = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Specialized Clusters for Each Symbol
        self.cluster_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        ) for _ in range(num_clusters)])
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x, cluster_id):
        features = self.shared_feature(x)
        cluster_features = self.cluster_layers[cluster_id](features)
        values = self.value_stream(cluster_features)
        advantages = self.advantage_stream(cluster_features)
        # Combine streams: Q = V + (A - mean(A))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

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

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done', 'cluster_id'))

class DQNAgent:
    def __init__(self, in_dim: int, out_dim: int, num_clusters: int, device: torch.device,
                 memory_size: int, batch_size: int, gamma: float, learning_rate: float,
                 epsilon_decay: int, alpha: float, beta_start: float, logger: logging.Logger):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_clusters = num_clusters
        self.device = device
        self.logger = logger

        self.policy_net = DuelingDQN(in_dim, out_dim, num_clusters).to(device)
        self.target_net = DuelingDQN(in_dim, out_dim, num_clusters).to(device)
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

        # Keep track of hyperparams for GPT adjustments, if desired
        self.hyperparams = {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_decay": epsilon_decay,
        }

    def select_action(self, state: torch.Tensor, cluster_id: int) -> int:
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
                q_values = self.policy_net(state, cluster_id)
                action = q_values.argmax(dim=1).item()
                self.logger.debug(f"Selected best action: {action} with Q-values: {q_values}")
                return action

    def store_transition(self, s: torch.Tensor, a: int, r: float, ns: torch.Tensor,
                         d: int, cluster_id: int):
        transition = (s, a, r, ns, d, cluster_id)
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
        cluster_ids = torch.tensor(batch.cluster_id, dtype=torch.long).to(self.device)

        # Current Q Values
        current_q_values = self.policy_net(state_batch, cluster_ids).gather(1, action_batch)

        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch, cluster_ids).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state_batch, cluster_ids).gather(1, next_actions)
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute TD Error
        td_errors = current_q_values - expected_q_values
        loss = (td_errors.pow(2) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities.flatten())

        self.logger.debug(f"Optimizing model. Loss: {loss.item()}")

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.logger.info("Updated target network.")

    def save_checkpoint(self, filepath: str):
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
        if not os.path.exists(filepath):
            self.logger.warning(f"Checkpoint file {filepath} does not exist. Starting fresh.")
            return
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint.get('steps_done', 0)
            self.logger.info(f"Loaded checkpoint from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

    def update_hyperparam(self, name: str, value):
        """
        Example method for GPT to adjust RL hyperparameters at runtime.
        """
        if name == "learning_rate":
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = float(value)
            self.hyperparams["learning_rate"] = float(value)
            self.logger.info(f"Updated learning_rate to {value}")
        elif name == "gamma":
            self.gamma = float(value)
            self.hyperparams["gamma"] = float(value)
            self.logger.info(f"Updated gamma to {value}")
        elif name == "epsilon_decay":
            self.epsilon_decay = float(value)
            self.hyperparams["epsilon_decay"] = float(value)
            self.logger.info(f"Updated epsilon_decay to {value}")
        else:
            self.logger.warning(f"Hyperparam {name} not recognized. No update performed.")

###############################################################################
# 5) GPT Controller Class
###############################################################################
class GPTController:
    """
    Provides a prompt to GPT that includes:
    - Current RL agent's recommended action
    - Basic environment/symbol info
    - RL hyperparameters
    Then parses GPT's response for a final action override + optional hyperparam changes.
    """

    def __init__(self, agent: DQNAgent, logger: logging.Logger):
        self.agent = agent
        self.logger = logger

    def build_prompt_for_gpt(self, symbol: str, rl_action: str, state_info: dict):
        hyper_json = json.dumps(self.agent.hyperparams, indent=2)

        prompt = f"""
You are a trading decision AI with permission to override an RL agent's actions to maximize profits.
Symbol: {symbol}
RL Action suggestion: {rl_action}
Technical/State Info: {state_info}

Current RL hyperparameters:
{hyper_json}

Output format instructions:
1) Must include exactly one line "ACTION: BUY" or "ACTION: SELL" or "ACTION: HOLD".
2) Optionally, any number of lines starting with "HPARAM:" to indicate a hyperparameter update,
   e.g. "HPARAM: learning_rate=0.0001".

Be concise, focusing on profit and pattern detection.
"""
        return prompt

    def parse_gpt_output(self, gpt_output):
        """
        Parse lines looking for:
        - "ACTION: X"
        - "HPARAM: name=value"
        """
        final_action = "HOLD"
        hyperparam_changes = []

        lines = gpt_output.strip().split("\n")
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.upper().startswith("ACTION:"):
                # e.g. "ACTION: BUY"
                parts = line_stripped.split(":", 1)
                if len(parts) > 1:
                    chosen = parts[1].strip().upper()
                    if chosen in ["BUY", "SELL", "HOLD"]:
                        final_action = chosen

            elif line_stripped.upper().startswith("HPARAM:"):
                # e.g. "HPARAM: learning_rate=0.00005"
                splitted = line_stripped.split("=", 1)
                if len(splitted) == 2:
                    left_side = splitted[0].replace("HPARAM:", "").strip()
                    param_value = splitted[1].strip()
                    param_name = left_side
                    hyperparam_changes.append((param_name, param_value))

        return final_action, hyperparam_changes

    def override_action(self, symbol: str, rl_action: str, state_info: dict):
        # Build the prompt
        prompt = self.build_prompt_for_gpt(symbol, rl_action, state_info)
        self.logger.debug(f"\n[GPT Prompt]\n{prompt}\n")

        # Generate from GPT
        gpt_response = generate_text(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9)
        self.logger.info(f"[GPT Response]\n{gpt_response}\n")

        # Parse the response
        final_action, param_changes = self.parse_gpt_output(gpt_response)

        # Apply hyperparameter changes
        for (name, val_str) in param_changes:
            try:
                val_float = float(val_str)
                self.agent.update_hyperparam(name, val_float)
            except ValueError:
                self.logger.warning(f"Could not parse hyperparam value: {val_str}")

        return final_action

###############################################################################
# 6) Merging RL with Real Trading => PnL-based Reward
###############################################################################
class TradeState:
    """
    Keeps track of trade state for each symbol to implement trailing stop-loss and take-profit.
    """
    def __init__(self):
        self.active_trades = {}  # symbol -> { entry_price, entry_time, stop_loss, take_profit }

    def update_trade(self, symbol: str, entry_price: float):
        now = datetime.now(timezone.utc)
        if symbol not in self.active_trades:
            self.active_trades[symbol] = {
                'entry_price': entry_price,
                'entry_time': now,
                'stop_loss': entry_price * (1 - TRAILING_STOP_LOSS),
                'take_profit': entry_price * (1 + TAKE_PROFIT)
            }

    def adjust_stop_loss(self, symbol: str, current_price: float):
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            if current_price > trade['entry_price']:
                new_sl = current_price * (1 - TRAILING_STOP_LOSS)
                if new_sl > trade['stop_loss']:
                    trade['stop_loss'] = new_sl

    def check_exit(self, symbol: str, current_price: float,
                   current_time: datetime, min_hold_days: int, max_hold_days: int) -> Optional[str]:
        if symbol not in self.active_trades:
            return None
        trade = self.active_trades[symbol]
        hold_duration = (current_time - trade['entry_time']).days

        if current_price <= trade['stop_loss']:
            del self.active_trades[symbol]
            return 'stop_loss'
        elif current_price >= trade['take_profit']:
            del self.active_trades[symbol]
            return 'take_profit'
        if hold_duration >= max_hold_days:
            del self.active_trades[symbol]
            return 'time_exit'
        return None

old_equities: Dict[str, float] = {}
trade_state = TradeState()

def get_reward(symbol: str, eq_now: float) -> float:
    eq_old_val = old_equities.get(symbol, eq_now)
    reward = eq_now - eq_old_val
    old_equities[symbol] = eq_now
    return reward

def check_drawdown(account) -> bool:
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
class PrioritizedReplayMemoryPersistence:
    def __init__(self, filepath: str, logger: logging.Logger):
        self.filepath = filepath
        self.logger = logger

    def save_memory(self, memory: PrioritizedReplayMemory):
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

class RLTrader:
    def __init__(self, device: torch.device, logger: logging.Logger,
                 checkpoint_path: str = "agent_checkpoint.pth",
                 replay_memory_path: str = "replay_memory.pkl"):
        self.device = device
        self.logger = logger
        self.state_dim = 14  # technical indicators + position info
        self.action_dim = 3  # 0=Hold, 1=Buy, 2=Sell
        self.checkpoint_path = checkpoint_path
        self.replay_memory_path = replay_memory_path
        self.num_clusters = 10  # cluster per symbol

        self.agent = DQNAgent(
            in_dim=self.state_dim,
            out_dim=self.action_dim,
            num_clusters=self.num_clusters,
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

        # GPT controller to override RL actions
        self.gpt_controller = GPTController(self.agent, logger)

    def load_agent(self):
        self.agent.load_checkpoint(self.checkpoint_path)
        self.memory_persistence.load_memory(self.agent.memory)

    def save_agent(self):
        self.agent.save_checkpoint(self.checkpoint_path)
        self.memory_persistence.save_memory(self.agent.memory)

    def pick_action(self, state: torch.Tensor, cluster_id: int, symbol: str, extra_info: dict) -> str:
        """
        1) RL agent picks an action (0=Hold,1=Buy,2=Sell)
        2) Convert to text
        3) GPT can override
        4) Return final action as text
        """
        action_num = self.agent.select_action(state, cluster_id)
        if action_num == 0:
            rl_action_str = "HOLD"
        elif action_num == 1:
            rl_action_str = "BUY"
        else:
            rl_action_str = "SELL"

        # Let GPT override
        final_action = self.gpt_controller.override_action(symbol, rl_action_str, extra_info)
        return final_action

    def store_transition(self, s, a, r, ns, d, cluster_id):
        # Convert textual action to numeric to store in replay
        if a.upper() == "HOLD":
            a_idx = 0
        elif a.upper() == "BUY":
            a_idx = 1
        else:
            a_idx = 2
        self.agent.store_transition(s, a_idx, r, ns, d, cluster_id)

    def optimize(self, beta: float):
        self.agent.optimize_model(beta)

    def update_target(self):
        self.agent.update_target_net()

###############################################################################
# 8) Technical Indicators
###############################################################################
def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
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

    final_upperband[0] = df2["basic_upperband"].iloc[0]
    final_lowerband[0] = df2["basic_lowerband"].iloc[0]

    for i in range(1, len(df2)):
        if df2["close"].iloc[i - 1] <= final_upperband[i - 1]:
            final_upperband[i] = min(df2["basic_upperband"].iloc[i], final_upperband[i - 1])
        else:
            final_upperband[i] = df2["basic_upperband"].iloc[i]

        if df2["close"].iloc[i - 1] >= final_lowerband[i - 1]:
            final_lowerband[i] = max(df2["basic_lowerband"].iloc[i], final_lowerband[i - 1])
        else:
            final_lowerband[i] = df2["basic_lowerband"].iloc[i]

    st = np.zeros(len(df2))
    direction = np.zeros(len(df2))

    if df2["close"].iloc[0] <= final_upperband[0]:
        st[0] = final_upperband[0]
        direction[0] = -1
    else:
        st[0] = final_lowerband[0]
        direction[0] = 1

    for i in range(1, len(df2)):
        if df2["close"].iloc[i] <= final_upperband[i]:
            st[i] = final_upperband[i]
            direction[i] = -1
        else:
            st[i] = final_lowerband[i]
            direction[i] = 1

    df2["supertrend"] = st
    df2["supertrend_direction"] = direction
    return df2

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    st_df = compute_supertrend(df, period=10, multiplier=3.0)
    df['supertrend'] = st_df['supertrend']
    df['supertrend_direction'] = st_df['supertrend_direction']

    # RSI
    rsi = ta.momentum.RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()

    # MACD
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Bollinger
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()

    # SMA
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['sma_crossover'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)

    # ATR
    atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()

    # Volume change
    df['volume_change'] = df['volume'].pct_change().fillna(0)

    df.dropna(inplace=True)
    return df

###############################################################################
# 9) Trade Helpers
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

async def apply_action(symbol: str, final_action: str, trader: AlpacaTrader,
                       logger: logging.Logger, trade_state: TradeState):
    acct = await trader.get_account()
    if not acct:
        return
    eq = float(acct.equity)
    last_price = await trader.get_latest_price(symbol)
    if not last_price or last_price <= 0:
        return
    max_alloc = eq * CAPITAL_ALLOCATION
    shares = int(max_alloc // last_price)
    if shares < 1:
        logger.info(f"Not enough equity for {symbol}. Requires 1 share, can allocate: {max_alloc}")
        return

    if final_action.upper() == "HOLD":
        logger.info(f"{symbol} => HOLD => no trade.")
    elif final_action.upper() == "BUY":
        pos = await trader.get_position(symbol)
        if pos and float(pos.qty) > 0:
            logger.info(f"{symbol} already has long pos. Skipping BUY.")
            return
        order = await trader.submit_order(
            symbol=symbol,
            qty=shares,
            side="buy",
            type="market",
            time_in_force="gtc"
        )
        if order:
            logger.info(f"Bought {shares} shares of {symbol}")
            trade_details = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "action": "Buy",
                "qty": shares,
                "price": last_price,
                "timestamp_executed": datetime.now(timezone.utc).isoformat()
            }
            record_trade(trade_details, logger)
            trade_state.update_trade(symbol, last_price)
    elif final_action.upper() == "SELL":
        pos = await trader.get_position(symbol)
        if pos and float(pos.qty) > 0:
            order = await trader.submit_order(
                symbol=symbol,
                qty=shares,
                side="sell",
                type="market",
                time_in_force="gtc"
            )
            if order:
                logger.info(f"Sold {shares} shares of {symbol}")
                trade_details = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "action": "Sell",
                    "qty": shares,
                    "price": last_price,
                    "timestamp_executed": datetime.now(timezone.utc).isoformat()
                }
                record_trade(trade_details, logger)
                if symbol in trade_state.active_trades:
                    del trade_state.active_trades[symbol]
        else:
            logger.info(f"{symbol} no active long position to sell.")

def record_trade(trade_details: Dict, logger: logging.Logger):
    logger.info(f"Trade Recorded: {trade_details}")
    try:
        with open("trade_log.json", "a") as f:
            json.dump(trade_details, f)
            f.write('\n')
    except Exception as e:
        logger.error(f"Error writing trade_log.json: {e}", exc_info=True)

###############################################################################
# 10) State Construction
###############################################################################
def build_state(df: pd.DataFrame, symbol: str, trade_state: TradeState) -> torch.Tensor:
    last_row = df.iloc[-1]

    supertrend_direction = float(last_row['supertrend_direction'])
    rsi = float(last_row['rsi'])
    macd = float(last_row['macd'])
    macd_signal = float(last_row['macd_signal'])
    macd_diff = float(last_row['macd_diff'])
    bb_mavg = float(last_row['bb_bbm'])
    bb_hband = float(last_row['bb_bbh'])
    bb_lband = float(last_row['bb_bbl'])
    sma_crossover = float(last_row['sma_crossover'])
    atr = float(last_row['atr'])
    volume_change = float(last_row['volume_change'])

    equity = float(trade_state.active_trades.get(symbol, {}).get('entry_price', 0.0))

    now = datetime.now(timezone.utc)
    hour_fraction = (now.hour * 60 + now.minute) / (24.0 * 60.0)

    if symbol in trade_state.active_trades:
        entry_time = trade_state.active_trades[symbol]['entry_time']
        days_held = (now - entry_time).days
    else:
        days_held = 0.0

    pos_flag = 1.0 if symbol in trade_state.active_trades else 0.0

    state = [
        supertrend_direction,
        rsi,
        macd,
        macd_signal,
        macd_diff,
        bb_mavg,
        bb_hband,
        bb_lband,
        sma_crossover,
        atr,
        volume_change,
        hour_fraction,
        days_held,
        pos_flag
    ]
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

###############################################################################
# 11) RL Environment
###############################################################################
async def fetch_historical_bars(symbol: str, trader: AlpacaTrader, logger: logging.Logger,
                                days: int = 400, timeframe: str = "1D") -> pd.DataFrame:
    try:
        now_utc = datetime.now(timezone.utc)
        start_utc = now_utc - timedelta(days=days)
        if timeframe == "1D":
            tf = TimeFrame.Day
        elif timeframe == "15Min":
            tf = TimeFrame.FifteenMin
        else:
            tf = TimeFrame.Day
        bars = trader.rest.get_bars(
            symbol,
            tf,
            start_utc.isoformat(),
            now_utc.isoformat(),
            adjustment="raw"
        ).df
        if bars.empty:
            logger.warning(f"No bars returned for {symbol}")
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

def eq_old(symbol: str, eq_new: float) -> float:
    return old_equities.get(symbol, eq_new)

async def process_symbol(symbol: str, rl: RLTrader, trader: AlpacaTrader,
                         logger: logging.Logger, trade_state: TradeState, symbol_id: int):
    bars = await fetch_historical_bars(symbol, trader, logger, days=400, timeframe="1D")
    if bars.empty or len(bars) < 200:
        logger.info(f"{symbol} => Not enough data for indicators.")
        return
    bars = compute_technical_indicators(bars)

    state = build_state(bars, symbol, trade_state).to(rl.device)
    # Provide GPT some basic info about last row of indicators
    last_row = bars.iloc[-1].to_dict()
    # RL picks action, GPT can override
    final_action = rl.pick_action(state, symbol_id, symbol, last_row)
    logger.info(f"{symbol} => Final Action: {final_action}")
    await apply_action(symbol, final_action, trader, logger, trade_state)

    # trailing stop update
    pos = await trader.get_position(symbol)
    if pos:
        entry_price = float(pos.avg_entry_price)
        trade_state.update_trade(symbol, entry_price)
        current_price = await trader.get_latest_price(symbol)
        if current_price:
            trade_state.adjust_stop_loss(symbol, current_price)

    current_price = await trader.get_latest_price(symbol)
    current_time = datetime.now(timezone.utc)
    exit_reason = None
    if current_price is not None:
        exit_reason = trade_state.check_exit(symbol, current_price, current_time,
                                             MIN_HOLD_DAYS, MAX_HOLD_DAYS)
    if exit_reason:
        logger.info(f"{symbol} => Exiting trade due to {exit_reason}")
        await apply_action(symbol, "SELL", trader, logger, trade_state)

    acct_updated = await trader.get_account()
    if not acct_updated:
        logger.warning("No account info for reward.")
        return
    eq_new = float(acct_updated.equity)
    reward = get_reward(symbol, eq_new)

    done = 0
    if (eq_new - eq_old(symbol, eq_new)) < (-MAX_DRAWDOWN * eq_new):
        done = 1
        logger.warning(f"{symbol} => equity dropped by {MAX_DRAWDOWN:.2%}. done=1.")

    # Build next state for the memory
    next_state = build_state(bars, symbol, trade_state).to(rl.device)
    rl.store_transition(state, final_action, reward, next_state, done, symbol_id)

###############################################################################
# 12) Main Bot Loop
###############################################################################
async def run_bot():
    trader = AlpacaTrader(config, logger)
    symbol_scanner = SymbolListScanner(SYMBOL_LIST_FILE, logger)
    symbols = await symbol_scanner.scan()
    if not symbols:
        logger.error("No symbols loaded. Exiting.")
        return
    if len(symbols) != 10:
        logger.warning(f"Symbols: {len(symbols)}, clusters=10 mismatch. Proceeding...")

    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl = RLTrader(device_torch, logger)

    episode = 0
    while True:
        try:
            logger.info("Starting new trading cycle.")
            tasks = []
            for idx, sym in enumerate(symbols):
                tasks.append(process_symbol(sym, rl, trader, logger, trade_state, idx))
            await asyncio.gather(*tasks)

            beta = min(1.0, rl.agent.beta_start + episode * (1.0 - rl.agent.beta_start) / rl.agent.beta_frames)
            rl.optimize(beta)

            if episode % TARGET_UPDATE == 0:
                rl.update_target()

            if episode % 100 == 0:
                rl.save_agent()

            episode += 1
            logger.info("Trading cycle complete. Sleeping 60s.\n")
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            await asyncio.sleep(60)

###############################################################################
# 13) Script Entry
###############################################################################
if __name__ == "__main__":
    rl_instance = None
    async def main():
        global rl_instance
        rl_instance = RLTrader(torch.device("cuda" if torch.cuda.is_available() else "cpu"), logger)
        await run_bot()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("User interrupted. Exiting gracefully.")
        if rl_instance:
            rl_instance.save_agent()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)

