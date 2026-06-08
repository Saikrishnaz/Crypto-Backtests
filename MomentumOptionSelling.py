"""
Momentum Option Selling Backtest Strategy
==========================================
This backtest program implements an intraday ATM option selling strategy based on
Pivot Point breakouts confirmed by SuperTrend direction on spot index data (e.g., NIFTY).

1. Core Concept & Indicators
-----------------------------
- The strategy uses a 5-minute consolidated spot price chart.
- Two indicators are computed:
  - Standard Pivot Points (R1, S1) derived from the PREVIOUS day's High, Low, Close:
      Pivot = (prev_High + prev_Low + prev_Close) / 3
      R1    = (2 * Pivot) - prev_Low
      S1    = (2 * Pivot) - prev_High
  - SuperTrend (ATR Period = 7, Multiplier = 3) to identify trend direction.
    Direction +1 = Bullish (price above SuperTrend), -1 = Bearish (price below).

2. Entry Conditions
--------------------
A position is entered when BOTH conditions align on the same 5-min candle:
- Bullish Setup (Sell ATM Put):
    - 5-min Close > R1 (price breaks above resistance).
    - SuperTrend direction == +1 (bullish, price above SuperTrend line).
    → Action: SELL ATM PE (current weekly expiry).
- Bearish Setup (Sell ATM Call):
    - 5-min Close < S1 (price breaks below support).
    - SuperTrend direction == -1 (bearish, price below SuperTrend line).
    → Action: SELL ATM CE (current weekly expiry).

3. Option Selection
--------------------
- ATM strike is determined by rounding the spot Close to the nearest step size (50 for NIFTY).
- The corresponding 1-minute options data file is loaded from OPTIONS_PATH
  (e.g., NIFTYyymmdd[Strike][CE/PE].csv).
- Entry price is the option Close at the 1-min candle matching the 5-min signal timestamp.

4. Exit Conditions
-------------------
- SuperTrend Flip:
    - For Sell Put: SuperTrend flips bearish (direction changes to -1) → Exit immediately.
    - For Sell Call: SuperTrend flips bullish (direction changes to +1) → Exit immediately.
- Time Exit: At 15:20 or later, all open positions are squared off. No overnight holding.
- Exit price is fetched from the 1-min option Close at the exit timestamp.

5. Risk Management
-------------------
- Maximum 3 trades per day. After 3 entries, no more trades even if signals appear.
- No new entry after 15:00. Only existing positions are managed after 3 PM.

6. Performance Reporting & Metrics
-----------------------------------
After the backtest completes, it calculates:
- Monthly & Yearly PNL.
- Annualized Max Drawdown.
- ROI % based on initial capital of 25,000 RS.
- Recovery Days from Max Drawdown.
- Highest single-trade profit and loss.
- Total trade count per month.
- Saves the trade-by-trade log and summary statistics to BACKTEST_MOMENTUM_OPTION_SELLING.csv.
"""

import pandas as pd
import numpy as np
import pandas_ta as pdt
from backtesting import Backtest, Strategy
from datetime import datetime, time as dtime
import os


# --- OHLC Consolidate for 5 Min Intraday Data ---
def ohlc_consolidate(df: pd.DataFrame, timevalue: str, Isvolume: bool = True) -> pd.DataFrame:
    df = df.copy()
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)

    # Filter time range
    df = df[(df.index.time >= dtime(9, 15)) & (df.index.time < dtime(15, 30))]

    # Resample
    ohlc_df = df.resample(
        timevalue, offset=(pd.Timestamp('09:15:00') - pd.Timestamp('00:00:00'))
    ).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    if Isvolume and 'volume' in df.columns:
        resampled_volume = df['volume'].resample(
            timevalue, offset=(pd.Timestamp('09:15:00') - pd.Timestamp('00:00:00'))
        ).sum()
        ohlc_df['volume'] = resampled_volume
    else:
        ohlc_df['volume'] = 0

    ohlc_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    return ohlc_df


# --- Standard Pivot Points (R1, S1) from Previous Day ---
def compute_pivot_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Standard Pivot Points using the previous trading day's High, Low, Close.
    Only R1 and S1 are kept (Pivot center, R2/R3, S2/S3 are omitted per strategy rules).

    Formula:
        Pivot = (prev_H + prev_L + prev_C) / 3
        R1    = (2 * Pivot) - prev_L
        S1    = (2 * Pivot) - prev_H
    """
    df = df.copy()
    df['date'] = df.index.date

    # Get daily High, Low, Close
    daily = df.groupby('date').agg(
        day_high=('high', 'max'),
        day_low=('low', 'min'),
        day_close=('close', 'last')
    )

    # Shift by 1 day to get PREVIOUS day's values
    daily['prev_high'] = daily['day_high'].shift(1)
    daily['prev_low'] = daily['day_low'].shift(1)
    daily['prev_close'] = daily['day_close'].shift(1)

    # Calculate Pivot, R1, S1
    daily['pivot'] = (daily['prev_high'] + daily['prev_low'] + daily['prev_close']) / 3
    daily['R1'] = (2 * daily['pivot']) - daily['prev_low']
    daily['S1'] = (2 * daily['pivot']) - daily['prev_high']

    # Merge back to intraday dataframe
    df = df.merge(daily[['R1', 'S1']], left_on='date', right_index=True, how='left')
    df.drop(columns=['date'], inplace=True)

    return df


# --- SuperTrend Indicator ---
def compute_supertrend(df: pd.DataFrame, length: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute SuperTrend using pandas_ta.
    Adds columns: SUPERT (value) and SUPERTd (direction: +1 bullish, -1 bearish).
    """
    df = df.copy()
    supertrend = pdt.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
    
    st_val_col = f'SUPERT_{length}_{multiplier}'
    st_dir_col = f'SUPERTd_{length}_{multiplier}'

    df['SUPERT'] = supertrend[st_val_col]
    df['SUPERTd'] = supertrend[st_dir_col]

    return df


def parse_strike(rule, ltp, step=50):
    """Parse strike rule like 'ATM', 'ATM+100', 'ATM-100'. Returns strike price (int)."""
    if isinstance(rule, str) and "ATM" in rule:
        atm = get_strike(ltp, step)
        expression = rule.replace("ATM", str(atm))
        try:
            return int(eval(expression))
        except Exception as e:
            print(f"Error parsing strike rule '{rule}': {e}")
            return atm
    return int(rule)


def get_strike(ltp, step_size):
    """Round LTP to nearest strike price"""
    remainder = ltp % step_size
    if remainder >= step_size / 2:
        return int(ltp + (step_size - remainder))
    else:
        return int(ltp - remainder)


# --- Trade Record Template ---
def default_records():
    return {
        "signal_timestamp": None,        # Time when entry conditions met
        "signal_type": None,             # "BULLISH" or "BEARISH"
        "pivot_level_broken": None,      # R1 or S1 value that was broken
        "supertrend_value": None,        # SuperTrend value at entry
        "supertrend_direction": None,    # "BULLISH" or "BEARISH" at entry
        "spot_price_at_entry": None,     # Spot Close at signal candle
        "symbol": None,                  # "NIFTY"
        "expiry_date": None,
        "strike_price": None,
        "option_type": None,             # "CE" or "PE"
        "entry_signal": None,            # Always "SELL" (option selling strategy)
        "entry_time": None,
        "entry_price": None,             # Option premium at entry
        "exit_timestamp": None,
        "exit_price": None,              # Option premium at exit
        "profit_points": None,           # entry_price - exit_price (SELL)
        "reason_for_exit": None,         # "SuperTrend Flip" / "Exittime"
        "options_data": None,            # 1-min options DF for the day
        "trade_number_today": None,      # 1, 2, or 3
    }


def merge_expires(data, symbol):
    exp_collection = {"NIFTY_EXPIRES":["2020-01-02","2020-01-09","2020-01-16","2020-01-23","2020-01-30","2020-02-06","2020-02-13","2020-02-20","2020-02-27","2020-03-05","2020-03-12","2020-03-19","2020-03-26","2020-04-01","2020-04-09","2020-04-16","2020-04-23","2020-04-30","2020-05-07","2020-05-14","2020-05-21","2020-05-28","2020-06-04","2020-06-11","2020-06-18","2020-06-25","2020-07-02","2020-07-09","2020-07-16","2020-07-23","2020-07-30","2020-08-06","2020-08-13","2020-08-20","2020-08-27","2020-09-03","2020-09-10","2020-09-17","2020-09-24","2020-10-01","2020-10-08","2020-10-15","2020-10-22","2020-10-29","2020-11-05","2020-11-12","2020-11-19","2020-11-26","2020-12-03","2020-12-10","2020-12-17","2020-12-24","2020-12-31","2021-01-07","2021-01-14","2021-01-21","2021-01-28","2021-02-04","2021-02-11","2021-02-18","2021-02-25","2021-03-04","2021-03-10","2021-03-18","2021-03-25","2021-04-01","2021-04-08","2021-04-15","2021-04-22","2021-04-29","2021-05-06","2021-05-12","2021-05-20","2021-05-27","2021-06-03","2021-06-10","2021-06-17","2021-06-24","2021-07-01","2021-07-08","2021-07-15","2021-07-22","2021-07-29","2021-08-05","2021-08-12","2021-08-18","2021-08-26","2021-09-02","2021-09-09","2021-09-16","2021-09-23","2021-09-30","2021-10-07","2021-10-14","2021-10-21","2021-10-28","2021-11-03","2021-11-11","2021-11-18","2021-11-25","2021-12-02","2021-12-09","2021-12-16","2021-12-23","2021-12-30","2022-01-06","2022-01-13","2022-01-20","2022-01-27","2022-02-03","2022-02-10","2022-02-17","2022-02-24","2022-03-03","2022-03-10","2022-03-17","2022-03-24","2022-03-31","2022-04-07","2022-04-13","2022-04-21","2022-04-28","2022-05-05","2022-05-12","2022-05-19","2022-05-26","2022-06-02","2022-06-09","2022-06-16","2022-06-23","2022-06-30","2022-07-07","2022-07-14","2022-07-21","2022-07-28","2022-08-04","2022-08-11","2022-08-18","2022-08-25","2022-09-01","2022-09-08","2022-09-15","2022-09-22","2022-09-29","2022-10-06","2022-10-13","2022-10-20","2022-10-27","2022-11-03","2022-11-10","2022-11-17","2022-11-24","2022-12-01","2022-12-08","2022-12-15","2022-12-22","2022-12-29","2023-01-05","2023-01-12","2023-01-19","2023-01-25","2023-02-02","2023-02-09","2023-02-16","2023-02-23","2023-03-02","2023-03-09","2023-03-16","2023-03-23","2023-03-29","2023-03-30","2023-04-06","2023-04-13","2023-04-20","2023-04-27","2023-05-04","2023-05-11","2023-05-18","2023-05-25","2023-06-01","2023-06-08","2023-06-15","2023-06-22","2023-06-28","2023-06-29","2023-07-06","2023-07-13","2023-07-20","2023-07-27","2023-08-03","2023-08-10","2023-08-17","2023-08-24","2023-08-31","2023-09-07","2023-09-14","2023-09-21","2023-09-28","2023-10-05","2023-10-12","2023-10-19","2023-10-26","2023-11-02","2023-11-09","2023-11-16","2023-11-23","2023-11-30","2023-12-07","2023-12-14","2023-12-21","2023-12-28","2024-01-04","2024-01-11","2024-01-18","2024-01-25","2024-02-01","2024-02-08","2024-02-15","2024-02-22","2024-02-29","2024-03-07","2024-03-14","2024-03-21","2024-03-28","2024-04-04","2024-04-10","2024-04-18","2024-04-25","2024-05-02","2024-05-09","2024-05-16","2024-05-23","2024-05-30","2024-06-06","2024-06-13","2024-06-20","2024-06-27","2024-07-04","2024-07-11","2024-07-18","2024-07-25","2024-08-01","2024-08-08","2024-08-14","2024-08-22","2024-08-29","2024-09-05","2024-09-12","2024-09-19","2024-09-26","2024-10-03","2024-10-10","2024-10-17","2024-10-24","2024-10-31","2024-11-07","2024-11-14","2024-11-21","2024-11-28","2024-12-05","2024-12-12","2024-12-19","2024-12-26","2025-01-02","2025-01-09","2025-01-16","2025-01-23","2025-01-30","2025-02-06","2025-02-13","2025-02-20","2025-02-27","2025-03-06","2025-03-13","2025-03-20","2025-03-27","2025-04-03","2025-04-09","2025-04-17","2025-04-24","2025-04-30","2025-05-08","2025-05-15","2025-05-22","2025-05-29","2025-06-05","2025-06-12","2025-06-19","2025-06-26","2025-07-03","2025-07-10","2025-07-17","2025-07-24","2025-07-31","2025-08-07","2025-08-14","2025-08-21","2025-08-28","2025-09-02","2025-09-09","2025-09-16","2025-09-23","2025-09-25","2025-09-30","2025-10-07","2025-10-14","2025-10-20","2025-10-28","2025-11-04","2025-11-11","2026-01-06","2026-01-13","2026-01-20","2026-01-27","2026-02-03","2026-02-10","2026-02-17","2026-02-24","2026-03-02","2026-03-10","2026-03-17","2026-03-24","2026-03-26","2026-03-30","2026-03-31","2026-04-07","2026-04-13","2026-04-21","2026-04-28","2026-05-05","2026-05-12","2026-05-19","2026-05-26","2026-06-02","2026-06-09","2026-06-16","2026-06-23","2026-06-25","2026-06-30","2026-07-07","2026-07-28","2026-08-25","2026-09-29","2026-12-29","2026-12-31"]}

    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data['datetime'] = pd.to_datetime(data.index)

    # 1) Add a pure date column to data
    data["trade_date"] = data.index.normalize()  # strips time -> 2024-01-04 00:00:00
    # 2) Prepare expiry df with a date column
    exp_df = pd.DataFrame({
        "expiry": pd.to_datetime(exp_collection[f"{symbol}_EXPIRES"])
    }).sort_values("expiry")

    exp_df["expiry_date"] = exp_df["expiry"].dt.normalize()

    # 3) merge_asof on date, not full timestamp
    final_df = pd.merge_asof(
        data.sort_values("trade_date"),
        exp_df[["expiry_date", "expiry"]].sort_values("expiry_date"),
        left_on="trade_date",
        right_on="expiry_date",
        direction="forward"
    )

    # 4) Clean up
    final_df = final_df.drop(columns=["trade_date", "expiry_date"])
    final_df.set_index(keys="datetime", inplace=True)
    final_df["expiry"] = pd.to_datetime(final_df["expiry"]).dt.date
    final_df.dropna(subset=["expiry"], inplace=True)
    final_df.sort_values(by="datetime", inplace=True)
    return final_df


# --- Strategy Class ---
class MomentumOptionSellingStrategy(Strategy):
    strike_config = "ATM"
    step_size = {"NIFTY": 50, "SENSEX": 100, "BANKNIFTY": 100}
    OPTIONS_PATH = "options_dir"
    symbol = "NIFTY"
    supertrend_length = 7
    supertrend_multiplier = 3.0
    max_trades_per_day = 3
    no_entry_after = dtime(15, 0)
    exit_time = dtime(15, 20)
    signals = []

    def generate_symbol(self, option_type):
        """Build the options CSV filename key like NIFTY24060623250CE"""
        expiry_val = self.data.expiry[-1]
        if isinstance(expiry_val, str):
            expiry_dt = datetime.strptime(expiry_val, "%Y-%m-%d")
        elif hasattr(expiry_val, "strftime"):
            expiry_dt = expiry_val
        else:
            expiry_dt = pd.to_datetime(expiry_val)

        expiry_str = expiry_dt.strftime("%y%m%d")
        ATM_KEY = (
            f"{self.symbol.upper()}"
            f"{expiry_str}"
            f"{int(float(self.current_trade['strike_price']))}"
            f"{option_type.upper()}"
        )
        return ATM_KEY

    def trade_finished(self):
        """
        Calculate PNL for the completed trade (always SELL: profit = entry - exit),
        append to the class-level signals list, and reset trade state.
        """
        self.current_trade["profit_points"] = (
            self.current_trade["entry_price"] -
            self.current_trade["exit_price"]
        )

        MomentumOptionSellingStrategy.signals.append([
            self.current_trade["signal_timestamp"],
            self.current_trade["signal_type"],
            self.current_trade["pivot_level_broken"],
            self.current_trade["supertrend_value"],
            self.current_trade["supertrend_direction"],
            self.current_trade["spot_price_at_entry"],
            self.current_trade["symbol"],
            self.current_trade["expiry_date"],
            self.current_trade["strike_price"],
            self.current_trade["option_type"],
            self.current_trade["entry_signal"],
            self.current_trade["entry_time"],
            self.current_trade["entry_price"],
            self.current_trade["exit_timestamp"],
            self.current_trade["exit_price"],
            self.current_trade["profit_points"],
            self.current_trade["reason_for_exit"],
            self.current_trade["trade_number_today"],
        ])
        self.current_trade = default_records()

    def init(self):
        self.current_trade = default_records()
        self.trades_today = 0
        self.current_date = None

    def next(self):
        # Need at least a few candles for SuperTrend to stabilize
        if len(self.data) < 10:
            return

        current_time = self.data.index[-1].time()
        current_date = self.data.index[-1].date()

        # --- Day Reset: reset trade counter on new trading day ---
        if self.current_date != current_date:
            self.current_date = current_date
            self.trades_today = 0

        # --- Read Indicator Values ---
        st_dir = self.data.SUPERTd[-1]        # +1 bullish, -1 bearish
        st_val = self.data.SUPERT[-1]          # SuperTrend line value
        close = self.data.Close[-1]
        r1 = self.data.R1[-1]
        s1 = self.data.S1[-1]

        # Skip candles where pivot levels are NaN (first day of data)
        if pd.isna(r1) or pd.isna(s1):
            return

        # Detect SuperTrend direction flip (current vs previous candle)
        prev_st_dir = self.data.SUPERTd[-2] if len(self.data) >= 2 else st_dir
        st_flipped_bullish = (prev_st_dir == -1) and (st_dir == 1)
        st_flipped_bearish = (prev_st_dir == 1) and (st_dir == -1)

        # =====================================================================
        # EXIT MANAGEMENT (checked first, if in a trade)
        # =====================================================================
        if self.current_trade["entry_price"] is not None:

            exit_triggered = False
            exit_reason = ""

            # --- Exit Rule 1: SuperTrend Flip ---
            if self.current_trade["option_type"] == "PE":
                # Selling PUT (bullish position) → exit if ST flips bearish
                if st_flipped_bearish:
                    exit_triggered = True
                    exit_reason = "SuperTrend Flip"

            elif self.current_trade["option_type"] == "CE":
                # Selling CALL (bearish position) → exit if ST flips bullish
                if st_flipped_bullish:
                    exit_triggered = True
                    exit_reason = "SuperTrend Flip"

            # --- Exit Rule 2: Time Exit at 15:20 ---
            if current_time >= self.exit_time:
                exit_triggered = True
                exit_reason = "Exittime"

            # --- Execute Exit ---
            if exit_triggered:
                ATM_DF: pd.DataFrame = self.current_trade["options_data"].sort_values(by="timestamp")
                exit_candles = ATM_DF[ATM_DF["timestamp"].dt.time >= current_time]

                if not exit_candles.empty:
                    exit_price = exit_candles.iloc[0]["Close"]
                else:
                    # Fallback: last available candle
                    exit_price = ATM_DF.iloc[-1]["Close"]

                self.current_trade["exit_timestamp"] = self.data.index[-1]
                self.current_trade["exit_price"] = exit_price
                self.current_trade["reason_for_exit"] = exit_reason
                self.trade_finished()
                return

        # =====================================================================
        # ENTRY LOGIC (only if NOT in a trade)
        # =====================================================================
        if self.current_trade["entry_price"] is not None:
            return  # Already in a trade, skip entry checks

        # Risk Rule 1: Max 3 trades per day
        if self.trades_today >= self.max_trades_per_day:
            return

        # Risk Rule 2: No new entry after 15:00
        if current_time >= self.no_entry_after:
            return

        # --- Check Entry Conditions ---
        triggered = False
        signal_type = ""
        option_type = ""       # "CE" or "PE"
        pivot_broken = None

        # Bullish Setup: Close > R1 AND SuperTrend bullish
        if close > r1 and st_dir == 1:
            triggered = True
            signal_type = "BULLISH"
            option_type = "PE"    # Sell ATM Put
            pivot_broken = r1

        # Bearish Setup: Close < S1 AND SuperTrend bearish
        elif close < s1 and st_dir == -1:
            triggered = True
            signal_type = "BEARISH"
            option_type = "CE"    # Sell ATM Call
            pivot_broken = s1

        if not triggered:
            return

        # --- Select ATM Strike ---
        strike = parse_strike(
            self.strike_config,
            close,
            self.step_size.get(self.symbol, 50)
        )

        # --- Build option symbol and load 1-min data ---
        self.current_trade["strike_price"] = strike
        ATM_KEY = self.generate_symbol(option_type)
        csv_filename = os.path.join(self.OPTIONS_PATH, ATM_KEY + ".csv")

        if not os.path.exists(csv_filename):
            print(f"Option data file not found: {csv_filename} - skipping entry.")
            self.current_trade = default_records()
            return

        try:
            ATM_DF = pd.read_csv(
                csv_filename,
                header=None,
                names=["Date", "Time", "Open", "High", "Low", "Close", "Volume", "IO"]
            )
            ATM_DF["Datetime"] = pd.to_datetime(
                ATM_DF["Date"].astype(str) + " " + ATM_DF["Time"].astype(str),
                format="%Y%m%d %H:%M"
            )
            ATM_DF.rename(columns={"Datetime": "timestamp"}, inplace=True)
            ATM_DF = ATM_DF[ATM_DF["timestamp"].dt.date == self.data.index[-1].date()].sort_values(by="timestamp")

            if ATM_DF.empty:
                print(f"No options data for date {self.data.index[-1].date()} in {csv_filename} - skipping.")
                self.current_trade = default_records()
                return

            # Entry price: option Close at or before the 5-min signal candle time
            entry_candles = ATM_DF[ATM_DF["timestamp"] <= self.data.index[-1]]
            opt_entry_price = entry_candles.iloc[-1]["Close"] if not entry_candles.empty else ATM_DF.iloc[0]["Close"]

        except Exception as e:
            print(f"Error parsing option file {csv_filename}: {e} - skipping entry.")
            self.current_trade = default_records()
            return

        # --- Record Entry ---
        self.trades_today += 1
        self.current_trade["signal_timestamp"] = self.data.index[-1]
        self.current_trade["signal_type"] = signal_type
        self.current_trade["pivot_level_broken"] = pivot_broken
        self.current_trade["supertrend_value"] = st_val
        self.current_trade["supertrend_direction"] = "BULLISH" if st_dir == 1 else "BEARISH"
        self.current_trade["spot_price_at_entry"] = close
        self.current_trade["symbol"] = self.symbol
        self.current_trade["expiry_date"] = self.data.expiry[-1]
        self.current_trade["strike_price"] = strike
        self.current_trade["option_type"] = option_type
        self.current_trade["entry_signal"] = "SELL"
        self.current_trade["entry_time"] = self.data.index[-1]
        self.current_trade["entry_price"] = opt_entry_price
        self.current_trade["options_data"] = ATM_DF
        self.current_trade["trade_number_today"] = self.trades_today


# ------------------------------------------------------------------------------------------------
# Performance Metrics
# ------------------------------------------------------------------------------------------------
def recovery_days(cum_pnl):
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak

    if drawdown.min() >= 0:
        return 0

    max_dd_idx = drawdown.idxmin()
    peak_before_dd = peak[:max_dd_idx].iloc[-1]

    dd_start_idx = peak[:max_dd_idx][peak[:max_dd_idx] == peak_before_dd].index[-1]

    recovery_point = cum_pnl[max_dd_idx:]
    recovered = recovery_point[recovery_point >= peak_before_dd]

    if recovered.empty:
        return (cum_pnl.index[-1] - dd_start_idx).days

    recovery_idx = recovered.index[0]
    recovery_duration = (recovery_idx - dd_start_idx).days
    return recovery_duration


def main(backtest_from_to, csv_file, only_expiry, symbol, OPTIONS_PATH, formated_symbols,
         step_size, timeframe, strike_config, supertrend_length, supertrend_multiplier,
         max_trades_per_day, no_entry_after, exit_time):
    cash_rs = 25000

    if not os.path.exists(csv_file):
        print(f"Data file not found: {csv_file}")
        return

    # Load 1-min spot data
    print("Loading spot data...")
    df = pd.read_csv(
        csv_file,
        header=None,
        names=["Date", "Time", "open", "high", "low", "close", "volume", "io"]
    )

    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%Y%m%d %H:%M"
    )
    df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    df = df[(df["timestamp"] >= backtest_from_to["start_date"]) & (df["timestamp"] <= backtest_from_to["end_date"])]

    # Consolidate to 5-min
    print(f"Consolidating to {timeframe} timeframe...")
    consolidated_df = ohlc_consolidate(df, timeframe, Isvolume=True)

    # Compute Pivot Points (R1, S1) from previous day
    print("Computing Pivot Point levels (R1, S1)...")
    consolidated_df = compute_pivot_levels(consolidated_df)

    # Compute SuperTrend
    print(f"Computing SuperTrend (length={supertrend_length}, multiplier={supertrend_multiplier})...")
    consolidated_df = compute_supertrend(consolidated_df, length=supertrend_length, multiplier=supertrend_multiplier)

    # Rename to match backtesting.py expectations (uppercase OHLC)
    consolidated_df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close"
    }, inplace=True)

    # Merge weekly expiry dates
    consolidated_df = merge_expires(consolidated_df, symbol)
    if only_expiry:
        consolidated_df = consolidated_df[
            consolidated_df.index.date == pd.to_datetime(consolidated_df["expiry"]).dt.date
        ]

    # Save indicator levels for verification
    consolidated_df.to_csv("levels_momentum.csv", index=True)

    # --- Parse time strings to dtime objects ---
    if isinstance(no_entry_after, str):
        h, m = map(int, no_entry_after.split(":"))
        no_entry_time = dtime(h, m)
    else:
        no_entry_time = no_entry_after

    if isinstance(exit_time, str):
        h, m = map(int, exit_time.split(":"))
        exit_time_obj = dtime(h, m)
    else:
        exit_time_obj = exit_time

    # --- Configure Strategy Class Parameters ---
    print("Starting Backtest...")
    MomentumOptionSellingStrategy.strike_config = strike_config
    MomentumOptionSellingStrategy.symbol = symbol
    MomentumOptionSellingStrategy.OPTIONS_PATH = OPTIONS_PATH
    MomentumOptionSellingStrategy.step_size = step_size
    MomentumOptionSellingStrategy.supertrend_length = supertrend_length
    MomentumOptionSellingStrategy.supertrend_multiplier = supertrend_multiplier
    MomentumOptionSellingStrategy.max_trades_per_day = max_trades_per_day
    MomentumOptionSellingStrategy.no_entry_after = no_entry_time
    MomentumOptionSellingStrategy.exit_time = exit_time_obj
    MomentumOptionSellingStrategy.signals = []  # Reset static variable

    bt = Backtest(
        consolidated_df,
        MomentumOptionSellingStrategy,
        cash=100_000_000,
        commission=0.0002,
        trade_on_close=False
    )
    stats = bt.run()

    # --- Process Trade Log ---
    print("\n--- Backtest Statistics ---")
    signals_list = MomentumOptionSellingStrategy.signals
    if len(signals_list) == 0:
        print("\nNo trades executed during this period.")
        return

    trades_df = pd.DataFrame(
        signals_list,
        columns=[
            "signal_timestamp",
            "signal_type",
            "pivot_level_broken",
            "supertrend_value",
            "supertrend_direction",
            "spot_price_at_entry",
            "symbol",
            "expiry_date",
            "strike_price",
            "option_type",
            "entry_signal",
            "ENTRY_TIME",
            "entry_price",
            "EXIT_TIME",
            "exit_price",
            "PNL",
            "reason_for_exit",
            "trade_number_today",
        ]
    )

    trades_df['ENTRY_TIME'] = pd.to_datetime(trades_df['ENTRY_TIME'])
    trades_df['EXIT_TIME'] = pd.to_datetime(trades_df['EXIT_TIME'])
    trades_df['month'] = trades_df['EXIT_TIME'].dt.to_period('M')
    trades_df['year'] = trades_df['EXIT_TIME'].dt.year

    monthly_pnl = trades_df.groupby('month')['PNL'].sum()
    yearly_pnl = trades_df.groupby('year')['PNL'].sum()

    drawdown_data = []
    for year, group in trades_df.groupby('year'):
        cum_pnl = group['PNL'].cumsum()
        peak = cum_pnl.cummax()
        drawdown = (cum_pnl - peak)
        max_dd = drawdown.min()
        drawdown_data.append((year, max_dd))

    monthly_trades = trades_df.groupby('month').size()
    cum_pnl = trades_df['PNL'].cumsum()
    cum_pnl.index = trades_df['EXIT_TIME']

    recovery_days_from_dd = recovery_days(cum_pnl)

    highest_profit = trades_df['PNL'].max()
    highest_loss = trades_df['PNL'].min()

    roi_data = (yearly_pnl / cash_rs) * 100
    summary_rows = []

    for period, pnl in monthly_pnl.items():
        summary_rows.append({'ENTRY_TIME': None, 'EXIT_TIME': None, 'PNL': pnl, 'option_type': f'Monthly PnL ({period})'})

    for year, pnl in yearly_pnl.items():
        summary_rows.append({'ENTRY_TIME': None, 'EXIT_TIME': None, 'PNL': pnl, 'option_type': f'Total Year PnL ({year})'})

    for year, dd in drawdown_data:
        summary_rows.append({'ENTRY_TIME': None, 'EXIT_TIME': None, 'PNL': dd, 'option_type': f'Max Drawdown ({year})'})

    for year, roi in roi_data.items():
        summary_rows.append({'ENTRY_TIME': None, 'EXIT_TIME': None, 'PNL': roi, 'option_type': f'ROI % ({year})'})

    for period, count in monthly_trades.items():
        summary_rows.append({'ENTRY_TIME': None, 'EXIT_TIME': None, 'PNL': count, 'option_type': f'Trades in ({period})'})

    summary_rows.append({'ENTRY_TIME': None, 'EXIT_TIME': None, 'PNL': recovery_days_from_dd, 'option_type': 'Recovery Days from MaxDD'})
    summary_rows.append({'ENTRY_TIME': None, 'EXIT_TIME': None, 'PNL': highest_profit, 'option_type': 'Highest Single Trade Profit'})
    summary_rows.append({'ENTRY_TIME': None, 'EXIT_TIME': None, 'PNL': highest_loss, 'option_type': 'Highest Single Trade Loss'})

    summary_df = pd.DataFrame(summary_rows)
    final_df = pd.concat([trades_df, summary_df], ignore_index=True)
    final_df.drop(columns=["year", "month"], inplace=True)

    output_filename = "BACKTEST_MOMENTUM_OPTION_SELLING.csv"
    final_df.to_csv(output_filename, index=False)
    print(f"\nTrades and summaries saved to {output_filename}")
    print(f"Total trades executed: {len(trades_df)}")
    print(f"Cumulative PNL: {trades_df['PNL'].sum():.2f}")


if __name__ == "__main__":

    kwags = dict(
        # --- Data & Backtest Paths ---
        backtest_from_to={"start_date": datetime(2024, 6, 1), "end_date": datetime(2024, 12, 31)},
        symbol="NIFTY",
        formated_symbols={"NIFTY": "NIFTY 50", "SENSEX": "SENSEX", "BANKNIFTY": "NIFTY BANK"},
        timeframe="5min",
        only_expiry=False,
        OPTIONS_PATH=r"C:\Users\Saikr\Downloads\rohith_02\Rohith Set (1 min)\Jan 2020 to Dec 2024 - NIFTY Spot & Options\NIFTY Options",
        csv_file=r"C:\Users\Saikr\Downloads\rohith_02\Rohith Set (1 min)\Jan 2020 to Dec 2024 - NIFTY Spot & Options\NIFTY Spot\NIFTY.csv",

        # --- Options Trading Settings ---
        strike_config="ATM",  # "ATM", "ATM+100", "ATM-100" etc.
        step_size={"NIFTY": 50, "SENSEX": 100, "BANKNIFTY": 100},

        # --- Indicator Settings ---
        supertrend_length=7,
        supertrend_multiplier=3.0,

        # --- Risk Management ---
        max_trades_per_day=3,
        no_entry_after="15:00",
        exit_time="15:20",
    )
    main(**kwags)
