"""
Momentum Crypto Spot Backtest Strategy
======================================
This script adapts the original NIFTY option-selling strategy to a crypto spot trading
backtest. It uses 5-minute consolidated spot candles, daily pivot point levels, and
SuperTrend direction to generate long/short spot trades directly on the crypto asset.

Core rules:
- Buy when price breaks above R1 and SuperTrend is bullish.
- Short when price breaks below S1 and SuperTrend is bearish.
- Exit when SuperTrend flips or the opposite pivot condition is met.
- Maximum trades per day is configurable.
"""

import os
from datetime import datetime

import pandas as pd
import pandas_ta as pdt
from backtesting import Backtest, Strategy


def ohlc_consolidate(df: pd.DataFrame, timeframe: str, Isvolume: bool = True) -> pd.DataFrame:
    df = df.copy()
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)

    ohlc = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    if Isvolume and 'volume' in df.columns:
        ohlc['volume'] = df['volume'].resample(timeframe).sum()
    else:
        ohlc['volume'] = 0

    ohlc.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    return ohlc


def compute_pivot_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = df.index.date

    daily = df.groupby('date').agg(
        day_high=('high', 'max'),
        day_low=('low', 'min'),
        day_close=('close', 'last')
    )

    daily['prev_high'] = daily['day_high'].shift(1)
    daily['prev_low'] = daily['day_low'].shift(1)
    daily['prev_close'] = daily['day_close'].shift(1)

    daily['pivot'] = (daily['prev_high'] + daily['prev_low'] + daily['prev_close']) / 3
    daily['R1'] = (2 * daily['pivot']) - daily['prev_low']
    daily['S1'] = (2 * daily['pivot']) - daily['prev_high']

    df = df.merge(daily[['R1', 'S1']], left_on='date', right_index=True, how='left')
    df.drop(columns=['date'], inplace=True)
    return df


def compute_supertrend(df: pd.DataFrame, length: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    supertrend = pdt.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
    df['SUPERT'] = supertrend[f'SUPERT_{length}_{multiplier}']
    df['SUPERTd'] = supertrend[f'SUPERTd_{length}_{multiplier}']
    return df


def default_trade_record():
    return {
        'entry_time': None,
        'exit_time': None,
        'entry_price': None,
        'exit_price': None,
        'position_type': None,
        'reason_for_exit': None,
        'signal_price': None,
        'R1': None,
        'S1': None,
        'SUPERT': None,
        'SUPERTd': None,
        'trade_date': None,
        'profit': None,
        'trade_number_today': None,
    }


class MomentumCryptoSpotStrategy(Strategy):
    symbol = 'BTCUSD'
    supertrend_length = 7
    supertrend_multiplier = 3.0
    max_trades_per_day = 3
    signals = []

    def init(self):
        self.current_trade = default_trade_record()
        self.current_date = None
        self.trades_today = 0

    def trade_finished(self):
        self.current_trade['profit'] = (
            self.current_trade['exit_price'] - self.current_trade['entry_price']
            if self.current_trade['position_type'] == 'LONG'
            else self.current_trade['entry_price'] - self.current_trade['exit_price']
        )
        MomentumCryptoSpotStrategy.signals.append(self.current_trade.copy())
        self.current_trade = default_trade_record()

    def next(self):
        if len(self.data) < 10:
            return

        current_dt = self.data.index[-1]
        current_date = current_dt.date()

        if self.current_date != current_date:
            self.current_date = current_date
            self.trades_today = 0

        close = self.data.Close[-1]
        r1 = self.data.R1[-1]
        s1 = self.data.S1[-1]
        st_dir = self.data.SUPERTd[-1]
        st_val = self.data.SUPERT[-1]

        if pd.isna(r1) or pd.isna(s1) or pd.isna(st_val) or pd.isna(st_dir):
            return

        prev_st_dir = self.data.SUPERTd[-2]
        supertrend_flip_to_bear = prev_st_dir == 1 and st_dir == -1
        supertrend_flip_to_bull = prev_st_dir == -1 and st_dir == 1

        if self.position:
            if self.position.is_long and (supertrend_flip_to_bear or close < s1):
                self.current_trade['exit_time'] = current_dt
                self.current_trade['exit_price'] = close
                self.current_trade['reason_for_exit'] = 'SuperTrend Flip' if supertrend_flip_to_bear else 'Pivot Breach'
                self.position.close()
                self.trade_finished()
                return

            if self.position.is_short and (supertrend_flip_to_bull or close > r1):
                self.current_trade['exit_time'] = current_dt
                self.current_trade['exit_price'] = close
                self.current_trade['reason_for_exit'] = 'SuperTrend Flip' if supertrend_flip_to_bull else 'Pivot Breach'
                self.position.close()
                self.trade_finished()
                return

        if self.position or self.trades_today >= self.max_trades_per_day:
            return

        if close > r1 and st_dir == 1:
            self.buy()
            self.trades_today += 1
            self.current_trade.update({
                'entry_time': current_dt,
                'entry_price': close,
                'position_type': 'LONG',
                'signal_price': close,
                'R1': r1,
                'S1': s1,
                'SUPERT': st_val,
                'SUPERTd': st_dir,
                'trade_date': current_date,
                'trade_number_today': self.trades_today,
            })
            return

        if close < s1 and st_dir == -1:
            self.sell()
            self.trades_today += 1
            self.current_trade.update({
                'entry_time': current_dt,
                'entry_price': close,
                'position_type': 'SHORT',
                'signal_price': close,
                'R1': r1,
                'S1': s1,
                'SUPERT': st_val,
                'SUPERTd': st_dir,
                'trade_date': current_date,
                'trade_number_today': self.trades_today,
            })
            return


def recovery_days(cum_pnl: pd.Series) -> int:
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    if drawdown.min() >= 0:
        return 0
    max_dd_idx = drawdown.idxmin()
    peak_before_dd = peak[:max_dd_idx].iloc[-1]
    dd_start_idx = peak[:max_dd_idx][peak[:max_dd_idx] == peak_before_dd].index[-1]
    recovered = cum_pnl[max_dd_idx:][cum_pnl[max_dd_idx:] >= peak_before_dd]
    if recovered.empty:
        return (cum_pnl.index[-1] - dd_start_idx).days
    return (recovered.index[0] - dd_start_idx).days


def main(csv_file: str, symbol: str = 'BTCUSD', timeframe: str = '5min',
         supertrend_length: int = 7, supertrend_multiplier: float = 3.0,
         max_trades_per_day: int = 3):

    if not os.path.exists(csv_file):
        print(f'Data file not found: {csv_file}')
        return

    df = pd.read_csv(
        csv_file,
        header=None,
        names=['Date', 'Time', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        format='%Y%m%d %H:%M'
    )
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    print('Consolidating spot candles...')
    consolidated = df.set_index('timestamp')
    # consolidated = ohlc_consolidate(df, timeframe, Isvolume=True)

    print('Computing pivot levels...')
    consolidated = compute_pivot_levels(consolidated)

    print('Computing SuperTrend...')
    consolidated = compute_supertrend(consolidated, length=supertrend_length, multiplier=supertrend_multiplier)

    consolidated.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)

    MomentumCryptoSpotStrategy.symbol = symbol
    MomentumCryptoSpotStrategy.supertrend_length = supertrend_length
    MomentumCryptoSpotStrategy.supertrend_multiplier = supertrend_multiplier
    MomentumCryptoSpotStrategy.max_trades_per_day = max_trades_per_day
    MomentumCryptoSpotStrategy.signals = []

    bt = Backtest(
        consolidated,
        MomentumCryptoSpotStrategy,
        cash=100_000,
        commission=0.0005,
        trade_on_close=True
    )

    stats = bt.run()
    print(stats)

    signals = MomentumCryptoSpotStrategy.signals
    if not signals:
        print('No trades executed in this backtest.')
        return

    trades_df = pd.DataFrame(signals)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
    trades_df['year'] = trades_df['exit_time'].dt.year

    monthly_pnl = trades_df.groupby('month')['profit'].sum()
    yearly_pnl = trades_df.groupby('year')['profit'].sum()
    monthly_trades = trades_df.groupby('month').size()
    cum_pnl = trades_df.set_index('exit_time')['profit'].cumsum()

    recovery = recovery_days(cum_pnl)
    highest_profit = trades_df['profit'].max()
    highest_loss = trades_df['profit'].min()
    roi = (yearly_pnl / 100_000) * 100

    summary_rows = []
    for period, pnl in monthly_pnl.items():
        summary_rows.append({'entry_time': None, 'exit_time': None, 'profit': pnl, 'position_type': f'Monthly PnL ({period})'})
    for year, pnl in yearly_pnl.items():
        summary_rows.append({'entry_time': None, 'exit_time': None, 'profit': pnl, 'position_type': f'Yearly PnL ({year})'})
    for year, value in roi.items():
        summary_rows.append({'entry_time': None, 'exit_time': None, 'profit': value, 'position_type': f'ROI% ({year})'})
    for period, count in monthly_trades.items():
        summary_rows.append({'entry_time': None, 'exit_time': None, 'profit': count, 'position_type': f'Trades ({period})'})
    summary_rows.append({'entry_time': None, 'exit_time': None, 'profit': recovery, 'position_type': 'Recovery Days'})
    summary_rows.append({'entry_time': None, 'exit_time': None, 'profit': highest_profit, 'position_type': 'Highest Profit'})
    summary_rows.append({'entry_time': None, 'exit_time': None, 'profit': highest_loss, 'position_type': 'Highest Loss'})

    summary_df = pd.DataFrame(summary_rows)
    final_df = pd.concat([trades_df, summary_df], ignore_index=True, sort=False)
    final_df.to_csv('BACKTEST_MOMENTUM_CRYPTO_SPOT.csv', index=False)

    print(f'Backtest complete. Results saved to BACKTEST_MOMENTUM_CRYPTO_SPOT.csv')
    print(f'Total trades: {len(trades_df)}')
    print(f'Cumulative PnL: {trades_df["profit"].sum():.2f}')


if __name__ == '__main__':
    main(
        csv_file=r'/workspaces/Crypto-Backtests/spot_data/BTCUSD_5m_20260101_20260610.csv',
        symbol='BTCUSD',
        timeframe='5min',
        supertrend_length=7,
        supertrend_multiplier=3.0,
        max_trades_per_day=3
    )
