"""
Fetch Crypto Spot Data for MomentumCryptoSpot.py
================================================
This script pulls historical spot candle data from Delta Exchange and saves it
as a CSV file formatted for use with MomentumCryptoSpot.py.

Output format (no header):
    Date,Time,open,high,low,close,volume
    20260601,00:00,27000.0,27120.0,26980.0,27050.0,12.34
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests


TIMEFRAME_MINUTES = {
    '1m': 1,
    '3m': 3,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '6h': 360,
    '1d': 1440,
    '1w': 10080,
}


def calculate_time_range(timeframe: str, max_candles: int = 1999) -> timedelta:
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    minutes_per_candle = TIMEFRAME_MINUTES[timeframe]
    return timedelta(minutes=minutes_per_candle * max_candles)


def fetch_spot_historic(symbol: str, timeframe: str, start_date: datetime, end_date: datetime):
    print(f"Starting fetch for {symbol} from {start_date} to {end_date} with timeframe {timeframe}")
    """Fetch historical spot candles from Delta Exchange."""
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'python-historical-data-client'
    }

    time_range = calculate_time_range(timeframe)
    current_end = end_date
    candle_frames = []
    request_count = 0

    while current_end > start_date:
        current_start = max(current_end - time_range, start_date)
        start_ts = int(current_start.timestamp())
        end_ts = int(current_end.timestamp())

        response = requests.get(
            'https://api.india.delta.exchange/v2/history/candles',
            params={
                'resolution': timeframe,
                'symbol': symbol,
                'start': start_ts,
                'end': end_ts,
            },
            headers=headers,
            timeout=30,
        )

        request_count += 1

        if response.status_code == 200:
            payload = response.json()
            if payload.get('success') and 'result' in payload:
                result = payload['result']
                if result:
                    candle_frames.append(pd.DataFrame(result))
                    print(f"Fetched {len(result)} candles: {current_start} to {current_end}")
                else:
                    print(f"No candles returned for period: {current_start} to {current_end}")
            else:
                raise RuntimeError(f"API error: {payload.get('error', payload)}")
        elif response.status_code == 429:
            print('Rate limit hit; waiting 60 seconds before retrying...')
            time.sleep(60)
            continue
        else:
            raise RuntimeError(f"HTTP error {response.status_code}: {response.text}")

        current_end = current_start - timedelta(seconds=1)
        if request_count % 10 == 0:
            print('Pausing to avoid rate limits...')
            time.sleep(2)

    if not candle_frames:
        raise RuntimeError('No spot data fetched for the requested range.')

    df = pd.concat(candle_frames, ignore_index=True)
    df.drop_duplicates(subset=['time'], keep='first', inplace=True)
    df.sort_values('time', inplace=True)
    return df


def format_for_momentum_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert API candle data into CSV format for MomentumCryptoSpot.py."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['Date'] = df['datetime'].dt.strftime('%Y%m%d')
    df['Time'] = df['datetime'].dt.strftime('%H:%M')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    output_df = df[['Date', 'Time', 'open', 'high', 'low', 'close', 'volume']]
    return output_df


def save_spot_csv(df: pd.DataFrame, output_path: str, overwrite: bool = False):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")
    df.to_csv(output_path, index=False, header=False)
    print(f"Saved spot CSV: {output_path}")


def build_default_output(symbol: str, timeframe: str, start_date: datetime, end_date: datetime, output_dir: str = 'spot_data') -> str:
    symbol_safe = symbol.replace('/', '').replace('-', '').upper()
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir,
        f"{symbol_safe}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    )


def main(symbol: str,
         timeframe: str,
         start_date: datetime,
         end_date: datetime,
         output_dir: str = 'spot_data',
         overwrite: bool = False):

    if end_date <= start_date:
        raise ValueError('End date must be later than start date.')

    output_path = build_default_output(symbol, timeframe, start_date, end_date, output_dir)
    print(f"Fetching spot data for {symbol} from {start_date} to {end_date} using {timeframe} candles")

    df = fetch_spot_historic(symbol, timeframe, start_date.replace(hour=17,minute=30), end_date.replace(hour=17,minute=30))
    formatted = format_for_momentum_csv(df)
    save_spot_csv(formatted, output_path, overwrite=overwrite)

    print('Spot data fetch complete. Use this file as csv_file in MomentumCryptoSpot.py')
    return output_path


if __name__ == '__main__':
    kwags = dict(
        symbol='BTCUSD',
        timeframe='5m',
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 6, 10),
        output_dir='/workspaces/Crypto-Backtests/spot_data',
        overwrite=False,
    )
    main(**kwags)
