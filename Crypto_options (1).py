from datetime import datetime, timedelta
import requests
import os
import copy
import pandas as pd
import time
import warnings
from dateutil.parser import parse
warnings.filterwarnings("ignore")

# Define your options data storage path
def calculate_time_range(time_frame, max_candles=2000):
    """Calculate how far back we can fetch in one request based on timeframe"""
    timeframe_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360,
        '1d': 1440, '1w': 10080
    }
    
    if time_frame not in timeframe_minutes:
        raise ValueError(f"Invalid timeframe: {time_frame}")
    
    minutes_per_candle = timeframe_minutes[time_frame]
    # Use 1999 instead of 2000 to avoid edge cases
    total_minutes = minutes_per_candle * 1999
    return timedelta(minutes=total_minutes)

def convert_symbol_to_filename(symbol):
    """
    Convert Delta Exchange symbol to Bybit-style filename.
    e.g. C-BTC-81400-060425 -> BTC25040681400CE
         P-BTC-76000-140325 -> BTC25031476000PE
    """
    parts = symbol.split('-')
    opt_type_char = parts[0]   # C or P
    base = parts[1]            # BTC
    strike = parts[2]          # 81400
    exp_str = parts[3]         # 060425 (ddmmyy)
    
    # Parse ddmmyy -> yymmdd
    exp_date = datetime.strptime(exp_str, '%d%m%y')
    exp_fmt = exp_date.strftime('%y%m%d')  # yymmdd
    
    opt_type = 'CE' if opt_type_char == 'C' else 'PE'
    return f"{base}{exp_fmt}{strike}{opt_type}"

def fetch_options_historic(start_date: datetime, end_date: datetime, symbol, time_frame, data_type="MARK:"):
    """
    Fetch historical options data from Delta Exchange
    
    Args:
        start_date: Start date for historical data
        end_date: End date for historical data
        symbol: Option symbol (e.g., 'C-BTC-90000-310125')
        time_frame: Candle timeframe ('1m', '5m', '1h', etc.)
        data_type: Data type prefix ('MARK:', 'OI:', or '' for regular price)
    """
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'python-historical-data-client'
    }
    
    # Convert symbol to Bybit-style filename (e.g. BTC25040681400CE.csv)
    formatted_name = convert_symbol_to_filename(symbol)
    file_path = os.path.join(options_path, f"{formatted_name}.csv")
        
    # Check if data already exists
    if os.path.exists(file_path):
        print(f"Data already exists - {symbol}")
        return
        # user_input = input("Do you want to overwrite? (yes/no): ").strip().lower()
        # if user_input != 'yes':
        #     return
    # Calculate optimal time range per request
    time_range_per_request = calculate_time_range(time_frame)
    
    df_list = []
    current_end = copy.deepcopy(end_date)
    request_count = 0
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    print(f"Time range per request: {time_range_per_request}")
    
    while current_end > start_date:
        # Calculate start time for this request
        current_start = max(current_end - time_range_per_request, start_date)
        
        # Convert to Unix timestamps
        start_ts = int(current_start.timestamp())
        end_ts = int(current_end.timestamp())
        
        print(f"Fetching: {current_start} to {current_end}")
        
        try:
            # Make API request - Using India base URL
            response = requests.get(
                'https://api.india.delta.exchange/v2/history/candles',
                params={
                    'resolution': time_frame,
                    'symbol': f"{data_type}{symbol}",
                    'start': start_ts,
                    'end': end_ts
                },
                headers=headers,
                timeout=30
            )
            
            request_count += 1
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success') and 'result' in data:
                    result = data['result']
                    
                    if result:  # Check if result is not empty
                        sub_df = pd.DataFrame(result)
                        df_list.append(sub_df)
                        print(f"  ✓ Fetched {len(sub_df)} candles")
                    else:
                        print(f"  ⚠ No data available for this period")
                else:
                    print(f"  ✗ API Error: {data.get('error', 'Unknown error')}")
                    
            elif response.status_code == 429:
                print("  ⚠ Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                continue  # Retry the same request
                
            else:
                print(f"  ✗ HTTP Error {response.status_code}: {response.text}")
        
        except requests.exceptions.Timeout:
            print("  ✗ Request timeout. Retrying in 5 seconds...")
            time.sleep(5)
            continue
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            time.sleep(5)
        
        # Move to next time window
        current_end = current_start - timedelta(seconds=1)
        
        # Rate limiting: Sleep to avoid hitting rate limits
        # API weight is 3 per request, limit is per 5-minute window
        if request_count % 10 == 0:
            print("  ⏸ Pausing for rate limit (2 seconds)...")
            time.sleep(2)
    
    # Process and save data
    if df_list:
        print(f"\nProcessing {len(df_list)} data chunks...")
        df = pd.concat(df_list, ignore_index=True)
        
        # Remove duplicates based on timestamp
        df.drop_duplicates(subset=['time'], keep='first', inplace=True)
        
        # Sort by time
        df.sort_values('time', inplace=True)
        
        # Convert timestamps to IST
        df['global_time'] = pd.to_datetime(df['time'], unit='s')
        df['ist_time'] = df['global_time'] + pd.Timedelta('05:30:00')
        
        # Format Date (YYYYMMDD) and Time (HH:MM) columns
        df['Date'] = df['ist_time'].dt.strftime('%Y%m%d')
        df['Time'] = df['ist_time'].dt.strftime('%H:%M')
        
        # Rename OHLCV columns to match Bybit format
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        # Keep actual volume from API (don't overwrite with 0)
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        df['OI'] = 0
        
        # Select final columns in Bybit format
        cols = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']
        df = df[cols]
        
        # Save to CSV - append if file exists, otherwise create new
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, header=None, names=cols)
            df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates based on Date+Time (keep latest)
            df.drop_duplicates(subset=['Date', 'Time'], keep='last', inplace=True)
            df.sort_values(['Date', 'Time'], inplace=True)
        df.to_csv(file_path, index=False, header=False)
        print(f"✓ Data saved successfully: {file_path}")
        print(f"  Total candles: {len(df)}")
        print(f"  Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    else:
        print(f"✗ No data received for symbol: {symbol}")


def fetch_spot_historic(start_date: datetime, end_date: datetime, spot_symbol: str, time_frame: str, save_path: str):
    """
    Fetch historical spot/perpetual data from Delta Exchange and save as CSV.
    
    Args:
        start_date: Start date for historical data
        end_date: End date for historical data
        spot_symbol: Spot symbol (e.g., 'BTCUSD')
        time_frame: Candle timeframe ('1m', '5m', '1h', etc.)
        save_path: Full file path to save the CSV
    """
    if os.path.exists(save_path):
        print(f"Spot data already exists at {save_path}, skipping fetch.")
        return

    headers = {
        'Accept': 'application/json',
        'User-Agent': 'python-historical-data-client'
    }

    time_range_per_request = calculate_time_range(time_frame)

    df_list = []
    current_end = copy.deepcopy(end_date)
    request_count = 0

    print(f"\n{'='*60}")
    print(f"Fetching SPOT data for {spot_symbol} from {start_date} to {end_date}")
    print(f"Time range per request: {time_range_per_request}")
    print(f"{'='*60}")

    while current_end > start_date:
        current_start = max(current_end - time_range_per_request, start_date)

        start_ts = int(current_start.timestamp())
        end_ts = int(current_end.timestamp())

        print(f"Fetching: {current_start} to {current_end}")

        try:
            response = requests.get(
                'https://api.india.delta.exchange/v2/history/candles',
                params={
                    'resolution': time_frame,
                    'symbol': f"{spot_symbol}",
                    'start': start_ts,
                    'end': end_ts
                },
                headers=headers,
                timeout=30
            )

            request_count += 1

            if response.status_code == 200:
                data = response.json()

                if data.get('success') and 'result' in data:
                    result = data['result']

                    if result:
                        sub_df = pd.DataFrame(result)
                        df_list.append(sub_df)
                        print(f"  ✓ Fetched {len(sub_df)} candles")
                    else:
                        print(f"  ⚠ No data available for this period")
                else:
                    print(f"  ✗ API Error: {data.get('error', 'Unknown error')}")

            elif response.status_code == 429:
                print("  ⚠ Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                continue

            else:
                print(f"  ✗ HTTP Error {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            print("  ✗ Request timeout. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            time.sleep(5)

        current_end = current_start - timedelta(seconds=1)

        if request_count % 10 == 0:
            print("  ⏸ Pausing for rate limit (2 seconds)...")
            time.sleep(2)

    if df_list:
        print(f"\nProcessing {len(df_list)} spot data chunks...")
        df = pd.concat(df_list, ignore_index=True)
        df.drop_duplicates(subset=['time'], keep='first', inplace=True)
        df.sort_values('time', inplace=True)

        df['global_time'] = pd.to_datetime(df['time'], unit='s')
        df['datetime'] = df['global_time'] + pd.Timedelta('05:30:00')  # IST conversion

        df = df[["datetime", "open", "high", "low", "close", "volume"]]

        df.to_csv(save_path, index=False)
        print(f"✓ Spot data saved successfully: {save_path}")
        print(f"  Total candles: {len(df)}")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"{'='*60}\n")
    else:
        print(f"✗ No spot data received for symbol: {spot_symbol}")


def main(Options_dir,symbol,regular_date_range,strike_range,expiry:str,formated_symbols,step_size,time_frame,start_date,end_date):
    global options_path
    options_path = os.path.join(Options_dir,expiry.upper())
    os.makedirs(options_path,exist_ok=True)

    # Auto-generate unique spot data filename from inputs
    spot_symbol = formated_symbols.get(symbol, f"{symbol}USD")
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    spot_data_path = os.path.join(Options_dir, f"{spot_symbol}_{time_frame}_{start_str}_{end_str}.csv")
    print(f"Spot data file: {spot_data_path}")
    fetch_spot_historic(start_date, end_date, spot_symbol, time_frame, spot_data_path)
``
    df = pd.read_csv(spot_data_path,index_col=False,parse_dates=["datetime"])
    df = df[(df['datetime'] < end_date) & (df['datetime'] > start_date)]
    df["strike"] = ((df["close"]/step_size[symbol]).round() * step_size[symbol]).astype("int")
    

    df["date"] = df["datetime"].dt.date
    grouped_df = df.groupby(by="date")
    for _date,d_df in grouped_df:
        strikes = d_df["strike"].unique().tolist()
        timestamp = parse(str(_date)).replace(hour=0,minute=0,second=0,microsecond=0)

        start_date = timestamp
        if expiry == "regular":
            end_date = (timestamp + timedelta(regular_date_range))
        
        elif expiry == "monthly":

            if timestamp.date() <= timestamp.replace(day=15).date():
                end_date = (timestamp + pd.offsets.MonthEnd(0))
            else:
                end_date = (timestamp + pd.offsets.MonthEnd(1))
        else:
            print("Please select regular or monthly expiry")
            exit(0)
        # print("end_date == ",end_date)
        # exit(0)
        end_date= end_date.replace(hour=17,minute=30)
        for strike in strikes:
            call_symbol = f"C-{symbol}-{strike}-{end_date.strftime("%d%m%y")}"
            put_symbol = f"P-{symbol}-{strike}-{end_date.strftime("%d%m%y")}"
            fetch_options_historic(start_date,end_date,call_symbol,time_frame)
            fetch_options_historic(start_date,end_date,put_symbol,time_frame)
            if expiry in ["regular"]:
                for i in range(strike_range):
                    fsteps = (step_size[symbol] * (i+1))

                    upper_strike = (strike + fsteps)
                    lower_strike = (strike - fsteps)

                    call_symbol_up = f"C-{symbol}-{upper_strike}-{end_date.strftime("%d%m%y")}"
                    put_symbol_up = f"P-{symbol}-{upper_strike}-{end_date.strftime("%d%m%y")}"
                    fetch_options_historic(start_date,end_date,call_symbol_up,time_frame)
                    fetch_options_historic(start_date,end_date,put_symbol_up,time_frame)

                    call_symbol_lo = f"C-{symbol}-{lower_strike}-{end_date.strftime("%d%m%y")}"
                    put_symbol_lo = f"P-{symbol}-{lower_strike}-{end_date.strftime("%d%m%y")}"
                    fetch_options_historic(start_date,end_date,call_symbol_lo,time_frame)
                    fetch_options_historic(start_date,end_date,put_symbol_lo,time_frame)
if __name__ == "__main__":
    kwags = dict(
        # Base directory path where the fetched options data CSV files will be saved
        Options_dir = r"/workspaces/Crypto-Backtests/HISTORIC",
        
        # The base cryptocurrency symbol to fetch options for (e.g., 'BTC' or 'ETH')
        symbol = 'BTC',
        
        # Number of strikes to fetch above and below the ATM strike (e.g., 3 means 3 upper and 3 lower strikes)
        strike_range = 3,
        
        # The type of option expiry to fetch. Can be "regular" (daily), "monthly", or "weekly"
        expiry = "regular" ,#[montly or regular,weekly]
        
        # For 'regular' expiry, the number of days until the option expires (e.g., 3 means a 3-day expiry option)
        regular_date_range = 3,
        
        # Mapping of the base symbol to its corresponding Spot/Perpetual ticker symbol on the exchange
        formated_symbols={"BTC":"BTCUSD","ETH":"ETHUSD"},
        
        # The strike price interval/gap for each cryptocurrency (e.g., BTC strikes are available every $200)
        step_size = {"BTC":200,"ETH":100},        
        
        # The candle resolution/timeframe for the historical data (e.g., '1m' for 1-minute candles)
        time_frame = '1m',
        
        # The beginning date of the historical period to simulate/fetch data for
        start_date = datetime(2026,5,1,0,0,0),
        
        # The end date of the historical period to simulate/fetch data for
        end_date =  datetime(2026,6,10,23,59,59),
    )
    main(**kwags)


    