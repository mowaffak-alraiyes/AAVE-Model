"""
Data fetching module for AAVE cryptocurrency price data.
Supports multiple data sources including CoinGecko and Binance.
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# ccxt is optional - only needed for Binance data source
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class AAVEDataFetcher:
    """Fetches AAVE price data from various sources."""
    
    def __init__(self, source='coingecko'):
        """
        Initialize data fetcher.
        
        Args:
            source: 'coingecko', 'binance', 'coinbase', or 'cryptocompare'
        """
        self.source = source
        if source == 'binance':
            if not CCXT_AVAILABLE:
                raise ImportError("ccxt library is required for Binance data source. Install it with: pip install ccxt")
            self.exchange = ccxt.binance()
    
    def fetch_coingecko_data(self, days=365):
        """
        Fetch AAVE data from CoinGecko API (free, no API key needed).
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        url = "https://api.coingecko.com/api/v3/coins/aave/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        # Add headers to avoid rate limiting
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        try:
            # Add small delay to avoid rate limiting
            time.sleep(1)
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 401:
                print("CoinGecko API returned 401. This might be a rate limit. Trying with smaller date range...")
                # Try with smaller range
                params['days'] = min(days, 365)
                time.sleep(2)
                response = requests.get(url, params=params, headers=headers, timeout=30)
            
            response.raise_for_status()
            data = response.json()
            
            # Extract prices
            prices = data['prices']
            market_caps = data['market_caps']
            volumes = data['total_volumes']
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add market cap and volume
            df['market_cap'] = [mc[1] for mc in market_caps]
            df['volume'] = [vol[1] for vol in volumes]
            
            # For OHLC, we'll approximate from close prices
            # (CoinGecko free API doesn't provide full OHLC)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df['close'] * (1 + np.random.uniform(0, 0.05, len(df)))
            df['low'] = df['close'] * (1 - np.random.uniform(0, 0.05, len(df)))
            
            # Reorder columns
            df = df[['open', 'high', 'low', 'close', 'volume', 'market_cap']]
            
            return df
            
        except Exception as e:
            print(f"Error fetching CoinGecko data: {e}")
            return None
    
    def fetch_binance_data(self, timeframe='1d', limit=1000):
        """
        Fetch AAVE data from Binance.
        
        Args:
            timeframe: '1d', '4h', '1h', etc.
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            symbol = 'AAVE/USDT'
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching Binance data: {e}")
            return None
    
    def fetch_coinbase_data(self, timeframe='1d', days=30):
        """
        Fetch AAVE data from Coinbase Pro API using cbpro library.
        Falls back to generating synthetic data from daily candles if API fails.
        
        Args:
            timeframe: '1m', '5m', '15m', '30m', '1h', '6h', '1d'
            days: Number of days of data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try using cbpro library first
        try:
            try:
                import cbpro
            except ImportError:
                # Try alternative import name
                import coinbasepro as cbpro
            
            granularity_map = {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '30m': 1800,
                '1h': 3600,
                '6h': 21600,
                '1d': 86400
            }
            
            granularity = granularity_map.get(timeframe, 3600)
            
            # Calculate start and end times
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Use cbpro PublicClient (no API key needed for public data)
            client = cbpro.PublicClient()
            
            # Get historic rates
            # Note: Coinbase uses ISO format dates
            start_iso = start_time.isoformat()
            end_iso = end_time.isoformat()
            
            print(f"  Fetching from Coinbase Pro API (cbpro)...")
            candles = client.get_product_historic_rates(
                'AAVE-USD',
                start=start_iso,
                end=end_iso,
                granularity=granularity
            )
            
            if candles and len(candles) > 0:
                # Coinbase returns: [time, low, high, open, close, volume]
                df = pd.DataFrame(candles, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                print(f"  ✓ Fetched {len(df)} candles from Coinbase")
                return df
        except ImportError:
            print("  cbpro not available, trying direct API...")
        except Exception as e:
            print(f"  Coinbase cbpro failed: {e}")
            print("  Trying direct API...")
        
        # Fallback: Try direct API call
        try:
            granularity_map = {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '30m': 1800,
                '1h': 3600,
                '6h': 21600,
                '1d': 86400
            }
            
            granularity = granularity_map.get(timeframe, 3600)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Use Unix timestamps
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            url = f"https://api.exchange.coinbase.com/products/AAVE-USD/candles"
            params = {
                'start': start_ts,
                'end': end_ts,
                'granularity': granularity
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            time.sleep(1)
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                # Coinbase returns: [time, low, high, open, close, volume]
                df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                return df
        except Exception as e:
            print(f"  Direct API also failed: {e}")
        
        # Fallback: Generate synthetic shorter timeframe data from daily data
        print(f"  Coinbase API failed, generating synthetic {timeframe} data from daily candles...")
        try:
            # Get daily data first
            daily_df = self.fetch_coingecko_data(days=days)
            if daily_df is None or len(daily_df) == 0:
                return None
            
            # Resample daily data to shorter timeframes
            return self._resample_to_timeframe(daily_df, timeframe)
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return None
    
    def _resample_to_timeframe(self, daily_df, timeframe):
        """
        Resample daily data to shorter timeframes with realistic volatility.
        Creates intraday price movements with proper OHLC structure.
        """
        import numpy as np
        
        daily_df = daily_df.copy()
        
        # Calculate daily volatility
        daily_returns = daily_df['close'].pct_change().dropna()
        avg_volatility = daily_returns.std()
        
        # Create minute-by-minute data with realistic intraday movements
        minute_df = daily_df.resample('1min').asfreq()
        
        # Fill with interpolation and add realistic noise
        for col in ['open', 'high', 'low', 'close']:
            minute_df[col] = minute_df[col].interpolate(method='time')
            
            # Add realistic intraday volatility (smaller movements within day)
            intraday_vol = avg_volatility / np.sqrt(1440)  # Scale down for minute data
            noise = np.random.normal(0, intraday_vol, len(minute_df))
            minute_df[col] = minute_df[col] * (1 + noise)
        
        # Ensure OHLC relationships are maintained
        minute_df['high'] = minute_df[['open', 'high', 'low', 'close']].max(axis=1)
        minute_df['low'] = minute_df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Distribute volume across minutes (more volume during day, less at night)
        minute_df['volume'] = minute_df['volume'] / 1440  # Distribute evenly for now
        
        # Resample to target timeframe
        timeframe_map = {
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '6h': '6H',
            '1d': '1D'
        }
        
        freq = timeframe_map.get(timeframe, '30min')
        
        # Resample with proper OHLC aggregation
        resampled = minute_df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def fetch_cryptocompare_data(self, timeframe='30m', days=7):
        """
        Fetch AAVE data from CryptoCompare API (free, no API key needed).
        Makes multiple requests if needed to get more historical data.
        
        Args:
            timeframe: '5m', '15m', '30m', '1h', '4h', '1d'
            days: Number of days of data (max depends on timeframe)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map timeframes to CryptoCompare aggregate values
            timeframe_map = {
                '1m': (1, 'minute'),
                '5m': (5, 'minute'),
                '15m': (15, 'minute'),
                '30m': (30, 'minute'),
                '1h': (1, 'hour'),
                '4h': (4, 'hour'),
                '1d': (1, 'day')
            }
            
            aggregate, period = timeframe_map.get(timeframe, (30, 'minute'))
            
            # Calculate limit based on timeframe and days
            # CryptoCompare free tier: max 2000 data points per request
            if period == 'minute':
                limit = min(days * 24 * 60 // aggregate, 2000)
            elif period == 'hour':
                limit = min(days * 24 // aggregate, 2000)
            else:
                limit = min(days, 2000)
            
            # Ensure we get at least some data
            if limit < 100:
                limit = min(100, 2000)
            
            # For more historical data, make multiple requests
            all_candles = []
            max_requests = 5  # Limit to avoid rate limiting
            days_per_request = days // max_requests if days > 30 else days
            
            if days > 30 and period == 'minute':
                # Make multiple requests to get more history
                print(f"  Fetching {days} days in multiple requests...")
                for i in range(max_requests):
                    request_days = min(days_per_request, 30)  # Max 30 days per request
                    if request_days < 1:
                        break
                    
                    # Calculate limit for this request
                    if period == 'minute':
                        req_limit = min(request_days * 24 * 60 // aggregate, 2000)
                    else:
                        req_limit = min(request_days, 2000)
                    
                    if req_limit < 10:
                        break
            
            url = "https://min-api.cryptocompare.com/data/v2/histominute" if period == 'minute' else \
                  "https://min-api.cryptocompare.com/data/v2/histohour" if period == 'hour' else \
                  "https://min-api.cryptocompare.com/data/v2/histoday"
            
            # Single request for smaller timeframes or first request
            params = {
                'fsym': 'AAVE',
                'tsym': 'USD',
                'limit': limit,
                'aggregate': aggregate if period != 'day' else 1
            }
            
            print(f"  Fetching from CryptoCompare API (limit={limit})...")
            time.sleep(0.5)  # Rate limiting
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('Response') == 'Success' and data.get('Data', {}).get('Data'):
                candles = data['Data']['Data']
                
                # If we got max limit and need more, try to get older data
                if len(candles) >= 1900 and days > 30 and period == 'minute':
                    print(f"  Got {len(candles)} candles, requesting more historical data...")
                    # Get timestamp of oldest candle
                    oldest_ts = candles[0][0] if candles else None
                    if oldest_ts:
                        # Request data before this timestamp
                        params['toTs'] = oldest_ts - 1
                        time.sleep(1)
                        response2 = requests.get(url, params=params, timeout=30)
                        if response2.status_code == 200:
                            data2 = response2.json()
                            if data2.get('Response') == 'Success' and data2.get('Data', {}).get('Data'):
                                older_candles = data2['Data']['Data']
                                # Combine (older first, then newer)
                                candles = older_candles + candles
                                print(f"  Combined: {len(candles)} total candles")
                
                if not candles or len(candles) == 0:
                    print(f"  ❌ No candles in response")
                    return None
                
                # CryptoCompare returns: [time, open, high, low, close, volumefrom, volumeto]
                # Note: Some fields might be missing, handle gracefully
                try:
                    # Create DataFrame and handle variable column counts
                    df = pd.DataFrame(candles)
                    
                    # CryptoCompare format: time, open, high, low, close, volumefrom, volumeto (7 columns)
                    # But sometimes returns 9 columns (with conversion info)
                    num_cols = len(df.columns)
                    
                    if num_cols >= 7:
                        # Use first 7 columns
                        df = df.iloc[:, :7]
                        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto']
                    elif num_cols >= 5:
                        # At least get OHLC
                        df = df.iloc[:, :5]
                        df.columns = ['timestamp', 'open', 'high', 'low', 'close']
                        df['volumeto'] = 0
                        df['volumefrom'] = 0
                    else:
                        print(f"  ❌ Unexpected column count: {num_cols}")
                        return None
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                    df = df.dropna(subset=['timestamp'])  # Remove invalid timestamps
                    df.set_index('timestamp', inplace=True)
                    
                    # Use volumeto (volume in USD) as volume, or volumefrom if not available
                    if 'volumeto' in df.columns:
                        df['volume'] = df['volumeto']
                    elif 'volumefrom' in df.columns:
                        df['volume'] = df['volumefrom']
                    else:
                        df['volume'] = 0
                    
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_index()
                    
                    print(f"  ✓ Fetched {len(df)} {timeframe} candles from CryptoCompare")
                    if len(df) > 0:
                        print(f"    Date range: {df.index[0]} to {df.index[-1]}")
                    return df
                except Exception as e:
                    print(f"  ❌ Error processing CryptoCompare data: {e}")
                    return None
            else:
                print(f"  ❌ No data returned from CryptoCompare")
                return None
                
        except Exception as e:
            print(f"  Error fetching CryptoCompare data: {e}")
            return None
    
    def fetch_data(self, days=365, timeframe='1d'):
        """
        Fetch AAVE data from the configured source.
        
        Args:
            days: Number of days (for CoinGecko/Coinbase)
            timeframe: Timeframe for Binance/Coinbase ('5m', '30m', '1h', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.source == 'coingecko':
            return self.fetch_coingecko_data(days=days)
        elif self.source == 'binance':
            limit = min(days, 1000)  # Binance limit
            return self.fetch_binance_data(timeframe=timeframe, limit=limit)
        elif self.source == 'coinbase':
            return self.fetch_coinbase_data(timeframe=timeframe, days=days)
        elif self.source == 'cryptocompare':
            return self.fetch_cryptocompare_data(timeframe=timeframe, days=days)
        else:
            raise ValueError(f"Unknown source: {self.source}. Use 'coingecko', 'binance', 'coinbase', or 'cryptocompare'")


if __name__ == "__main__":
    # Test data fetching
    fetcher = AAVEDataFetcher(source='coingecko')
    df = fetcher.fetch_data(days=365)
    if df is not None:
        print(f"Fetched {len(df)} days of data")
        print(df.head())
        print(df.tail())

