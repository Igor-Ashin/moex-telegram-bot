import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MOEXClient:
    """Клиент для работы с MOEX API"""
    
    BASE_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/securities"
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def get_daily_data(self, ticker: str, days: int = 120) -> pd.DataFrame:
        """Получение дневных данных (перенесено из main.py get_moex_data)"""
        try:
            till = datetime.today().strftime('%Y-%m-%d')
            from_date = (datetime.today() - timedelta(days=days * 1.5)).strftime('%Y-%m-%d')
            
            url = f"{self.BASE_URL}/{ticker}/candles.json"
            params = {
                'interval': 24,
                'from': from_date,
                'till': till
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return self._process_candles_data(data, days)
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_4h_data(self, ticker: str, days: int = 200) -> pd.DataFrame:
        """Получение 4-часовых данных (перенесено из main.py get_moex_data_4h)"""
        try:
            till = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')
            from_date = (datetime.today() - timedelta(days=days * 1.5)).strftime('%Y-%m-%dT%H:%M:%S')
            
            url = f"{self.BASE_URL}/{ticker}/candles.json"
            params = {
                'interval': 4,
                'from': from_date,
                'till': till
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return self._process_candles_data(data, days)
            
        except Exception as e:
            logger.error(f"Error fetching 4h data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_weekly_data(self, ticker: str, weeks: int = 80) -> pd.DataFrame:
        """Получение недельных данных (актуализированная версия из main.py)"""
        try:
            till = datetime.today().strftime('%Y-%m-%d')
            from_date = (datetime.today() - timedelta(weeks=weeks * 1.5)).strftime('%Y-%m-%d')
            
            url = f"{self.BASE_URL}/{ticker}/candles.json"
            params = {
                'interval': 7,
                'from': from_date,
                'till': till
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            candles = data['candles']['data']
            columns = data['candles']['columns']
            df = pd.DataFrame(candles, columns=columns)
            df['begin'] = pd.to_datetime(df['begin'])
            df = df.sort_values('begin')
            df.set_index('begin', inplace=True)
            df = df.rename(columns={'close': 'close'})
            df = df[['close']].dropna()
            return df.tail(weeks)
            
        except Exception as e:
            logger.error(f"Error fetching weekly data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _process_candles_data(self, data: dict, limit: int) -> pd.DataFrame:
        """Обработка данных свечей"""
        try:
            candles = data['candles']['data']
            columns = data['candles']['columns']
            
            df = pd.DataFrame(candles, columns=columns)
            
            if df.empty:
                return df
                
            df['begin'] = pd.to_datetime(df['begin'])
            df = df.sort_values('begin')
            df.set_index('begin', inplace=True)
            
            # Переименовываем колонки
            df = df.rename(columns={
                'close': 'close',
                'volume': 'volume', 
                'high': 'high',
                'low': 'low'
            })
            
            df = df[['close', 'volume', 'high', 'low']].dropna()
            return df.tail(limit)
            
        except Exception as e:
            logger.error(f"Error processing candles data: {e}")
            return pd.DataFrame()

# Глобальный экземпляр клиента
moex_client = MOEXClient()