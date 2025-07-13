import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging
import time
from data.cache import cache

logger = logging.getLogger(__name__)

class OptimizedMOEXClient:
    """
    Оптимизированный клиент для работы с MOEX API
    Основные оптимизации:
    1. Кэширование данных
    2. Переиспользование сессий
    3. Батчевая обработка
    4. Мониторинг производительности
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MOEX-Bot/2.0',
            'Accept': 'application/json'
        })
        self._stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
    
    def get_daily_data(self, ticker: str, days: int = 120) -> pd.DataFrame:
        """
        Получение дневных данных с кэшированием
        """
        start_time = time.time()
        
        # Проверяем кэш
        cached_df = cache.get_market_data(ticker, "daily", days)
        if cached_df is not None:
            self._stats['cache_hits'] += 1
            logger.debug(f"Using cached data for {ticker}")
            return cached_df
        
        try:
            # Запрос к API
            self._stats['api_calls'] += 1
            till = datetime.today().strftime('%Y-%m-%d')
            from_date = (datetime.today() - timedelta(days=days * 1.5)).strftime('%Y-%m-%d')
            
            url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"
            params = {
                'interval': 24,
                'from': from_date,
                'till': till
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            df = self._process_candles_data(data, days)
            
            # Кэшируем результат
            if not df.empty:
                cache.set_market_data(ticker, "daily", days, df, ttl=300)
            
            duration = time.time() - start_time
            self._stats['total_time'] += duration
            logger.debug(f"Fetched data for {ticker} in {duration:.3f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_multiple_daily_data(self, tickers: List[str], days: int = 120) -> Dict[str, pd.DataFrame]:
        """
        Оптимизированное получение данных для нескольких тикеров
        Использует кэш для минимизации API вызовов
        """
        results = {}
        uncached_tickers = []
        
        # Сначала проверяем кэш для всех тикеров
        for ticker in tickers:
            cached_df = cache.get_market_data(ticker, "daily", days)
            if cached_df is not None:
                results[ticker] = cached_df
                self._stats['cache_hits'] += 1
            else:
                uncached_tickers.append(ticker)
        
        logger.info(f"Cache hits: {len(results)}/{len(tickers)}, API calls needed: {len(uncached_tickers)}")
        
        # Получаем некэшированные данные батчами
        batch_size = 10  # Ограничиваем нагрузку на API
        for i in range(0, len(uncached_tickers), batch_size):
            batch = uncached_tickers[i:i + batch_size]
            
            for ticker in batch:
                df = self.get_daily_data(ticker, days)
                results[ticker] = df
                
                # Небольшая пауза между запросами
                time.sleep(0.1)
            
            # Пауза между батчами
            if i + batch_size < len(uncached_tickers):
                time.sleep(0.5)
        
        return results
    
    def get_4h_data(self, ticker: str, days: int = 200) -> pd.DataFrame:
        """Получение 4-часовых данных с кэшированием"""
        # Проверяем кэш
        cached_df = cache.get_market_data(ticker, "4h", days)
        if cached_df is not None:
            self._stats['cache_hits'] += 1
            return cached_df
        
        try:
            self._stats['api_calls'] += 1
            till = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')
            from_date = (datetime.today() - timedelta(days=days * 1.5)).strftime('%Y-%m-%dT%H:%M:%S')
            
            url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"
            params = {
                'interval': 4,
                'from': from_date,
                'till': till
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            df = self._process_candles_data(data, days)
            
            # Кэшируем с коротким TTL для 4H данных
            if not df.empty:
                cache.set_market_data(ticker, "4h", days, df, ttl=900)  # 15 минут
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching 4h data for {ticker}: {e}")
            return pd.DataFrame()
    
    def warm_up_cache(self, popular_tickers: List[str]) -> None:
        """
        Прогрев кэша для популярных тикеров
        Вызывается в фоновых задачах
        """
        logger.info(f"Warming up cache for {len(popular_tickers)} popular tickers...")
        
        start_time = time.time()
        success_count = 0
        
        for ticker in popular_tickers:
            try:
                # Загружаем дневные данные
                df_daily = self.get_daily_data(ticker, days=120)
                if not df_daily.empty:
                    success_count += 1
                
                # Загружаем 4H данные для самых популярных
                if ticker in ["SBER", "GAZP", "LKOH", "YDEX"]:
                    self.get_4h_data(ticker, days=25)
                
                # Пауза между запросами
                time.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Failed to warm up cache for {ticker}: {e}")
        
        duration = time.time() - start_time
        logger.info(f"Cache warm-up completed: {success_count}/{len(popular_tickers)} in {duration:.1f}s")
    
    def _process_candles_data(self, data: dict, limit: int) -> pd.DataFrame:
        """Обработка данных свечей (без изменений)"""
        try:
            candles = data['candles']['data']
            columns = data['candles']['columns']
            
            df = pd.DataFrame(candles, columns=columns)
            
            if df.empty:
                return df
                
            df['begin'] = pd.to_datetime(df['begin'])
            df = df.sort_values('begin')
            df.set_index('begin', inplace=True)
            
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
    
    def get_performance_stats(self) -> Dict:
        """Получение статистики производительности"""
        total_requests = self._stats['api_calls'] + self._stats['cache_hits']
        cache_hit_rate = (self._stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        avg_response_time = (self._stats['total_time'] / self._stats['api_calls']) if self._stats['api_calls'] > 0 else 0
        
        return {
            'total_requests': total_requests,
            'api_calls': self._stats['api_calls'],
            'cache_hits': self._stats['cache_hits'],
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'avg_api_response_time': f"{avg_response_time:.3f}s",
            'total_api_time': f"{self._stats['total_time']:.3f}s"
        }
    
    def close(self):
        """Закрытие сессии"""
        self.session.close()

# Глобальный экземпляр оптимизированного клиента
optimized_moex_client = OptimizedMOEXClient()