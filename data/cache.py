import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

class DataCache:
    """
    Кэш для данных рынка с fallback на память
    Основная оптимизация: избегаем повторных запросов к MOEX API
    """
    
    def __init__(self, redis_url: str = None):
        self.redis = None
        self._fallback_cache: Dict[str, tuple] = {}
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        # Пытаемся подключиться к Redis (опционально)
        if redis_url:
            try:
                import redis
                self.redis = redis.from_url(redis_url)
                self.redis.ping()
                logger.info("Redis connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
    
    def _make_key(self, ticker: str, interval: str, days: int) -> str:
        """Генерация уникального ключа кэша"""
        return f"market_data:{ticker}:{interval}:{days}"
    
    def get_market_data(self, ticker: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """
        Получение данных из кэша
        Returns: DataFrame или None если данных нет
        """
        key = self._make_key(ticker, interval, days)
        
        try:
            # Пробуем Redis
            if self.redis:
                cached = self.redis.get(key)
                if cached:
                    data = json.loads(cached)
                    df = pd.DataFrame(data['data'])
                    if not df.empty:
                        df.index = pd.to_datetime(data['index'])
                        self._cache_stats['hits'] += 1
                        logger.debug(f"Redis cache HIT for {key}")
                        return df
            
            # Fallback на память
            if key in self._fallback_cache:
                cached_data, timestamp = self._fallback_cache[key]
                # Проверяем TTL (5 минут)
                if datetime.now() - timestamp < timedelta(minutes=5):
                    self._cache_stats['hits'] += 1
                    logger.debug(f"Memory cache HIT for {key}")
                    return cached_data
                else:
                    # Удаляем просроченные данные
                    del self._fallback_cache[key]
                    
        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
        
        self._cache_stats['misses'] += 1
        logger.debug(f"Cache MISS for {key}")
        return None
    
    def set_market_data(self, ticker: str, interval: str, days: int, 
                       df: pd.DataFrame, ttl: int = 300) -> None:
        """
        Сохранение данных в кэш
        ttl: время жизни в секундах (по умолчанию 5 минут)
        """
        if df.empty:
            return
            
        key = self._make_key(ticker, interval, days)
        
        try:
            # Подготавливаем данные для сериализации
            data = {
                'data': df.to_dict('records'),
                'index': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'cached_at': datetime.now().isoformat()
            }
            
            # Сохраняем в Redis
            if self.redis:
                self.redis.setex(key, ttl, json.dumps(data, default=str))
                logger.debug(f"Cached in Redis: {key} (TTL: {ttl}s)")
            
            # Всегда сохраняем в память как fallback
            self._fallback_cache[key] = (df.copy(), datetime.now())
            logger.debug(f"Cached in memory: {key}")
            
            # Ограничиваем размер cache в памяти
            if len(self._fallback_cache) > 100:
                # Удаляем самые старые записи
                oldest_keys = sorted(
                    self._fallback_cache.keys(),
                    key=lambda k: self._fallback_cache[k][1]
                )[:20]
                for old_key in oldest_keys:
                    del self._fallback_cache[old_key]
                    
        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
    
    def invalidate_ticker(self, ticker: str) -> int:
        """
        Инвалидация всех данных по тикеру
        Returns: количество удаленных записей
        """
        removed_count = 0
        
        try:
            # Очистка Redis
            if self.redis:
                pattern = f"market_data:{ticker}:*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
                    removed_count += len(keys)
                    logger.info(f"Invalidated {len(keys)} Redis entries for {ticker}")
            
            # Очистка памяти
            keys_to_remove = [
                k for k in self._fallback_cache.keys() 
                if k.startswith(f"market_data:{ticker}:")
            ]
            for key in keys_to_remove:
                del self._fallback_cache[key]
                removed_count += 1
            
            logger.info(f"Invalidated {removed_count} total cache entries for {ticker}")
            
        except Exception as e:
            logger.error(f"Cache invalidation error for {ticker}: {e}")
        
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'memory_cache_size': len(self._fallback_cache),
            'redis_connected': self.redis is not None and self.redis.ping()
        }
    
    def clear_all(self) -> None:
        """Полная очистка кэша"""
        try:
            if self.redis:
                # Удаляем только наши ключи
                pattern = "market_data:*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
                    logger.info(f"Cleared {len(keys)} Redis entries")
            
            self._fallback_cache.clear()
            self._cache_stats = {'hits': 0, 'misses': 0}
            logger.info("Cache cleared completely")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

# Глобальный экземпляр кэша
cache = DataCache()