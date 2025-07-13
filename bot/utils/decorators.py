import functools
import logging
import time
from typing import Callable, Dict, List
from collections import defaultdict
import asyncio
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

class RateLimiter:
    """Система ограничения частоты запросов"""
    
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[int, List[float]] = defaultdict(list)
        self._cleanup_task = None
        
        # Запускаем фоновую очистку старых записей
        asyncio.create_task(self._cleanup_old_requests())
    
    def is_allowed(self, user_id: int) -> bool:
        """Проверка, разрешен ли запрос для пользователя"""
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Удаляем старые запросы из текущего окна
        self.requests[user_id] = [
            req_time for req_time in user_requests 
            if now - req_time < self.window
        ]
        
        # Проверяем лимит
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Добавляем текущий запрос
        self.requests[user_id].append(now)
        return True
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Получение статистики по пользователю"""
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Считаем актуальные запросы
        recent_requests = [
            req_time for req_time in user_requests 
            if now - req_time < self.window
        ]
        
        return {
            'requests_in_window': len(recent_requests),
            'max_requests': self.max_requests,
            'window_seconds': self.window,
            'remaining_requests': max(0, self.max_requests - len(recent_requests)),
            'reset_time': max(recent_requests) + self.window if recent_requests else now
        }
    
    async def _cleanup_old_requests(self):
        """Периодическая очистка старых запросов"""
        while True:
            try:
                await asyncio.sleep(300)  # Каждые 5 минут
                now = time.time()
                
                # Очищаем старые записи
                for user_id in list(self.requests.keys()):
                    self.requests[user_id] = [
                        req_time for req_time in self.requests[user_id]
                        if now - req_time < self.window * 2  # Держим данные чуть дольше
                    ]
                    
                    # Удаляем пустые записи
                    if not self.requests[user_id]:
                        del self.requests[user_id]
                        
                logger.debug(f"Cleaned up rate limiter, active users: {len(self.requests)}")
                
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")

# Глобальный rate limiter
rate_limiter = RateLimiter(max_requests=15, window=60)  # 15 запросов в минуту

def rate_limit(func: Callable) -> Callable:
    """Декоратор для ограничения частоты запросов"""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        
        if not rate_limiter.is_allowed(user_id):
            stats = rate_limiter.get_user_stats(user_id)
            
            await update.message.reply_text(
                f"⏳ Слишком много запросов!\n"
                f"Лимит: {stats['max_requests']} запросов в {stats['window_seconds']} секунд\n"
                f"Попробуйте через {stats['window_seconds']} секунд"
            )
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return
        
        return await func(update, context, *args, **kwargs)
    
    return wrapper

def handle_errors(func: Callable) -> Callable:
    """Декоратор для обработки ошибок"""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        try:
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            
            # Отправляем пользователю дружелюбное сообщение
            error_message = "❌ Произошла ошибка при обработке запроса. Попробуйте позже."
            
            # Для разных типов ошибок - разные сообщения
            if "timeout" in str(e).lower():
                error_message = "⏱️ Превышено время ожидания. MOEX API временно недоступен."
            elif "connection" in str(e).lower():
                error_message = "🌐 Проблемы с соединением. Попробуйте через несколько минут."
            elif "json" in str(e).lower():
                error_message = "📊 Ошибка обработки данных. Попробуйте другой тикер."
            
            try:
                await update.message.reply_text(error_message)
            except:
                # Если даже отправка сообщения об ошибке не удалась
                logger.error("Failed to send error message to user")
    
    return wrapper

def log_command(func: Callable) -> Callable:
    """Декоратор для логирования команд"""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        command = func.__name__
        start_time = time.time()
        
        logger.info(f"Command '{command}' started by user {user_id} ({username})")
        
        try:
            result = await func(update, context, *args, **kwargs)
            
            duration = time.time() - start_time
            logger.info(f"Command '{command}' completed for user {user_id} in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Command '{command}' failed for user {user_id} after {duration:.3f}s: {e}")
            raise
    
    return wrapper

def performance_monitor(endpoint: str):
    """Декоратор для мониторинга производительности"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                # Логируем медленные операции
                if duration > 5.0:
                    logger.warning(f"Slow operation: {endpoint} took {duration:.3f}s")
                elif duration > 2.0:
                    logger.info(f"Operation: {endpoint} took {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Operation failed: {endpoint} after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator

def cache_result(ttl: int = 300):
    """
    Декоратор для кэширования результатов функций
    ttl: время жизни кэша в секундах
    """
    def decorator(func: Callable):
        cache = {}
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Создаем ключ кэша из аргументов
            cache_key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            # Проверяем кэш
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if now - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    # Удаляем просроченную запись
                    del cache[cache_key]
            
            # Вычисляем результат
            result = await func(*args, **kwargs)
            
            # Сохраняем в кэш
            cache[cache_key] = (result, now)
            logger.debug(f"Cached result for {func.__name__}")
            
            # Ограничиваем размер кэша
            if len(cache) > 100:
                # Удаляем 20 самых старых записей
                oldest_keys = sorted(
                    cache.keys(),
                    key=lambda k: cache[k][1]
                )[:20]
                for key in oldest_keys:
                    del cache[key]
            
            return result
        
        return wrapper
    return decorator

# Комбинированный декоратор для команд бота
def telegram_handler(func: Callable) -> Callable:
    """
    Комбинированный декоратор для обработчиков команд
    Включает: rate limiting, error handling, logging
    """
    return log_command(handle_errors(rate_limit(func)))

# Декоратор для быстрых команд (без rate limiting)
def fast_telegram_handler(func: Callable) -> Callable:
    """
    Декоратор для быстрых команд (без rate limiting)
    Включает: error handling, logging
    """
    return log_command(handle_errors(func))

def typing_action(func: Callable) -> Callable:
    """Декоратор для показа индикатора "печатает" """
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        # Показываем индикатор "печатает"
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        
        return await func(update, context, *args, **kwargs)
    
    return wrapper