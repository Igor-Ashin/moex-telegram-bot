# caching.py - Модуль кэширования для MOEX бота

import time
import os
from datetime import datetime
import pandas as pd
import requests

# Конфигурация кэширования
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 минут по умолчанию
WEEKLY_CACHE_TTL = int(os.getenv("WEEKLY_CACHE_TTL", "600"))  # 10 минут
MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", "50"))
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"

# Глобальные кэши
moex_cache = {}
weekly_cache = {}
figi_cache = {}

def get_cache_key(ticker, days):
    return f"moex_{ticker}_{days}"

def is_cache_valid(cache_entry, ttl_seconds):
    return time.time() - cache_entry['timestamp'] < ttl_seconds

def cleanup_cache():
    # Ваш код очистки кэша здесь
    pass

def get_moex_data_with_cache(ticker="SBER", days=120):
    # Ваш код кэшированной функции здесь
    pass

def get_moex_weekly_data_with_cache(ticker="SBER", weeks=80):
    # Ваш код кэшированной функции здесь
    pass

def get_figi_by_ticker_with_cache(ticker: str):
    # Ваш код кэшированной функции здесь
    pass

def enable_caching():
    """Включает кэширование, заменяя оригинальные функции"""
    global get_moex_data, get_moex_weekly_data, get_figi_by_ticker
    
    # Импортируем оригинальные функции
    from moex_stock_bot import get_moex_data, get_moex_weekly_data, get_figi_by_ticker
    
    # Сохраняем ссылки на оригинальные функции
    globals()['_original_get_moex_data'] = get_moex_data
    globals()['_original_get_moex_weekly_data'] = get_moex_weekly_data
    globals()['_original_get_figi_by_ticker'] = get_figi_by_ticker
    
    # Заменяем функции в основном модуле
    import moex_stock_bot
    moex_stock_bot.get_moex_data = get_moex_data_with_cache
    moex_stock_bot.get_moex_weekly_data = get_moex_weekly_data_with_cache
    moex_stock_bot.get_figi_by_ticker = get_figi_by_ticker_with_cache
    
    print("✅ Кэширование включено")

# Автоматическое включение при импорте
if ENABLE_CACHING:
    enable_caching()
