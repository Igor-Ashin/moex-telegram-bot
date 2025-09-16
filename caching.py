# caching.py - ПОЛНАЯ ВЕРСИЯ

import time
import os
from datetime import datetime
import pandas as pd
import requests
import json

# Конфигурация кэширования
#CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 минут
#WEEKLY_CACHE_TTL = int(os.getenv("WEEKLY_CACHE_TTL", "600"))  # 10 минут
#MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", "1000"))
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"

CACHE_FILE = "figi_cache.json"

# Глобальные кэши
#moex_cache = {}
#weekly_cache = {}
figi_cache = {}

#def get_cache_key(ticker, days):
#    return f"moex_{ticker}_{days}"

#def is_cache_valid(cache_entry, ttl_seconds):
#    return time.time() - cache_entry['timestamp'] < ttl_seconds

#def cleanup_cache():
#    Удаляет устаревшие записи из кэша
#    current_time = time.time()
"""    
    # Очистка основного кэша
    keys_to_remove = []
    for key, entry in list(moex_cache.items()):
        if current_time - entry['timestamp'] > CACHE_TTL * 2:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del moex_cache[key]
    
    # Очистка недельного кэша
    keys_to_remove = []
    for key, entry in list(weekly_cache.items()):
        if current_time - entry['timestamp'] > WEEKLY_CACHE_TTL * 2:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del weekly_cache[key]
    
    # Ограничиваем размер кэшей
    if len(moex_cache) > MAX_CACHE_ENTRIES:
        sorted_items = sorted(moex_cache.items(), key=lambda x: x[1]['timestamp'])
        for key, _ in sorted_items[:10]:
            del moex_cache[key]

"""

def load_figi_cache():
    """Загружаем кэш из файла при старте"""
    global figi_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                figi_cache = json.load(f)
            print("✅ figi_cache загружен из файла")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки figi_cache: {e}")
            figi_cache = {}
    else:
        print("ℹ️ Файл figi_cache.json не найден, создаём пустой кэш")


def save_figi_cache():
    """Сохраняем figi_cache в файл"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(figi_cache, f, ensure_ascii=False, indent=2)
        print("💾 figi_cache сохранён в файл")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения figi_cache: {e}")


def activate_caching_if_enabled():
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    print(f"🔍 ENABLE_CACHING = {ENABLE_CACHING}")

    if not ENABLE_CACHING:
        print("ℹ️ Кэширование отключено через переменные окружения.")
        return False

    # ВКЛЮЧАЕМ КЭШ
    success = enable_caching()
    return success


"""
def get_moex_data_with_cache(ticker="SBER", days=120):
   # Кэшированная версия get_moex_data
    cache_key = get_cache_key(ticker, days)
    
    # Проверяем кэш
    if cache_key in moex_cache:
        cache_entry = moex_cache[cache_key]
        if is_cache_valid(cache_entry, CACHE_TTL):
            print(f"📋 Данные для {ticker} взяты из кэша")
            return cache_entry['data']
    
    # Загружаем из API
    print(f"🌐 Загружаем данные для {ticker} из MOEX API")
    try:
        till = datetime.today().strftime('%Y-%m-%d')
        from_date = (datetime.today() - pd.Timedelta(days=days * 1.5)).strftime('%Y-%m-%d')
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?interval=24&from={from_date}&till={till}"
        
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        candles = data['candles']['data']
        columns = data['candles']['columns']
        
        df = pd.DataFrame(candles, columns=columns)
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
        result = df.tail(days)
        
        # Сохраняем в кэш
        moex_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        
        cleanup_cache()
        return result
        
    except Exception as e:
        print(f"Ошибка получения данных для {ticker}: {e}")
        return pd.DataFrame()
"""
"""
def get_moex_weekly_data_with_cache(ticker="SBER", weeks=80):
    #Кэшированная версия get_moex_weekly_data
    cache_key = f"weekly_{ticker}_{weeks}"
    
    if cache_key in weekly_cache:
        cache_entry = weekly_cache[cache_key]
        if is_cache_valid(cache_entry, WEEKLY_CACHE_TTL):
            print(f"📋 Недельные данные для {ticker} взяты из кэша")
            return cache_entry['data']
    
    print(f"🌐 Загружаем недельные данные для {ticker} из MOEX API")
    try:
        till = datetime.today().strftime('%Y-%m-%d')
        from_date = (datetime.today() - pd.Timedelta(weeks=weeks * 1.5)).strftime('%Y-%m-%d')
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?interval=7&from={from_date}&till={till}"
        
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        candles = data['candles']['data']
        columns = data['candles']['columns']
        
        df = pd.DataFrame(candles, columns=columns)
        df['begin'] = pd.to_datetime(df['begin'])
        df = df.sort_values('begin')
        df.set_index('begin', inplace=True)
        df = df.rename(columns={'close': 'close'})
        df = df[['close']].dropna()
        result = df.tail(weeks)
        
        weekly_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        
        cleanup_cache()
        return result
        
    except Exception as e:
        print(f"Ошибка получения данных для {ticker}: {e}")
        return pd.DataFrame()
"""
def get_figi_by_ticker_with_cache(ticker: str) -> str | None:
    # Кэшированная версия get_figi_by_ticker
    if ticker in figi_cache:
        print(f"📋 FIGI для {ticker} взят из кэша")
        return figi_cache[ticker]
    
    print(f"🌐 Загружаем FIGI для {ticker} из Tinkoff API")
    try:
        from tinkoff.invest import Client
        TINKOFF_API_TOKEN = os.getenv("TINKOFF_API_TOKEN")
        
        with Client(TINKOFF_API_TOKEN) as client:
            instruments = client.instruments.shares().instruments
            for instr in instruments:
                if instr.ticker == ticker:
                    figi_cache[ticker] = instr.figi
                    save_figi_cache()   # ← 🟢 сохраняем кэш в файл
                    return instr.figi
        
        print(f"FIGI не найден для {ticker} в TQBR")
        figi_cache[ticker] = None
        save_figi_cache()   # ← 🟢 сохраняем, чтобы больше не дёргать API
        return None
    except Exception as e:
        print(f"Ошибка поиска FIGI для {ticker}: {e}")
        return None

def get_cache_stats():
    #Возвращает статистику кэша
    import sys
    
    print(f"🔍 Отладка кэша:")
   # print(f"   moex_cache: {len(moex_cache)} записей")
   # print(f"   weekly_cache: {len(weekly_cache)} записей") 
    print(f"   figi_cache: {len(figi_cache)} записей")
    
   # moex_size = sys.getsizeof(moex_cache) / 1024 / 1024
   # weekly_size = sys.getsizeof(weekly_cache) / 1024 / 1024
    figi_size = sys.getsizeof(figi_cache) / 1024 / 1024
    
    return {
      #  'moex_entries': len(moex_cache),
      #  'weekly_entries': len(weekly_cache),
        'figi_entries': len(figi_cache),
        'total_size_mb': round(figi_size, 2),
        'entries': len(figi_cache),
        'size_mb': round(figi_size, 2)
    }


def enable_caching():
    #Включает кэширование, заменяя оригинальные функции
    try:
        import sys
        time.sleep(0.5)  # Задержка для полной загрузки main
                # Проверяем, что модуль main уже загружен
        if '__main__' in sys.modules:
            main_module = sys.modules['__main__']
            
            # Проверяем наличие функций и заменяем их
            """
            if hasattr(main_module, 'get_moex_data'):
                print(f"🔄 Заменяем get_moex_data на кэшированную версию")
                main_module._original_get_moex_data = main_module.get_moex_data
                main_module.get_moex_data = get_moex_data_with_cache
                
            if hasattr(main_module, 'get_moex_weekly_data'):
                print(f"🔄 Заменяем get_moex_weekly_data на кэшированную версию")
                main_module._original_get_moex_weekly_data = main_module.get_moex_weekly_data
                main_module.get_moex_weekly_data = get_moex_weekly_data_with_cache
            """    
            if hasattr(main_module, 'get_figi_by_ticker'):
                print(f"🔄 Заменяем get_figi_by_ticker на кэшированную версию")
                main_module._original_get_figi_by_ticker = main_module.get_figi_by_ticker
                main_module.get_figi_by_ticker = get_figi_by_ticker_with_cache
            
            print("✅ Кэширование включено успешно")
            return True
        else:
            print("⚠️ Модуль __main__ еще не загружен")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка включения кэширования: {e}")
        return False

def print_cache_stats():
    """Выводит информацию о размере figi_cache"""
    figi_size = sys.getsizeof(figi_cache) / 1024 / 1024
    print(f"ℹ️ figi_cache: {len(figi_cache)} записей, {figi_size:.2f} MB")


# Загружаем figi_cache при старте
load_figi_cache()

# Убираем автоматическое включение при импорте
# if ENABLE_CACHING:
#     enable_caching()

# Добавляем функцию для отложенной активации

