# main.py 

import os

os.environ.setdefault("SSL_TBANK_VERIFY", "True")

import matplotlib
matplotlib.use('Agg')  # Включаем "безголовый" режим для matplotlib
import requests
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from scipy.signal import argrelextrema
import asyncio
import html
import concurrent.futures
import time as _time


# Активация Токена Tinkoff
from t_tech.invest import Client, CandleInterval

TINKOFF_API_TOKEN = os.getenv("TINKOFF_API_TOKEN")
client = Client(TINKOFF_API_TOKEN)

def set_webhook():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("Ошибка: TELEGRAM_TOKEN не найден в переменных окружения")
        return

    webhook_url = f"https://moex-telegram-bot-sra8.onrender.com/"

    response = requests.get(
        f"https://api.telegram.org/bot{token}/setWebhook",
        params={"url": webhook_url}
    )

    if response.status_code == 200:
        print("Webhook установлен успешно!")
    else:
        print(f"Ошибка при установке webhook: {response.text}")


#if __name__ == "__main__":
#    set_webhook()

# Telegram импорты
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove
    from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, ConversationHandler, MessageHandler, filters
except ModuleNotFoundError:
    print("Библиотека 'python-telegram-bot' не установлена.")
    Update = None
    ApplicationBuilder = None
    CommandHandler = None
    CallbackQueryHandler = None
    ContextTypes = None

# Секторы акций
SECTORS = {
    "Финансы": ["SBER", "T", "VTBR", "MOEX", "SPBE", "RENI", "BSPB", "SVCB", "MBNK", "LEAS", "SFIN", "AFKS", "CARM", "ZAYM", "MGKL", "DOMRF"],
    "Нефтегаз": ["GAZP", "NVTK", "LKOH", "ROSN", "TATNP", "TATN", "SNGS", "SNGSP", "BANE", "BANEP", "RNFT", "SIBN"],
    "Металлы и добыча": ["ALRS", "GMKN", "RUAL", "TRMK", "MAGN", "NLMK", "CHMF", "MTLRP", "MTLR", "RASP", "PLZL", "UGLD", "SGZH" ,"ENPG", "SELG"],
    "IT": ["YDEX", "DATA", "HEAD", "POSI", "VKCO", "ASTR", "IVAT", "DELI", "WUSH", "CNRU", "DIAS", "SOFL", "ELMT", "BAZA"],
    "Телеком": ["MTSS", "RTKMP", "RTKM", "MGTSP"],
    "Строители": ["SMLT", "PIKK", "LSRG", "ETLN", "GLRX"],
    "Ритейл": ["X5", "MGNT", "LENT", "BELU", "OZON", "EUTR", "ABRD", "GCHE", "AQUA", "HNFG", "MVID", "VSEH", "FIXR"],
    "Электро": ["IRAO", "UPRO", "LSNGP", "MSRS", "MRKU", "MRKC", "MRKP", "MRKS", "MRKV", "MRKY", "FEES", "HYDR", "ELFV", "MSNG", "TGKA", "TGKN"],
    "Транспорт и логистика": ["TRNFP", "AFLT", "FESH", "NMTP", "FLOT"],
    "Агро": ["PHOR", "RAGR", "KZOS", "AKRN", "NKHP"],
    "Медицина": ["MDMG", "OZPH", "PRMD", "ABIO", "GEMC"],
    "Машиностроение": ["UWGN", "SVAV", "KMAZ", "UNAC", "IRKT", "VSMO"]
}


TICKERS_PER_PAGE = 10

# Состояния для диалогов
ASK_DAYS = 1
ASK_TICKER = 2
ASK_DELTA_DAYS = 3

FIGI_CACHE_FILE = "figi_cache.json"

def load_figi_cache_from_file():
    if os.path.exists(FIGI_CACHE_FILE):
        with open(FIGI_CACHE_FILE, "r", encoding="utf-8") as f:
            figi_cache = json.load(f)
        print(f"✅ figi_cache загружен из файла: {len(figi_cache)} записей")
        return figi_cache
    else:
        print("⚠️ Файл figi_cache.json не найден, возвращаем пустой словарь")
        return {}

# Загружаем figi_cache из файла
figi_cache = load_figi_cache_from_file()


# === ЗАЩИТА ОТ ДВОЙНОГО ЗАПУСКА ===
# Telegram повторяет webhook если не получает 200 OK за 60 сек.
# Декоратор игнорирует повторные update_id для одной команды.

def no_duplicate(key: str):
    #Декоратор: пропускает повторные вызовы с одинаковым update_id
    def decorator(func):
        async def wrapper(update, context, *args, **kwargs):
            uid = update.update_id
            running = context.bot_data.setdefault(f"_dedup_{key}", set())
            if uid in running:
                print(f"⚠️ [{key}] дубликат update_id={uid}, пропускаем")
                return
            running.add(uid)
            try:
                return await func(update, context, *args, **kwargs)
            finally:
                running.discard(uid)
        return wrapper
    return decorator


# === RATE LIMITER ДЛЯ T-INVEST API ===
# Лимиты: 120 unary запросов/мин — используем 100 с запасом

class RateLimiter:
    def __init__(self, rate_per_minute: int = 100):
        self._interval = 60.0 / rate_per_minute
        self._last_call: float = 0.0
        # Asyncio-примитивы создаём лениво (при первом вызове),
        # чтобы не привязываться к loop'у до его запуска
        self._semaphore: asyncio.Semaphore | None = None
        self._lock: asyncio.Lock | None = None

    def _ensure_primitives(self):
        """Создаём Semaphore и Lock в контексте работающего event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(8)
        if self._lock is None:
            self._lock = asyncio.Lock()

    async def acquire(self):
        self._ensure_primitives()
        async with self._lock:
            now = _time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = _time.monotonic()

    async def __aenter__(self):
        self._ensure_primitives()
        await self._semaphore.acquire()
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        self._semaphore.release()


_rate_limiter = RateLimiter(rate_per_minute=100)


async def async_get_moex_data(ticker: str, days: int = 100) -> pd.DataFrame:
    #Async-обёртка над get_moex_data с соблюдением rate limit
    async with _rate_limiter:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: get_moex_data(ticker, days))


async def async_get_moex_weekly_data(ticker: str, weeks: int = 80) -> pd.DataFrame:
    #Async-обёртка над get_moex_weekly_data с соблюдением rate limit
    async with _rate_limiter:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: get_moex_weekly_data(ticker, weeks))

"""
def get_moex_data_old(ticker="SBER", days=120):
    #Получение дневных данных с MOEX
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
        return df.tail(days)
    except Exception as e:
        print(f"Ошибка получения данных для {ticker}: {e}")
        return pd.DataFrame() 
"""     
    
def get_moex_data(ticker: str = "SBER", days: int = 120) -> pd.DataFrame:
    #Загружает 1D свечи по тикеру из Tinkoff Invest API (вместо ISS MOEX)
    try:
        figi = figi_cache.get(ticker)
        if figi is None:
            print(f"❌ FIGI для тикера {ticker} не найдено")
            return pd.DataFrame()

        print(f"📡 Используем FIGI {figi} для загрузки данных {ticker}")

        to_dt = datetime.now(ZoneInfo("Europe/Moscow"))
        # запас по времени, чтобы наверняка набрать days дневных свечей (выходные/праздники)
        from_dt = to_dt - timedelta(days=int(days * 2.2))

        with Client(TINKOFF_API_TOKEN) as client:
            candles_response = client.market_data.get_candles(
                figi=figi,
                from_=from_dt,
                to=to_dt,
                interval=CandleInterval.CANDLE_INTERVAL_DAY,
            )

        import time
        time.sleep(0.1)  # 100мс задержка после каждого запроса к API

        if not candles_response.candles:
            print(f"❌ Нет данных свечей для {ticker}")
            return pd.DataFrame()

        data = []
        for c in candles_response.candles:
            try:
                open_p = c.open.units + c.open.nano / 1e9
                high_p = c.high.units + c.high.nano / 1e9
                low_p = c.low.units + c.low.nano / 1e9
                close_p = c.close.units + c.close.nano / 1e9
                volume = c.volume
                timestamp = pd.to_datetime(c.time)

                data.append({
                    "time": timestamp,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": volume
                })
            except Exception as candle_e:
                print(f"❌ Ошибка обработки свечи для {ticker}: {candle_e}")
                continue

        if not data:
            print(f"❌ Нет валидных данных для {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()

        # Обработка временных зон
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Europe/Moscow')

        # ВАЖНО: дальше по коду у тебя часто ожидаются колонки close/volume/high/low
        df = df[["close", "volume", "high", "low"]].dropna()

        print(f"✅ Загружено {len(df)} свечей для {ticker}")
        return df.tail(days)

    except Exception as e:
        print(f"❌ Ошибка получения данных для {ticker}: {e}")
        return pd.DataFrame()



"""
def get_moex_weekly_data_old(ticker="SBER", weeks=80):
    #Получение недельных данных с MOEX
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
        return df.tail(weeks)
    except Exception as e:
        print(f"Ошибка получения данных для {ticker}: {e}")
        return pd.DataFrame()
""" """
def get_figi_by_ticker(ticker: str) -> str | None:
    #Получение FIGI по тикеру
    try:
        with Client(TINKOFF_API_TOKEN) as client:
            instruments = client.instruments.shares().instruments
            for instr in instruments:
                if instr.ticker == ticker:
                    return instr.figi
        print(f"FIGI не найден для {ticker} в TQBR")
        return None
    except Exception as e:
        print(f"Ошибка поиска FIGI для {ticker}: {e}")
        return None
"""

def get_moex_weekly_data(ticker: str = "SBER", weeks: int = 80) -> pd.DataFrame:
    try:
        figi = figi_cache.get(ticker)
        if figi is None:
            print(f"❌ FIGI для тикера {ticker} не найден")
            return pd.DataFrame()

        to_dt = datetime.now(ZoneInfo("Europe/Moscow"))
        from_dt = to_dt - timedelta(weeks=weeks + 2)

        candles = []
        with Client(TINKOFF_API_TOKEN) as client:
            for candle in client.get_all_candles(
                figi=figi,
                from_=from_dt,
                to=to_dt,
                interval=CandleInterval.CANDLE_INTERVAL_WEEK,
            ):
                candles.append({
                    "time": candle.time,
                    "open": candle.open.units + candle.open.nano / 1e9,
                    "high": candle.high.units + candle.high.nano / 1e9,
                    "low": candle.low.units + candle.low.nano / 1e9,
                    "close": candle.close.units + candle.close.nano / 1e9,
                    "volume": candle.volume
                })

        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles).set_index("time").sort_index()
        return df.tail(weeks)

    except Exception as e:
        print(f"❌ Ошибка weekly данных для {ticker}: {e}")
        return pd.DataFrame()



def get_moex_data_4h_tinkoff(ticker: str = "SBER", days: int = 25) -> pd.DataFrame:
    #Загружает 4H свечи по тикеру из Tinkoff Invest API
    try:
        figi = figi_cache.get(ticker)
        if figi is None:
            print(f"❌ FIGI для тикера {ticker} не найдено")
            return pd.DataFrame()
            
        print(f"📡 Используем FIGI {figi} для загрузки данных {ticker}")
        
        to_dt = datetime.now(ZoneInfo("Europe/Moscow"))
        from_dt = to_dt - timedelta(days=days)
        
        with Client(TINKOFF_API_TOKEN) as client:
            candles_response = client.market_data.get_candles(
                figi=figi,
                from_=from_dt,
                to=to_dt,
                interval=CandleInterval.CANDLE_INTERVAL_4_HOUR,
            )
            
        import time
        time.sleep(0.1)  # 100мс задержка после каждого запроса к API
            
        if not candles_response.candles:
            print(f"❌ Нет данных свечей для {ticker}")
            return pd.DataFrame()
        
        data = []
        for c in candles_response.candles:
            try:
                open_p = c.open.units + c.open.nano / 1e9
                high_p = c.high.units + c.high.nano / 1e9
                low_p = c.low.units + c.low.nano / 1e9
                close_p = c.close.units + c.close.nano / 1e9
                volume = c.volume
                timestamp = pd.to_datetime(c.time)
                
                data.append({
                    "time": timestamp,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": volume
                })
            except Exception as candle_e:
                print(f"❌ Ошибка обработки свечи для {ticker}: {candle_e}")
                continue
                
        if not data:
            print(f"❌ Нет валидных данных для {ticker}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        
        # Обработка временных зон
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Europe/Moscow')
        
        print(f"✅ Загружено {len(df)} свечей для {ticker}")
        return df
        
    except Exception as e:
        print(f"❌ Ошибка получения данных для {ticker}: {e}")
        return pd.DataFrame()


def fetch_4h_data_for_ticker(ticker, days=25):
    try:
        df = get_moex_data_4h_tinkoff(ticker, days=days)
        if df is not None and not df.empty:
            return ticker, df  # Можно вернуть df или len(df)
        else:
            return ticker, None
    except Exception as e:
        print(f"{ticker} error: {e}")
        return ticker, None

def parallel_get_4h_data(tickers, days=25, max_workers=10):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_4h_data_for_ticker, ticker, days): ticker for ticker in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker, df = future.result()
            results[ticker] = df
    return results  # словарь: {ticker: DataFrame}
    


# === ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ===

def compute_rsi(series, window=14):
    #Вычисляет RSI используя pandas ewm для сглаживания Wilder's
    if len(series) < window + 1:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)
    
    alpha = 1.0 / window
    avg_gain = gain_series.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss_series.ewm(alpha=alpha, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    return rsi.round(0)



def analyze_indicators(df):
    #Анализ технических индикаторов
    if df.empty:
        return df
    
    df['RSI'] = compute_rsi(df['close'], window=14)
    df['Volume_Mean'] = df['volume'].rolling(window=10).mean()
    df['Anomaly'] = df['volume'] > 1.5 * df['Volume_Mean']
    df['Volume_Multiplier'] = df['volume'] / df['Volume_Mean']
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    return df






# === ФУНКЦИИ ПОИСКА ПЕРЕСЕЧЕНИЙ ===

def find_sma30_crossover(ticker, days=7):
    #Находит пересечение цены снизу вверх через SMA30
    try:
        df = get_moex_data(ticker, days=60)
        if df.empty or len(df) < 35:
            return None

        # Фильтр по обороту
        filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
        filter_avg_turnover = filter_turnover_series.mean()
        
        if filter_avg_turnover < 50_000_000:
            return None

        df['SMA30'] = df['close'].rolling(window=30).mean()
        
        current_close = df['close'].iloc[-1]
        current_sma30 = df['SMA30'].iloc[-1]
        
        if current_close <= current_sma30:
            return None

        recent_df = df.tail(days + 1)
        crossover_date = None
        
        for i in range(1, len(recent_df)):
            prev_close = recent_df['close'].iloc[i-1]
            curr_close = recent_df['close'].iloc[i]
            prev_sma = recent_df['SMA30'].iloc[i-1]
            curr_sma = recent_df['SMA30'].iloc[i]
            
            if (prev_close < prev_sma and curr_close > curr_sma):
                crossover_date = recent_df.index[i]
                break
        
        return crossover_date
        
    except Exception as e:
        print(f"Ошибка при поиске пересечения SMA30 для {ticker}: {e}")
        return None

def find_sma30_crossover_short(ticker, days=7):
    #Находит пересечение цены сверху вниз через SMA30
    try:
        df = get_moex_data(ticker, days=60)
        if df.empty or len(df) < 35:
            return None

        # Фильтр по обороту
        filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
        filter_avg_turnover = filter_turnover_series.mean()
        
        if filter_avg_turnover < 50_000_000:
            return None

        df['SMA30'] = df['close'].rolling(window=30).mean()
        
        current_close = df['close'].iloc[-1]
        current_sma30 = df['SMA30'].iloc[-1]
        
        if current_close >= current_sma30:
            return None

        recent_df = df.tail(days + 1)
        crossover_date = None
        
        for i in range(1, len(recent_df)):
            prev_close = recent_df['close'].iloc[i-1]
            curr_close = recent_df['close'].iloc[i]
            prev_sma = recent_df['SMA30'].iloc[i-1]
            curr_sma = recent_df['SMA30'].iloc[i]
            
            if (prev_close > prev_sma and curr_close < curr_sma):
                crossover_date = recent_df.index[i]
                break
        
        return crossover_date
        
    except Exception as e:
        print(f"Ошибка при поиске пересечения SMA30 для {ticker}: {e}")
        return None

def find_sma30_crossover_week(ticker, weeks=5):
    #Находит пересечение цены снизу вверх через SMA30 на недельном ТФ
    try:
        df = get_moex_weekly_data(ticker, weeks=60)
        if df.empty or len(df) < 35:
            return None

        dfd = get_moex_data(ticker, days=20)
        if dfd.empty or len(dfd) < 15:
            return None

        # Фильтр по обороту
        filter_turnover_series = dfd['volume'].iloc[-10:] * dfd['close'].iloc[-10:]
        filter_avg_turnover = filter_turnover_series.mean()
        
        if filter_avg_turnover < 50_000_000:
            return None

        df['SMA30'] = df['close'].rolling(window=30).mean()
        
        current_close = df['close'].iloc[-1]
        current_sma30 = df['SMA30'].iloc[-1]
        
        if current_close <= current_sma30:
            return None

        recent_df = df.tail(weeks + 1)
        crossover_date = None
        
        for i in range(1, len(recent_df)):
            prev_close = recent_df['close'].iloc[i-1]
            curr_close = recent_df['close'].iloc[i]
            prev_sma = recent_df['SMA30'].iloc[i-1]
            curr_sma = recent_df['SMA30'].iloc[i]
            
            if (prev_close < prev_sma and curr_close > curr_sma):
                crossover_date = recent_df.index[i]
                break
        
        return crossover_date
        
    except Exception as e:
        print(f"Ошибка при поиске пересечения SMA30 для {ticker}: {e}")
        return None

# === TELEGRAM КОМАНДЫ ===



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    text = (
        "Привет! Это бот от команды @TradeAnsh для анализа акций Мосбиржи.\n"
        "Команды:\n"
        "/cross_ema20x50 — акции с пересечением EMA 20x50 на 1D\n"
        "/cross_ema20x50_4h — акции с пересечением EMA 20x50 на 4H\n"
        "/cross_ema9x50 — акции с пересечением EMA 20x50 на 1D\n"
        "/cross_ema200 — акции с пересечением цены и EMA200 на 1D\n"
        "/stan_recent — акции с лонг пересечением SMA30 на 1D\n"
        "/stan_recent_d_short — акции с шорт пересечением SMA30 на 1D\n"
        "/stan_recent_week — акции с лонг пересечением SMA30 на 1W\n"
        "/moneyflow - Топ по росту и оттоку денежного потока (Money A/D)\n"
        "/high_volume - Акции с повышенным объемом\n"
        "/delta — расчет дельты денежного потока для конкретной акции\n"
        "/rsi_top — Топ перекупленных и перепроданных акций по RSI и Стохастику\n"
    )
    await update.message.reply_text(text)

# Диалоговые функции
async def ask_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📅 Введите количество дней для расчета дельты денежного потока (например, 10):")
    return ASK_DAYS

async def receive_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        days = int(update.message.text)
        if not (1 <= days <= 100):
            await update.message.reply_text("⚠️ Введите число от 1 до 100.")
            return ASK_DAYS

        context.user_data['days'] = days
        await long_moneyflow(update, context)
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text("⚠️ Введите целое число, например: 10")
        return ASK_DAYS

async def ask_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Введите тикер (или список тикеров) акции (например, SBER):")
    return ASK_TICKER

async def receive_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ticker_input = update.message.text.strip().upper()
    
    if not ticker_input:
        await update.message.reply_text("⚠️ Введите один или несколько тикеров через запятую.")
        return ASK_TICKER
    
    context.user_data['delta_ticker'] = ticker_input
    await update.message.reply_text("📅 Укажите, за сколько дней рассчитать дельту (1–100):")
    return ASK_DELTA_DAYS


# Основные команды анализа

@no_duplicate("high_vol")
async def high_volume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу акции с повышенным объёмом… (параллельный режим)")
    all_tickers = sum(SECTORS.values(), [])

    # Волна 1: все дневные параллельно
    async def _fetch_daily(ticker: str):
        try:
            return ticker, await async_get_moex_data(ticker, days=100)
        except Exception as e:
            print(f"[high_volume] daily error {ticker}: {e}")
            return ticker, None

    daily_results = await asyncio.gather(*[_fetch_daily(t) for t in all_tickers])
    daily_map = {t: df for t, df in daily_results}

    # Фильтр: ratio >= 1.2
    need_weekly = []
    for ticker, df in daily_map.items():
        if df is None or df.empty or len(df) < 60:
            continue
        avg_turnover   = (df['volume'].iloc[-11:-1] * df['close'].iloc[-11:-1]).mean()
        today_turnover = df['volume'].iloc[-1] * df['close'].iloc[-1]
        if avg_turnover > 0 and today_turnover / avg_turnover >= 1.2:
            need_weekly.append(ticker)

    # Волна 2: недельные только для кандидатов
    async def _fetch_weekly(ticker: str):
        try:
            return ticker, await async_get_moex_weekly_data(ticker, weeks=80)
        except Exception as e:
            print(f"[high_volume] weekly error {ticker}: {e}")
            return ticker, None

    weekly_results = await asyncio.gather(*[_fetch_weekly(t) for t in need_weekly])
    weekly_map = {t: wdf for t, wdf in weekly_results}

    rows = []
    for ticker in need_weekly:
        try:
            df  = daily_map[ticker]
            wdf = weekly_map.get(ticker)

            avg_turnover   = (df['volume'].iloc[-11:-1] * df['close'].iloc[-11:-1]).mean()
            today_turnover = df['volume'].iloc[-1] * df['close'].iloc[-1]
            ratio = today_turnover / avg_turnover if avg_turnover > 0 else 0

            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            current_ema20  = df['EMA20'].iloc[-1]
            current_ema50  = df['EMA50'].iloc[-1]
            current_price  = df['close'].iloc[-1]

            ema20x50_long  = (current_ema20 > current_ema50) and (current_price > current_ema20)
            ema20x50_short = (current_ema20 < current_ema50) and (current_price < current_ema20)
            price_change   = (current_price / df['close'].iloc[-2] - 1) if len(df) > 1 else 0

            price_above_sma30 = False
            if wdf is not None and not wdf.empty and len(wdf) >= 30:
                wdf = wdf.copy()
                wdf['SMA30'] = wdf['close'].rolling(window=30).mean()
                weekly_sma30 = wdf['SMA30'].iloc[-1]
                price_above_sma30 = (wdf['close'].iloc[-1] > weekly_sma30) if pd.notna(weekly_sma30) else False

            money_df = calculate_money_ad(df)
            ad_delta = money_df['money_ad'].iloc[-1] - money_df['money_ad'].iloc[-11]
            money_flow_icon = "🟢" if ad_delta > 0 else "🔴"
            money_flow_str  = f"{ad_delta/1_000_000:+.0f}M"

            rows.append((ticker, current_price, price_change, ratio,
                         ema20x50_long, ema20x50_short, price_above_sma30,
                         money_flow_icon, money_flow_str))
        except Exception as e:
            print(f"[high_volume] calc error {ticker}: {e}")
            continue

    rows.sort(key=lambda x: x[3], reverse=True)
    rows = rows[:15]

    if not rows:
        await update.message.reply_text("📊 Акций с повышенным объёмом не найдено")
        return

    msg  = "📊 <b>Акции с повышенным объёмом</b>\n\n"
    msg += "<pre>"
    msg += f"{'Тикер':<6} {'Цена':>8} {'Δ Цены':>7} {'Объём':>6} {'ema20x50':>6} {'sma30':>6} {'Δ Потока':>10}\n"
    msg += "-" * 60 + "\n"
    for ticker, price, delta, ratio, ema20x50_long, ema20x50_short, sma_signal, mf_icon, mf_str in rows:
        ema_icon = "🟢" if ema20x50_long else ("🔴" if ema20x50_short else "⚫")
        sma_icon = "🟢" if sma_signal else "🔴"
        msg += f"{ticker:<6} {price:>8.2f} {delta*100:>6.1f}% {ratio:>5.1f}x {ema_icon:>6} {sma_icon:>4} {mf_icon}{mf_str:>6}\n"
    msg += "</pre>\n\n"
    msg += "<i>EMA - пересечение EMA20x50 (D) на дневном ТФ</i>\n"
    msg += "<i>SMA - цена выше SMA30 на недельном ТФ</i>\n"
    msg += "<i>Δ Потока - приток/отток денежных средств (посл. 10 дней)</i>"
    await update.message.reply_text(msg, parse_mode="HTML")

@no_duplicate("ema200")
async def cross_ema200(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу пересечения цены и EMA200 за последние 50 дней…")
    all_tickers = sum(SECTORS.values(), [])

    async def _fetch(ticker: str):
        try:
            return ticker, await async_get_moex_data(ticker, days=350)
        except Exception as e:
            print(f"Ошибка EMA200 для {ticker}: {e}")
            return ticker, None

    results = await asyncio.gather(*[_fetch(t) for t in all_tickers])

    long_hits, short_hits = [], []
    for ticker, df in results:
        try:
            if df is None or df.empty or len(df) < 200:
                continue

            df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
            recent = df.tail(51)
            ema200 = recent['EMA200']
            close  = recent['close']

            prev_close  = close.shift(1)
            prev_ema200 = ema200.shift(1)

            current_close  = df['close'].iloc[-1]
            current_ema200 = df['EMA200'].iloc[-1]

            cross_up   = (prev_close <= prev_ema200) & (close > ema200)
            cross_down = (prev_close >= prev_ema200) & (close < ema200)

            confirmed_up   = cross_up   & (close > ema200) & (current_close > current_ema200)
            confirmed_down = cross_down & (close < ema200) & (current_close < current_ema200)

            if confirmed_up.any():
                long_hits.append((ticker, confirmed_up[confirmed_up].index[-1].strftime('%d.%m.%Y')))
            elif confirmed_down.any():
                short_hits.append((ticker, confirmed_down[confirmed_down].index[-1].strftime('%d.%m.%Y')))

        except Exception as e:
            print(f"Ошибка EMA200 calc для {ticker}: {e}")
            continue

    long_hits.sort(key=lambda x:  datetime.strptime(x[1], '%d.%m.%Y'), reverse=True)
    short_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y'), reverse=True)

    msg = ""
    if long_hits:
        msg += f"🟢 *Лонг пересечение цены с EMA200 за последние 50 дней, всего: {len(long_hits)}:*\n"
        msg += "\n".join(f"{t} {d}" for t, d in long_hits) + "\n\n"
    else:
        msg += "🟢 *Лонг сигналов не найдено за последние 50 дней*\n\n"

    if short_hits:
        msg += f"🔴 *Шорт пересечение цены с EMA200 за последние 50 дней, всего: {len(short_hits)}:*\n"
        msg += "\n".join(f"{t} {d}" for t, d in short_hits) + "\n\n"
    else:
        msg += "🔴 *Шорт сигналов не найдено за последние 50 дней*\n\n"

    if long_hits or short_hits:
        parts = []
        if long_hits:
            parts.append(f"*Лонг:* " + ", ".join(t for t, _ in long_hits))
        if short_hits:
            parts.append(f"\n*Шорт:* " + ", ".join(t for t, _ in short_hits))
        msg += "\n" + "\n".join(parts)

    await update.message.reply_text(msg, parse_mode="Markdown")

@no_duplicate("ema20x50")
async def cross_ema20x50(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу пересечения EMA20 и EMA50 за последние 50 дней…")
    all_tickers = sum(SECTORS.values(), [])

    async def _fetch(ticker: str):
        try:
            return ticker, await async_get_moex_data(ticker, days=100)
        except Exception as e:
            print(f"Ошибка EMA для {ticker}: {e}")
            return ticker, None

    results = await asyncio.gather(*[_fetch(t) for t in all_tickers])

    long_hits, short_hits = [], []
    for ticker, df in results:
        try:
            if df is None or df.empty or len(df) < 100:
                continue

            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

            recent = df.tail(51)
            ema20 = recent['EMA20']
            ema50 = recent['EMA50']

            prev_ema20 = ema20.shift(1)
            prev_ema50 = ema50.shift(1)

            current_close  = df['close'].iloc[-1]
            current_ema20  = df['EMA20'].iloc[-1]
            current_ema50  = df['EMA50'].iloc[-1]

            cross_up   = (prev_ema20 <= prev_ema50) & (ema20 > ema50)
            cross_down = (prev_ema20 >= prev_ema50) & (ema20 < ema50)

            last_up_idx   = cross_up[cross_up].index[-1]   if cross_up.any()   else None
            last_down_idx = cross_down[cross_down].index[-1] if cross_down.any() else None

            chosen_signal, chosen_date = None, None
            if last_up_idx and last_down_idx:
                chosen_signal = 'long'  if last_up_idx > last_down_idx else 'short'
                chosen_date   = last_up_idx if last_up_idx > last_down_idx else last_down_idx
            elif last_up_idx:
                chosen_signal, chosen_date = 'long',  last_up_idx
            elif last_down_idx:
                chosen_signal, chosen_date = 'short', last_down_idx

            if chosen_signal:
                last_date_str = chosen_date.strftime('%d.%m.%Y')
                if chosen_signal == 'long':
                    mark = "🟢" if (current_close > current_ema20) and (current_ema20 > current_ema50) else "🟠"
                    long_hits.append((f"{mark} {ticker}", last_date_str))
                else:
                    mark = "🔴" if (current_close < current_ema20) and (current_ema20 < current_ema50) else "🟠"
                    short_hits.append((f"{mark} {ticker}", last_date_str))

        except Exception as e:
            print(f"Ошибка EMA calc для {ticker}: {e}")
            continue

    long_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y'),  reverse=True)
    short_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y'), reverse=True)

    msg = ""
    if long_hits:
        msg += f"🟢 *Лонг пересечение EMA20×50 за последние 50 дней, всего: {len(long_hits)}:*\n"
        msg += "\n".join(f"{t} {d}" for t, d in long_hits) + "\n\n"
    else:
        msg += "🟢 *Лонг сигналов не найдено за последние 50 дней*\n\n"

    if short_hits:
        msg += f"🔴 *Шорт пересечение EMA20×50 за последние 50 дней, всего: {len(short_hits)}:*\n"
        msg += "\n".join(f"{t} {d}" for t, d in short_hits) + "\n\n"
    else:
        msg += "🔴 *Шорт сигналов не найдено за последние 50 дней*\n\n"

    if long_hits or short_hits:
        parts = []
        if long_hits:
            parts.append(f"*Лонг:* " + ", ".join(t.split()[1] for t, _ in long_hits))
        if short_hits:
            parts.append(f"\n*Шорт:* " + ", ".join(t.split()[1] for t, _ in short_hits))
        msg += "\n" + "\n".join(parts)

    await update.message.reply_text(msg, parse_mode="Markdown")



@no_duplicate("ema9x50")
async def cross_ema9x50(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу пересечения EMA9 и EMA50 за последние 50 дней…")
    all_tickers = sum(SECTORS.values(), [])

    async def _fetch(ticker: str):
        try:
            return ticker, await async_get_moex_data(ticker, days=100)
        except Exception as e:
            print(f"Ошибка EMA9x50 для {ticker}: {e}")
            return ticker, None

    results = await asyncio.gather(*[_fetch(t) for t in all_tickers])

    long_hits, short_hits = [], []
    for ticker, df in results:
        try:
            if df is None or df.empty or len(df) < 100:
                continue

            df['EMA9']  = df['close'].ewm(span=9,  adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

            recent = df.tail(51)
            ema9  = recent['EMA9']
            ema50 = recent['EMA50']

            prev_ema9  = ema9.shift(1)
            prev_ema50 = ema50.shift(1)

            current_close = df['close'].iloc[-1]
            current_ema9  = df['EMA9'].iloc[-1]
            current_ema50 = df['EMA50'].iloc[-1]

            cross_up   = (prev_ema9 <= prev_ema50) & (ema9 > ema50)
            cross_down = (prev_ema9 >= prev_ema50) & (ema9 < ema50)

            last_up_idx   = cross_up[cross_up].index[-1]   if cross_up.any()   else None
            last_down_idx = cross_down[cross_down].index[-1] if cross_down.any() else None

            chosen_signal, chosen_date = None, None
            if last_up_idx and last_down_idx:
                chosen_signal = 'long'  if last_up_idx > last_down_idx else 'short'
                chosen_date   = last_up_idx if last_up_idx > last_down_idx else last_down_idx
            elif last_up_idx:
                chosen_signal, chosen_date = 'long',  last_up_idx
            elif last_down_idx:
                chosen_signal, chosen_date = 'short', last_down_idx

            if chosen_signal:
                last_date_str = chosen_date.strftime('%d.%m.%Y')
                if chosen_signal == 'long':
                    mark = "🟢" if (current_close > current_ema9) and (current_ema9 > current_ema50) else "🟠"
                    long_hits.append((f"{mark} {ticker}", last_date_str))
                else:
                    mark = "🔴" if (current_close < current_ema9) and (current_ema9 < current_ema50) else "🟠"
                    short_hits.append((f"{mark} {ticker}", last_date_str))

        except Exception as e:
            print(f"Ошибка EMA9x50 calc для {ticker}: {e}")
            continue

    long_hits.sort(key=lambda x:  datetime.strptime(x[1], '%d.%m.%Y'), reverse=True)
    short_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y'), reverse=True)

    msg = ""
    if long_hits:
        msg += f"🟢 *Лонг пересечения EMA9×50 за последние 50 дней, всего: {len(long_hits)}:*\n"
        msg += "\n".join(f"{t} {d}" for t, d in long_hits) + "\n\n"
    else:
        msg += "🟢 *Лонг сигналов не найдено за последние 50 дней*\n\n"

    if short_hits:
        msg += f"🔴 *Шорт пересечения EMA9×50 за последние 50 дней, всего: {len(short_hits)}:*\n"
        msg += "\n".join(f"{t} {d}" for t, d in short_hits) + "\n\n"
    else:
        msg += "🔴 *Шорт сигналов не найдено за последние 50 дней*\n\n"

    if long_hits or short_hits:
        parts = []
        if long_hits:
            parts.append(f"*Лонг:* " + ", ".join(t.split()[1] for t, _ in long_hits))
        if short_hits:
            parts.append(f"\n*Шорт:* " + ", ".join(t.split()[1] for t, _ in short_hits))
        msg += "\n" + "\n".join(parts)

    await update.message.reply_text(msg, parse_mode="Markdown")



@no_duplicate("cross_ema20x50_4h")
async def cross_ema20x50_4h(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🔍 Ищу пересечения EMA20 и EMA50 по 4H таймфрейму за последние 25 свечей...")
        print("▶ Запущена команда EMA CROSS")
        
        start_time = datetime.now()
        MAX_EXECUTION_TIME = 1800
        
        all_tickers = sum(SECTORS.values(), [])
        print(f"🔁 Всего тикеров для обработки: {len(all_tickers)}")
        
        long_hits, short_hits = [], []
        processed_count = 0
        
        for ticker in all_tickers:
            if (datetime.now() - start_time).seconds > MAX_EXECUTION_TIME:
                print(f"⏰ Превышено максимальное время выполнения ({MAX_EXECUTION_TIME} сек)")
                break
                
            try:
                print(f"🔁 Обрабатываем {ticker} ({processed_count + 1}/{len(all_tickers)})")
                import sys
                sys.stdout.flush()
                
                print(f"📡 Запрашиваем данные для {ticker}...")
                
                ticker_result = await asyncio.wait_for(
                    process_single_ticker(ticker),
                    timeout=20.0
                )
                
                if ticker_result:
                    long_signal, short_signal = ticker_result
                    if long_signal:
                        long_hits.append(long_signal)
                        print(f"✅ Лонг сигнал: {long_signal[0]} на {long_signal[1]}")
                    if short_signal:
                        short_hits.append(short_signal)
                        print(f"✅ Шорт сигнал: {short_signal[0]} на {short_signal[1]}")
                
                print(f"✅ Завершен анализ для {ticker}")
                processed_count += 1
                
                if processed_count % 20 == 0:
                    try:
                        progress_msg = f"⏳ Обработано {processed_count}/{len(all_tickers)} тикеров..."
                        await update.message.reply_text(progress_msg)
                        print(f"📱 Отправлено уведомление: {progress_msg}")
                    except Exception as progress_e:
                        print(f"❌ Ошибка отправки прогресса: {progress_e}")
                
                await asyncio.sleep(0.3)
                sys.stdout.flush()
                
            except asyncio.TimeoutError:
                print(f"⏰ Таймаут для {ticker}")
                sys.stdout.flush()
                continue
            except Exception as e:
                print(f"❌ Ошибка EMA для {ticker}: {e}")
                sys.stdout.flush()
                continue
        
        print(f"✅ Обработано тикеров: {processed_count}/{len(all_tickers)}")
        
        try:
            long_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y %H:%M'), reverse=True)
            short_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y %H:%M'), reverse=True)
        except Exception as e:
            print(f"❌ Ошибка сортировки: {e}")
        
        long_hits  = long_hits[:30]
        short_hits = short_hits[:30]
        
        execution_time = (datetime.now() - start_time).seconds
        msg = f"📊 *Анализ завершен* (обработано {processed_count} тикеров за {execution_time} сек)\n\n"
        
        if long_hits:
            msg += f"🟢 *Лонг пересечение EMA20×50 за последние 25 4Ч свечей, всего: {len(long_hits)}:*\n"
            msg += "\n".join(f"{t} {d}" for t, d in long_hits) + "\n\n"
        else:
            msg += "🟢 *Лонг сигналов не найдено за последние 25 4Ч свечей*\n\n"
            
        if short_hits:
            msg += f"🔴 *Шорт пересечение EMA20×50 за последние 25 4Ч свечей, всего: {len(short_hits)}:*\n\n"
            msg += "\n".join(f"{t} {d}" for t, d in short_hits) + "\n\n"
        else:
            msg += "🔴 *Шорт сигналов не найдено за последние 25 4Ч свечей*\n\n"

        if long_hits or short_hits:
            tickers_summary = []
            if long_hits:
                long_tickers = ", ".join(t.split()[-1] for t, _ in long_hits)
                tickers_summary.append(f"*Лонг:* {long_tickers}")
            if short_hits:
                short_tickers = ", ".join(t.split()[-1] for t, _ in short_hits)
                tickers_summary.append(f"\n*Шорт:* {short_tickers}")
            msg += "\n" + "\n".join(tickers_summary)

        await update.message.reply_text(msg, parse_mode="Markdown")
        print("✅ Команда EMA CROSS завершена успешно")
        
    except Exception as main_e:
        print(f"❌ Критическая ошибка в команде EMA CROSS: {main_e}")
        try:
            await update.message.reply_text(
                "❌ Произошла ошибка при анализе пересечений EMA. Попробуйте позже.",
                parse_mode="Markdown"
            )
        except:
            print("❌ Не удалось отправить сообщение об ошибке")



async def process_single_ticker(ticker: str):
    
   # Обрабатывает один тикер и возвращает найденные сигналы с цветными статусами для вывода
    
    try:
        # Получаем данные (async с переносом в поток для синхронной функции)
        df = await asyncio.to_thread(get_moex_data_4h_tinkoff, ticker, 50)
        if df.empty or len(df) < 50:
            print(f"❌ Недостаточно данных для {ticker}: {len(df) if not df.empty else 0}")
            return None

        # Рассчитываем EMA20 и EMA50
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

        # Для поиска последних пересечений берем последние 26 свечей
        recent = df.tail(26)
        ema20 = recent['EMA20']
        ema50 = recent['EMA50']
        close = recent['close']

        prev_ema20 = ema20.shift(1)
        prev_ema50 = ema50.shift(1)

        # Определяем пересечения EMA
        cross_up = (prev_ema20 <= prev_ema50) & (ema20 > ema50)
        cross_down = (prev_ema20 >= prev_ema50) & (ema20 < ema50)

        # Последние пересечения
        last_up_idx = cross_up[cross_up].index[-1] if cross_up.any() else None
        last_down_idx = cross_down[cross_down].index[-1] if cross_down.any() else None

        chosen_signal = None
        chosen_date = None

        if last_up_idx is not None and last_down_idx is not None:
            if last_up_idx > last_down_idx:
                chosen_signal = 'long'
                chosen_date = last_up_idx
            else:
                chosen_signal = 'short'
                chosen_date = last_down_idx
        elif last_up_idx is not None:
            chosen_signal = 'long'
            chosen_date = last_up_idx
        elif last_down_idx is not None:
            chosen_signal = 'short'
            chosen_date = last_down_idx

        if chosen_signal is None:
            return None

        # Текущие значения для окраски эмодзи
        current_close = df['close'].iloc[-1]
        current_ema20 = df['EMA20'].iloc[-1]
        current_ema50 = df['EMA50'].iloc[-1]

        if chosen_signal == 'long':
            if current_close > current_ema20 > current_ema50:
                emoji = "🟢"
            elif current_close < current_ema20:
                emoji = "🟠"
            else:
                emoji = "🟢"
        else:  # short
            if current_close < current_ema20 < current_ema50:
                emoji = "🔴"
            elif current_close > current_ema20:
                emoji = "🟠"
            else:
                emoji = "🔴"

        signal = (f"{emoji} {ticker}", chosen_date.strftime('%d.%m.%Y %H:%M'))
        return (signal if chosen_signal == 'long' else None,
                signal if chosen_signal == 'short' else None)

    except Exception as e:
        print(f"❌ Ошибка обработки тикера {ticker}: {e}")
        return None





async def receive_delta_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #Получает количество дней и выполняет расчет дельты
    try:
        days = int(update.message.text)
        if not (1 <= days <= 100):
            await update.message.reply_text("⚠️ Введите число от 1 до 100.")
            return ASK_DELTA_DAYS

        ticker_input = context.user_data['delta_ticker']  # Тут может быть строка типа: BSPB, RTKM, POSI
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

        if not tickers:
            await update.message.reply_text("⚠️ Не удалось распознать тикеры.")
            return ConversationHandler.END

        await update.message.reply_text(f"🔎 Обрабатываю {len(tickers)} тикеров за {days} дней...")
        
        for ticker in tickers:
            await calculate_single_delta(update, context, ticker, days)
            await asyncio.sleep(0.5)  # Небольшая задержка, чтобы Telegram не заспамился
        
        return ConversationHandler.END

    except ValueError:
        await update.message.reply_text("⚠️ Введите целое число, например: 10")
        return ASK_DELTA_DAYS


#/DELTA
async def calculate_single_delta(update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str, days: int):
    #Расчет дельты + график
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"🔍 Рассчитываю дельту денежного потока для {ticker} за {days} дней с графиком...")

    try:
        df = get_moex_data(ticker, days=100)
        if df.empty or len(df) < days + 1:
            await update.message.reply_text(f"❌ Недостаточно данных для {ticker}")
            return

        df = calculate_money_ad(df)

        ad_start = df['money_ad'].iloc[-(days+1)]
        ad_end = df['money_ad'].iloc[-1]
        ad_delta = ad_end - ad_start

        price_start = df['close'].iloc[-(days+1)]
        price_end = df['close'].iloc[-1]
        date_start = df.index[-(days+1)].strftime('%d.%m.%y')
        date_end = df.index[-1].strftime('%d.%m.%y')
        price_pct = 100 * (price_end - price_start) / price_start

        filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
        filter_avg_turnover = filter_turnover_series.mean()

        turnover_series = df['volume'].iloc[-days:] * df['close'].iloc[-days:]
        avg_turnover = turnover_series.mean()
        today_turnover = df['volume'].iloc[-1] * df['close'].iloc[-1]
        ratio = today_turnover / avg_turnover if avg_turnover > 0 else 0

        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        current_ema20 = df['EMA20'].iloc[-1]
        current_ema50 = df['EMA50'].iloc[-1]
        current_price = df['close'].iloc[-1]

        ema20x50_long = (current_ema20 > current_ema50) and (current_price > current_ema20)
        ema20x50_short = (current_ema20 < current_ema50) and (current_price < current_ema20)
        price_change_day = (current_price / df['close'].iloc[-2] - 1) if len(df) > 1 else 0

        try:
            wdf = get_moex_weekly_data(ticker, weeks=80)
            if len(wdf) >= 30:
                wdf['SMA30'] = wdf['close'].rolling(window=30).mean()
                weekly_sma30 = wdf['SMA30'].iloc[-1]
                weekly_price = wdf['close'].iloc[-1]
                price_above_sma30 = weekly_price > weekly_sma30 if pd.notna(weekly_sma30) else False
            else:
                price_above_sma30 = False
        except:
            price_above_sma30 = False

        delta_pct = 100 * ad_delta / avg_turnover if avg_turnover else 0

        # Формируем текст
        msg = f"📊 *Анализ дельты денежного потока для {ticker}*\n"
        msg += f"📅 *Период: {date_start} – {date_end} ({days} дней)*\n\n"

        if filter_avg_turnover < 50_000_000:
            msg += "⚠️ *Низкий среднедневной оборот (< 50 млн ₽)*\n\n"

        flow_icon = "🟢" if ad_delta > 0 else "🔴"
        ema_icon = "🟢" if ema20x50_long else ("🔴" if ema20x50_short else "⚫")
        sma_icon = "🟢" if price_above_sma30 else "🔴"

        msg += f"*Δ Цены:* {price_pct:+.1f}%\n"
        msg += f"*Δ Потока:* {ad_delta/1_000_000:+.0f} млн ₽ {flow_icon}   *Δ / Оборот:* {delta_pct:.1f}%\n"
        msg += f"*Δ Цены 1D:* {price_change_day*100:+.1f}%   *Объём:* {ratio:.1f}x\n"
        msg += f"*EMA20x50:* {ema_icon}   *SMA30:* {sma_icon}\n\n"
        msg += f"💰 *Среднедневной оборот:* {avg_turnover/1_000_000:.1f} млн ₽"

        await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

                # === ГРАФИК ===
        print(f"🔧 Начинаю создание графика для {ticker}")
        
        try:
            recent = df.tail(days + 1)
            print(f"🔧 Данные для графика: {len(recent)} точек")
        
            # Вычисляем дельту денежного потока относительно начальной точки
            money_ad_start = recent['money_ad'].iloc[0]
            money_ad_delta = recent['money_ad'] - money_ad_start
            
            # Создаем график
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Левая ось - цена
            color1 = 'blue'
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Цена (₽)', color=color1)
            line1 = ax1.plot(recent.index, recent['close'], label='Цена', color=color1, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True)
            
            # Правая ось - дельта денежного потока (начинается с 0)
            ax2 = ax1.twinx()
            
            # Определяем цвет линии в зависимости от итоговой дельты
            final_delta = money_ad_delta.iloc[-1] / 1_000_000
            color2 = 'green' if final_delta >= 0 else 'red'
            
            ax2.set_ylabel('Δ Денежного потока (млн ₽)', color=color2)
            line2 = ax2.plot(recent.index, money_ad_delta / 1_000_000, 
                             label=f'Δ Денежного потока ({final_delta:+.0f} млн ₽)', 
                             color=color2, linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Добавляем горизонтальную линию на уровне 0
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            # Настройка заголовка
            plt.title(f"{ticker} — Δ Денежного потока vs Цена ({days} дней)")
            
            # Легенда для обеих осей
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            plt.tight_layout()
            
            # Сохраняем
            filename = f"{ticker}_delta_chart.png"
            plt.savefig(filename)
            plt.close()
            
            print(f"✅ График сохранен: {filename}, итоговая дельта: {final_delta:+.0f} млн ₽")
            
            # Отправляем файл
            try:
                with open(filename, "rb") as img:
                    await context.bot.send_photo(chat_id=chat_id, photo=img)
                print("✅ График отправлен в чат")
                
                # Удаляем файл после отправки
                try:
                    os.remove(filename)
                    print("✅ Временный файл удален")
                except:
                    print("⚠️ Не удалось удалить временный файл")
                    
            except Exception as e:
                print(f"❌ Ошибка при отправке графика: {e}")
                await update.message.reply_text(f"⚠️ График создан, но не удалось отправить: {str(e)}")
        except Exception as e:
            print(f"❌ Ошибка при создании графика: {e}")
            plt.close()
            await update.message.reply_text(f"⚠️ Ошибка при создании графика: {str(e)}")

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при анализе {ticker}: {str(e)}")
        

# RSI TOP с Стохастиком
@no_duplicate("rsi_top")
async def rsi_top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    #Команда для показа топ перекупленных и топ перепроданных акций по RSI с добавлением Стохастика
    
    await update.message.reply_text("🔍 Анализирую RSI и Стохастик всех акций. Это может занять некоторое время...")
    
    overbought_stocks = []  # RSI > 70
    oversold_stocks = []    # RSI < 30
    
    # Функция расчета Стохастика %K
    def stochastic_k(df, k_period=14):
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        return stoch_k
    
    # Проходим по всем тикерам
    for ticker in sum(SECTORS.values(), []):
        try:
            df = get_moex_data(ticker, days=100)
            if df.empty or len(df) < 15:
                continue
            
            # 💰 Среднедневной оборот за последние 10 дней
            filter_avg_turnover = (df['volume'].iloc[-10:] * df['close'].iloc[-10:]).mean()
            if filter_avg_turnover < 50_000_000:
                continue
            
            # RSI и Стохастик
            rsi = compute_rsi(df['close'], window=14)
            stoch = stochastic_k(df, k_period=14)
            
            if rsi.empty or stoch.empty:
                continue
            
            current_rsi = rsi.iloc[-1]
            current_stoch = stoch.iloc[-1]
            if pd.isna(current_rsi) or pd.isna(current_stoch):
                continue
            
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) >= 2 else current_price
            price_change_pct = (current_price - prev_price) / prev_price * 100 if prev_price != 0 else 0
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-10:].mean()
            relative_volume_pct = current_volume / avg_volume * 100 if avg_volume != 0 else 100
            
            if current_rsi >= 70:
                overbought_stocks.append((ticker, current_rsi, current_stoch, current_price, price_change_pct, relative_volume_pct))
            elif current_rsi <= 30:
                oversold_stocks.append((ticker, current_rsi, current_stoch, current_price, price_change_pct, relative_volume_pct))
                
        except Exception as e:
            logger.error(f"Ошибка при анализе RSI для {ticker}: {e}")
            continue
    
    # Сортировка
    overbought_stocks.sort(key=lambda x: x[1], reverse=True)
    oversold_stocks.sort(key=lambda x: x[1])
    
    # Формирование сообщения
    msg = f"📊 RSI и Стохастик на {datetime.now().strftime('%d.%m.%Y')}:\n\n"
    
    # 🔴 Перекупленные
    if overbought_stocks:
        msg += "🔴 Топ перекупленных акций (RSI ≥ 70):\n<pre>\n"
        msg += f"{'Тикер':<6} {'RSI':<4} {'STOCH':<6} {'Цена':<8} {'Изм %':<7} {'Отн.об %':<8}\n"
        msg += f"{'─'*6} {'─'*4} {'─'*6} {'─'*8} {'─'*7} {'─'*8}\n"
        for ticker, rsi_val, stoch_val, price, price_change_pct, rel_volume in overbought_stocks[:30]:
            msg += f"{ticker:<6} {rsi_val:4.0f} {stoch_val:6.0f} {price:8.1f} {price_change_pct:+6.1f}% {rel_volume:7.0f}%\n"
        msg += "</pre>\n\n"
    else:
        msg += "🔴 Перекупленных акций не найдено\n\n"
    
    # 🟢 Перепроданные
    if oversold_stocks:
        msg += "🟢 Топ перепроданных акций (RSI ≤ 30):\n<pre>\n"
        msg += f"{'Тикер':<6} {'RSI':<4} {'STOCH':<6} {'Цена':<8} {'Изм %':<7} {'Отн.об %':<8}\n"
        msg += f"{'─'*6} {'─'*4} {'─'*6} {'─'*8} {'─'*7} {'─'*8}\n"
        for ticker, rsi_val, stoch_val, price, price_change_pct, rel_volume in oversold_stocks[:30]:
            msg += f"{ticker:<6} {rsi_val:4.0f} {stoch_val:6.0f} {price:8.1f} {price_change_pct:+6.1f}% {rel_volume:7.0f}%\n"
        msg += "</pre>\n\n"
    else:
        msg += "🟢 Перепроданных акций не найдено\n\n"
    
    # Статистика
    total_analyzed = len(overbought_stocks) + len(oversold_stocks)
    msg += f"📈 Статистика:\n• Всего акций в зонах экстремума: {total_analyzed}\n"
    msg += f"• Перекупленных: {len(overbought_stocks)}\n• Перепроданных: {len(oversold_stocks)}\n"
    msg += f"• Фильтр по обороту: ≥50 млн ₽/день"
    
    await update.message.reply_text(msg, parse_mode="HTML")


# === Новая команда: long_moneyflow ===
def calculate_money_ad(df):
    df = df.copy()
    df['TYP'] = (df['high'] + df['low'] + df['close']) / 3
    df['CLV'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['CLV'] = df['CLV'].fillna(0)
    df['money_flow'] = df['CLV'] * df['volume'] * df['TYP']
    df['money_ad'] = df['money_flow'].cumsum()
    return df

@no_duplicate("moneyflow")
async def long_moneyflow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    days = context.user_data.get("days", 10)  # по умолчанию 10
    await update.message.reply_text(f"🔍 Ищу Топ по притоку и оттоку денежного потока за {days} дней...")
    
    result = []
    for ticker in sum(SECTORS.values(), []):
        try:
            df = get_moex_data(ticker, days=100)  # с запасом
            if df.empty or len(df) < days + 1:
                continue

            df = df.rename(columns={'close': 'close', 'volume': 'volume'})  # если еще не переименовано
            df = calculate_money_ad(df)

            ad_start = df['money_ad'].iloc[-(days+1)]
            ad_end = df['money_ad'].iloc[-1]
            ad_delta = ad_end - ad_start

            price_start = df['close'].iloc[-(days+1)]
            price_end = df['close'].iloc[-1]
            date_start = df.index[-(days+1)].strftime('%d.%m.%y')
            date_end = df.index[-1].strftime('%d.%m.%y')
            
            price_delta = price_end - price_start
            price_pct = 100 * price_delta / price_start

            # 💰 Среднедневной оборот за фиксированные 10 дней (для фильтра)
            filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
            filter_avg_turnover = filter_turnover_series.mean()
            
            # ❌ Фильтр по минимальному обороту: 50 млн руб за последние 10 дней
            if filter_avg_turnover < 50_000_000:
                continue
                
            # 💰 Среднедневной денежный оборот за период
            turnover_series = df['volume'].iloc[-days:] * df['close'].iloc[-days:]
            avg_turnover = turnover_series.mean()
            
            # Сегодняшний оборот
            today_volume = df['volume'].iloc[-1]
            today_close = df['close'].iloc[-1]
            today_turnover = today_volume * today_close
            
            # Коэффициент превышения объёма
            ratio = today_turnover / avg_turnover if avg_turnover > 0 else 0

            # EMA20/EMA50 Daily
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            current_ema20 = df['EMA20'].iloc[-1]
            current_ema50 = df['EMA50'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Условие для лонг сигнала EMA20x50
            ema20x50_long = (current_ema20 > current_ema50) and (current_price > current_ema20)
            # Условие для лонг сигнала EMA20x50
            ema20x50_short = (current_ema20 < current_ema50) and (current_price < current_ema20)

            # Изменение цены за день
            price_change = (current_price / df['close'].iloc[-2] - 1) if len(df) > 1 else 0

            # SMA30 Weekly
            try:
                wdf = get_moex_weekly_data(ticker, weeks=80)  # Больше недель для SMA30
                if len(wdf) >= 30:
                    wdf['SMA30'] = wdf['close'].rolling(window=30).mean()
                    weekly_sma30 = wdf['SMA30'].iloc[-1]
                    weekly_price = wdf['close'].iloc[-1]
                    price_above_sma30 = weekly_price > weekly_sma30 if pd.notna(weekly_sma30) else False
                else:
                    price_above_sma30 = False
            except:
                price_above_sma30 = False
            
            # 📊 Отношение дельты потока к обороту (%)
            if avg_turnover != 0:
                delta_vs_turnover = 100 * ad_delta / avg_turnover
            else:
                delta_vs_turnover = 0
            
            # 🪵 Лог для отладки
            print(f"{ticker} — Δ: {ad_delta:.2f}, Price %: {price_pct:.2f}, AvgTurn: {avg_turnover:.2f}, Δ% от оборота: {delta_vs_turnover:.2f}%")
            
            # Добавим в итог
            if ad_delta != 0:
                result.append((
                    ticker,
                    round(price_pct, 2),
                    round(ad_delta, 2),
                    date_start,
                    date_end,
                    round(delta_vs_turnover, 2),
                    price_change, 
                    ratio, 
                    ema20x50_long, 
                    ema20x50_short,
                    price_above_sma30,
            ))
        except Exception as e:
            print(f"Ошибка Money A/D для {ticker}: {e}")
            continue

    if not result:
        await update.message.reply_text("Не найдено активов с ростом или падением денежного потока (Money A/D)")
        return

    # Разделим на положительные и отрицательные дельты
    result_up = [r for r in result if r[2] > 0]
    result_down = [r for r in result if r[2] < 0]

    result_up.sort(key=lambda x: x[5], reverse=True)     # по убыванию
    result_down.sort(key=lambda x: x[5])                 # по возрастанию

    period = f"{result[0][3]}–{result[0][4]}"

    msg = f"🏦 Топ по денежному потоку за период {date_start}–{date_end}:\n\n"

    # 📈 Рост
    if result_up:
        msg += "📈 Топ 10 по притоку:\n"
        msg += "<pre>\n"
        msg += f"{'Тикер':<6}  {'Δ Цены':<9}  {'Δ Потока':>11}  {'Δ / Оборот':>8} {'Δ Цены 1D':>8} {'Объём':>8} {'ema20х50':>7} {'sma30':>4}\n"
        # Убираем линию с дефисами, как просил
        for ticker, price_pct, ad_delta, _, _, delta_pct, price_change_day, ratio, ema20x50_long, ema20x50_short, sma_signal in result_up[:10]:
            if ema20x50_long:
                ema_icon = "🟢"
            elif ema20x50_short:
                ema_icon = "🔴"
            else:
                ema_icon = "⚫"
            sma_icon = "🟢" if sma_signal else "🔴"
            msg += f"{ticker:<6}  {price_pct:5.1f}%  {ad_delta/1_000_000:8,.0f} млн ₽  {delta_pct:8.1f}%  {price_change_day*100:>8.1f}%  {ratio:>6.1f}x  {ema_icon:>5} {sma_icon:>4}\n"
        msg += "</pre>\n\n"
    
    # 📉 Падение
    if result_down:
        msg += "📉 Топ 10 по оттоку:\n"
        msg += "<pre>\n"
        msg += f"{'Тикер':<6}  {'Δ Цены':<9}  {'Δ Потока':>11}  {'Δ / Оборот':>8} {'Δ Цены 1D':>8} {'Объём':>8} {'ema20х50':>7} {'sma30':>4}\n"
        # Линию тоже убираем
        for ticker, price_pct, ad_delta, _, _, delta_pct, price_change_day, ratio, ema20x50_long, ema20x50_short, sma_signal in result_down[:10]:
            if ema20x50_long:
                ema_icon = "🟢"
            elif ema20x50_short:
                ema_icon = "🔴"
            else:
                ema_icon = "⚫"
            sma_icon = "🟢" if sma_signal else "🔴"
            msg += f"{ticker:<6}  {price_pct:5.1f}%  {ad_delta/1_000_000:8,.0f} млн ₽  {delta_pct:8.1f}%  {price_change_day*100:>8.1f}%  {ratio:>6.1f}x  {ema_icon:>5} {sma_icon:>4}\n"
        msg += "</pre>\n"
    
    await update.message.reply_text(msg, parse_mode="HTML")



# Telegram команды

@no_duplicate("stan_rec")
async def stan_recent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу акции с недавним long пересечением цены через SMA30...")
    
    crossovers = []
    all_tickers = sum(SECTORS.values(), [])
    
    # Проверяем каждый тикер
    for ticker in all_tickers:
        try:
            crossover_date = find_sma30_crossover(ticker, days=7)
            if crossover_date:
                crossovers.append((ticker, crossover_date))
        except Exception as e:
            print(f"Ошибка при анализе {ticker}: {e}")
            continue
    
    if not crossovers:
        await update.message.reply_text("📊 За последние 7 дней не найдено акций с пересечением цены через SMA30 снизу вверх.")
        return
    
    # Сортируем по дате (от самого свежего к самому старому)
    crossovers.sort(key=lambda x: x[1], reverse=True)
    
    # Формируем результат
    result_text = "📈 Акции с пересечением цены через SMA30 снизу вверх за последние 7 дней:\n\n"
    
    for ticker, date in crossovers:
        formatted_date = date.strftime('%d.%m.%Y')
        result_text += f"{ticker} {formatted_date}\n"
    
    result_text += f"\n🔢 Всего найдено: {len(crossovers)} акций"
    
    await update.message.reply_text(result_text)


@no_duplicate("stan_short")
async def stan_recent_d_short(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу акции с недавним short пересечением цены через SMA30...")
    
    crossovers = []
    all_tickers = sum(SECTORS.values(), [])
    
    # Проверяем каждый тикер
    for ticker in all_tickers:
        try:
            crossover_date = find_sma30_crossover_short(ticker, days=7)
            if crossover_date:
                crossovers.append((ticker, crossover_date))
        except Exception as e:
            print(f"Ошибка при анализе {ticker}: {e}")
            continue
    
    if not crossovers:
        await update.message.reply_text("📊 За последние 7 дней не найдено акций с пересечением цены через SMA30 сверху вниз.")
        return
    
    # Сортируем по дате (от самого свежего к самому старому)
    crossovers.sort(key=lambda x: x[1], reverse=True)
    
    # Формируем результат
    result_text = "📈 Акции с Short пересечением цены через SMA30 сверху вниз за последние 7 дней:\n\n"
    
    for ticker, date in crossovers:
        formatted_date = date.strftime('%d.%m.%Y')
        result_text += f"{ticker} {formatted_date}\n"
    
    result_text += f"\n🔢 Всего найдено: {len(crossovers)} акций"
    
    await update.message.reply_text(result_text)

@no_duplicate("stan_week")
async def stan_recent_week(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу акции с недавним long пересечением цены через SMA30… (параллельный режим)")
    all_tickers = sum(SECTORS.values(), [])
    weeks = 5

    # Волна 1: все недельные параллельно
    async def _fetch_weekly(ticker: str):
        try:
            return ticker, await async_get_moex_weekly_data(ticker, weeks=60)
        except Exception as e:
            print(f"[stan_recent_week] weekly error {ticker}: {e}")
            return ticker, None

    weekly_results = await asyncio.gather(*[_fetch_weekly(t) for t in all_tickers])
    weekly_map = {t: wdf for t, wdf in weekly_results}

    # Быстрый фильтр в памяти: цена > SMA30 и есть crossover
    weekly_candidates = []
    for ticker, wdf in weekly_map.items():
        if wdf is None or wdf.empty or len(wdf) < 35:
            continue
        wdf_copy = wdf.copy()
        wdf_copy['SMA30'] = wdf_copy['close'].rolling(window=30).mean()
        current_sma30 = wdf_copy['SMA30'].iloc[-1]
        if pd.isna(current_sma30) or wdf_copy['close'].iloc[-1] <= current_sma30:
            continue
        recent = wdf_copy.tail(weeks + 1)
        crossed = any(
            recent['close'].iloc[i - 1] < recent['SMA30'].iloc[i - 1] and
            recent['close'].iloc[i]     > recent['SMA30'].iloc[i]
            for i in range(1, len(recent))
        )
        if crossed:
            weekly_candidates.append(ticker)

    # Волна 2: дневные только для кандидатов (фильтр оборота)
    async def _fetch_daily(ticker: str):
        try:
            return ticker, await async_get_moex_data(ticker, days=20)
        except Exception as e:
            print(f"[stan_recent_week] daily error {ticker}: {e}")
            return ticker, None

    daily_results = await asyncio.gather(*[_fetch_daily(t) for t in weekly_candidates])
    daily_map = {t: dfd for t, dfd in daily_results}

    crossovers = []
    for ticker in weekly_candidates:
        try:
            dfd = daily_map.get(ticker)
            if dfd is None or dfd.empty or len(dfd) < 15:
                continue
            if (dfd['volume'].iloc[-10:] * dfd['close'].iloc[-10:]).mean() < 50_000_000:
                continue
            wdf_copy = weekly_map[ticker].copy()
            wdf_copy['SMA30'] = wdf_copy['close'].rolling(window=30).mean()
            recent = wdf_copy.tail(weeks + 1)
            crossover_date = None
            for i in range(1, len(recent)):
                if (recent['close'].iloc[i - 1] < recent['SMA30'].iloc[i - 1] and
                        recent['close'].iloc[i] > recent['SMA30'].iloc[i]):
                    crossover_date = recent.index[i]
                    break
            if crossover_date:
                crossovers.append((ticker, crossover_date))
        except Exception as e:
            print(f"[stan_recent_week] final check error {ticker}: {e}")
            continue

    if not crossovers:
        await update.message.reply_text("📊 За последние 5 недель не найдено акций с пересечением цены через SMA30 снизу вверх.")
        return

    crossovers.sort(key=lambda x: x[1], reverse=True)
    result_text = "📈 Акции с пересечением цены через SMA30 снизу вверх за последние 5 недель:\n\n"
    for ticker, date in crossovers:
        result_text += f"{ticker} {date.strftime('%d.%m.%Y')}\n"
    result_text += f"\n🔢 Всего найдено: {len(crossovers)} акций"
    await update.message.reply_text(result_text)


# Обработчики callback
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    try:
        if data == "back_to_sectors":
            keyboard = [[InlineKeyboardButton(sector, callback_data=f"sector:{sector}:0")] for sector in SECTORS]
            await query.edit_message_text("Выберите отрасль:", reply_markup=InlineKeyboardMarkup(keyboard))

        elif data.startswith("sector:"):
            _, sector, page = data.split(":")
            page = int(page)
            tickers = SECTORS.get(sector, [])
            start = page * TICKERS_PER_PAGE
            end = start + TICKERS_PER_PAGE
            visible = tickers[start:end]

            keyboard = [[InlineKeyboardButton(t, callback_data=f"ticker:{t}")] for t in visible]
            nav = []
            if start > 0:
                nav.append(InlineKeyboardButton("⬅️", callback_data=f"sector:{sector}:{page-1}"))
            if end < len(tickers):
                nav.append(InlineKeyboardButton("➡️", callback_data=f"sector:{sector}:{page+1}"))
            if nav:
                keyboard.append(nav)
            keyboard.append([InlineKeyboardButton("🔙 Назад к отраслям", callback_data="back_to_sectors")])

            await query.edit_message_text(f"Вы выбрали отрасль: {sector}. Теперь выберите тикер:", reply_markup=InlineKeyboardMarkup(keyboard))

        elif data.startswith("ticker:"):
            ticker = data.split(":", 1)[1]
            await query.edit_message_text(f"Вы выбрали тикер: {ticker}. Выполняется анализ...")

            df = get_moex_data(ticker)
            if df.empty:
                await context.bot.send_message(chat_id=query.message.chat.id, text=f"❌ Не удалось получить данные для {ticker}")
                return

            df = analyze_indicators(df)

            
            if chart is None:
                await context.bot.send_message(chat_id=query.message.chat.id, text=f"❌ Ошибка при создании графика для {ticker}")
                return

            rsi_series = df['RSI'].dropna()
            rsi_value = rsi_series.iloc[-1] if not rsi_series.empty else "Недостаточно данных для RSI"
            latest_date = df.index.max().strftime('%Y-%m-%d')

            text_summary = f"\nПоследний RSI: {rsi_value}\n"
            text_summary += f"Актуальность данных: до {latest_date}\n"

            with open(chart, 'rb') as photo:
                await context.bot.send_photo(chat_id=query.message.chat.id, photo=photo)
            await context.bot.send_message(chat_id=query.message.chat.id, text=text_summary)
            
            if os.path.exists(chart):
                os.remove(chart)

    except Exception as e:
        await context.bot.send_message(chat_id=query.message.chat.id, text=f"❌ Произошла ошибка: {str(e)}")

# === ЗАПУСК БОТА ===

if __name__ == '__main__':

#    import caching
#    caching.enable_caching()
    
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TOKEN:
        print("❌ Переменная окружения TELEGRAM_TOKEN не установлена.")
        exit()

    # Создаём приложение
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .concurrent_updates(True)
        .build()
    )
    
    # === Добавляем хендлеры ===
    app.add_handler(CommandHandler("start", start))
    
    # ДОЛГИЕ КОМАНДЫ — обязательно block=False
    app.add_handler(CommandHandler("cross_ema20x50", cross_ema20x50, block=False))
    app.add_handler(CommandHandler("cross_ema20x50_4h", cross_ema20x50_4h, block=False))
    app.add_handler(CommandHandler("cross_ema9x50", cross_ema9x50, block=False))
    app.add_handler(CommandHandler("cross_ema200", cross_ema200, block=False))
    app.add_handler(CommandHandler("stan_recent", stan_recent, block=False))
    app.add_handler(CommandHandler("stan_recent_d_short", stan_recent_d_short, block=False))
    app.add_handler(CommandHandler("stan_recent_week", stan_recent_week, block=False))
    app.add_handler(CommandHandler("long_moneyflow", long_moneyflow, block=False))
    app.add_handler(CommandHandler("high_volume", high_volume, block=False))
    app.add_handler(CommandHandler("rsi_top", rsi_top, block=False))

    # app.add_handler(CommandHandler("cache_debug", cache_debug))
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Хендлеры с диалогами
    delta_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("delta", ask_ticker)],
        states={
            ASK_TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_ticker)],
            ASK_DELTA_DAYS: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_delta_days)]
        },
        fallbacks=[],
    )
    app.add_handler(delta_conv_handler)

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("moneyflow", ask_days)],
        states={
            ASK_DAYS: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_days)]
        },
        fallbacks=[],
    )
    app.add_handler(conv_handler)

    # Запуск с Webhook
    print("🚀 Запускаем бота через webhook...")

    app.run_webhook(
        listen="0.0.0.0",
        port=8080,
        url_path=TOKEN, 
        webhook_url=f"https://moex-telegram-bot-sra8.onrender.com/{TOKEN}"
    )
