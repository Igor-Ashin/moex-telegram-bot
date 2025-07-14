# main.py (moex_stock_bot.py с интеграцией кэширования)

import matplotlib
matplotlib.use('Agg')  # Включаем "безголовый" режим для matplotlib
import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import asyncio
import html

# === ИНТЕГРАЦИЯ КЭШИРОВАНИЯ ===
# Импортируем модуль кэширования в самом начале
try:
    import caching  # Автоматически включит кэширование
    print("✅ Модуль кэширования загружен успешно")
except ImportError:
    print("ℹ️ Модуль кэширования не найден, работаем без кэша")

# Активация Токена Tinkoff
from tinkoff.invest import Client, CandleInterval

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

if __name__ == "__main__":
    set_webhook()

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
    "Финансы": ["SBER", "T", "VTBR", "MOEX", "SPBE", "RENI", "BSPB", "SVCB", "MBNK", "LEAS", "SFIN", "AFKS", "CARM", "ZAYM", "MGKL"],
    "Нефтегаз": ["GAZP", "NVTK", "LKOH", "ROSN", "TATNP", "TATN", "SNGS", "SNGSP", "BANE", "BANEP", "RNFT"],
    "Металлы и добыча": ["ALRS", "GMKN", "RUAL", "TRMK", "MAGN", "NLMK", "CHMF", "MTLRP", "MTLR", "RASP", "PLZL", "UGLD", "SGZH"],
    "IT": ["YDEX", "DATA", "HEAD", "POSI", "VKCO", "ASTR", "IVAT", "DELI", "WUSH", "CNRU", "DIAS", "SOFL", "ELMT"],
    "Телеком": ["MTSS", "RTKMP", "RTKM", "MGTSP"],
    "Строители": ["SMLT", "PIKK", "LSRG"],
    "Ритейл": ["X5", "MGNT", "LENT", "BELU", "OZON", "EUTR", "ABRD", "GCHE", "AQUA", "HNFG", "MVID", "VSEH"],
    "Электро": ["IRAO", "UPRO", "LSNGP", "MSRS", "MRKU", "MRKC", "MRKP", "FEES", "HYDR", "ELFV"],
    "Транспорт и логистика": ["TRNFP", "AFLT", "FESH", "NMTP", "FLOT"],
    "Агро": ["PHOR", "RAGR", "KZOS", "AKRN", "NKHP"],
    "Медицина": ["MDMG", "OZPH", "PRMD", "ABIO", "GEMC"],
    "Машиностроение": ["UWGN", "SVAV", "KMAZ", "UNAC", "IRKT"]
}

SECTORS1 = {
    "Финансы": ["SBER", "T", "VTBR", "MOEX", "SPBE", "RENI", "BSPB", "SVCB", "MBNK", "LEAS", "SFIN", "AFKS"],
    "Нефтегаз": ["GAZP", "NVTK", "LKOH", "ROSN", "TATNP", "TATN", "SNGS", "SNGSP", "BANE", "BANEP", "RNFT"],
    "Металлы и добыча": ["ALRS", "GMKN", "RUAL", "TRMK", "MAGN", "NLMK", "CHMF", "MTLRP", "MTLR", "PLZL", "SGZH"],
    "IT": ["YDEX", "DATA", "HEAD", "POSI", "VKCO", "ASTR", "DELI", "WUSH", "CNRU", "DIAS"],
    "Телеком": ["MTSS", "RTKMP", "RTKM"],
    "Строители": ["SMLT", "PIKK"],
    "Ритейл": ["X5", "MGNT", "LENT", "BELU", "OZON", "EUTR", "ABRD", "GCHE", "AQUA", "HNFG", "MVID"],
    "Электро": ["IRAO", "UPRO", "LSNGP", "MRKP"],
    "Транспорт и логистика": ["TRNFP", "AFLT", "FESH", "NMTP", "FLOT"],
    "Агро": ["PHOR", "RAGR"],
    "Медицина": ["MDMG", "OZPH", "PRMD"],
    "Машиностроение": ["UWGN", "SVAV"]
}

TICKERS_PER_PAGE = 10

# Состояния для диалогов
ASK_DAYS = 1
ASK_TICKER = 2
ASK_DELTA_DAYS = 3

# === ФУНКЦИИ ПОЛУЧЕНИЯ ДАННЫХ ===

def get_moex_data(ticker="SBER", days=120):
    """Получение дневных данных с MOEX"""
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

def get_moex_weekly_data(ticker="SBER", weeks=80):
    """Получение недельных данных с MOEX"""
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

def get_figi_by_ticker(ticker: str) -> str | None:
    """Получение FIGI по тикеру"""
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

def get_moex_data_4h_tinkoff(ticker: str = "SBER", days: int = 25) -> pd.DataFrame:
    """Загружает 4H свечи по тикеру из Tinkoff Invest API"""
    try:
        figi = get_figi_by_ticker(ticker)
        if figi is None:
            print(f"❌ FIGI для тикера {ticker} не найдено")
            return pd.DataFrame()
            
        print(f"📡 Используем FIGI {figi} для загрузки данных {ticker}")
        
        to_dt = datetime.utcnow()
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

# === ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ===

def compute_rsi(series, window=14):
    """Вычисляет RSI используя pandas ewm для сглаживания Wilder's"""
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

def calculate_money_ad(df):
    """Расчет Money A/D индикатора"""
    df = df.copy()
    df['TYP'] = (df['high'] + df['low'] + df['close']) / 3
    df['CLV'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['CLV'] = df['CLV'].fillna(0)
    df['money_flow'] = df['CLV'] * df['volume'] * df['TYP']
    df['money_ad'] = df['money_flow'].cumsum()
    return df

def analyze_indicators(df):
    """Анализ технических индикаторов"""
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

# === ФУНКЦИИ ПОИСКА ПАТТЕРНОВ ===

def find_levels(df):
    """Поиск уровней поддержки и сопротивления"""
    if df.empty:
        return []
    
    levels = []
    closes = df['close'].values
    local_max = argrelextrema(closes, np.greater)[0]
    local_min = argrelextrema(closes, np.less)[0]

    extrema = sorted([(i, closes[i]) for i in np.concatenate((local_max, local_min))], key=lambda x: x[1])
    if len(extrema) > 0:
        grouped = pd.Series([round(p[1], 1) for p in extrema]).value_counts()
        strong_levels = grouped[grouped > 1].index.tolist()
        for level in strong_levels:
            for i, val in extrema:
                if abs(val - level) < 0.5:
                    levels.append((df.index[i], val))
                    break
    return levels

def detect_double_patterns(df):
    """Обнаружение двойных вершин и дна"""
    if df.empty or len(df) < 5:
        return []
    
    closes = df['close'].values
    patterns = []
    for i in range(2, len(closes) - 2):
        if closes[i-2] < closes[i-1] < closes[i] and closes[i] > closes[i+1] > closes[i+2]:
            patterns.append(('Double Top', df.index[i], closes[i]))
        if closes[i-2] > closes[i-1] > closes[i] and closes[i] < closes[i+1] < closes[i+2]:
            patterns.append(('Double Bottom', df.index[i], closes[i]))
    return patterns

# === ФУНКЦИИ ПОСТРОЕНИЯ ГРАФИКОВ ===

def plot_stock(df, ticker, levels=[], patterns=[]):
    """Построение графика акции с техническим анализом"""
    if df.empty:
        return None
    
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Цена', color='blue')

        plt.plot(df.index, df['EMA9'], label='EMA9', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['EMA20'], label='EMA20', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['EMA50'], label='EMA50', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['EMA100'], label='EMA100', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['EMA200'], label='EMA200', linestyle='--', alpha=0.7)

        # Аномальные объемы
        for idx in df[df['Anomaly']].index:
            volume_ratio = df.loc[idx, 'Volume_Multiplier']
            plt.scatter(idx, df.loc[idx, 'close'], color='red')
            plt.text(idx, df.loc[idx, 'close'], f"{volume_ratio:.1f}x", color='red', fontsize=8, ha='left')

        # Уровни поддержки/сопротивления
        for date, price in levels:
            plt.axhline(price, linestyle='--', alpha=0.3)

        # Паттерны
        plotted_top = False
        plotted_bottom = False
        for name, date, price in patterns:
            if name == 'Double Top':
                marker = '^'
                color = 'red'
                label = 'Double Top' if not plotted_top else None
                plotted_top = True
            else:
                marker = 'v'
                color = 'green'
                label = 'Double Bottom' if not plotted_bottom else None
                plotted_bottom = True
            plt.scatter(date, price, label=label, s=100, marker=marker, color=color)

        plt.title(f"{ticker}: График с анализом")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f"{ticker}_analysis.png"
        plt.savefig(filename)
        plt.close()
        return filename
    except Exception as e:
        print(f"Ошибка построения графика для {ticker}: {e}")
        plt.close()
        return None

def plot_stan_chart(df, ticker):
    """Построение графика по методу Вайнштейна"""
    if df.empty:
        return None
    
    try:
        df['SMA30'] = df['close'].rolling(window=30).mean()
        df['Upper'] = df['SMA30'] + 2 * df['close'].rolling(window=30).std()
        df['Lower'] = df['SMA30'] - 2 * df['close'].rolling(window=30).std()

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Цена', color='blue')
        plt.plot(df.index, df['SMA30'], label='SMA 30', linewidth=2.5, color='black')
        plt.plot(df.index, df['Upper'], label='BB верх', linestyle='--', color='gray')
        plt.plot(df.index, df['Lower'], label='BB низ', linestyle='--', color='gray')

        plt.title(f"Вайнштейн: {ticker} на 1W ТФ")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f"{ticker}_stan.png"
        plt.savefig(filename)
        plt.close()
        return filename
    except Exception as e:
        print(f"Ошибка построения графика для {ticker}: {e}")
        plt.close()
        return None

# === ФУНКЦИИ ПОИСКА ПЕРЕСЕЧЕНИЙ ===

def find_sma30_crossover(ticker, days=7):
    """Находит пересечение цены снизу вверх через SMA30"""
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
    """Находит пересечение цены сверху вниз через SMA30"""
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
    """Находит пересечение цены снизу вверх через SMA30 на недельном ТФ"""
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

if Update and ContextTypes:
    
    # Функция для получения статистики кэша
    def get_cache_stats():
        """Возвращает статистику кэша если модуль загружен"""
        try:
            if 'caching' in globals():
                return caching.get_cache_stats()
            else:
                return {'entries': 0, 'size_mb': 0, 'status': 'disabled'}
        except:
            return {'entries': 0, 'size_mb': 0, 'status': 'error'}

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Получаем статистику кэша
        cache_stats = get_cache_stats()
        
        if cache_stats.get('status') == 'disabled':
            cache_info = "🔄 Кэширование отключено\n"
        elif cache_stats.get('status') == 'error':
            cache_info = "⚠️ Ошибка кэширования\n"
        else:
            cache_info = f"📊 Кэш: {cache_stats.get('entries', 0)} записей, {cache_stats.get('size_mb', 0)} MB\n"
        
        text = (
            "Привет! Это бот от команды @TradeAnsh для анализа акций Мосбиржи.\n"
            f"{cache_info}"
            "Команды:\n"
            "/chart_hv — выбрать акцию через кнопки\n"
            "/stan — анализ акции по методу Стэна Вайнштейна\n"
            "/cross_ema20x50 — акции с пересечением EMA 20x50 на 1D\n"
            "/cross_ema20x50_4h — акции с пересечением EMA 20x50 на 4H\n"
            "/stan_recent — акции с лонг пересечением SMA30 на 1D\n"
            "/stan_recent_short — акции с шорт пересечением SMA30 на 1D\n"
            "/stan_recent_week — акции с лонг пересечением SMA30 на 1W\n"
            "/moneyflow - Топ по росту и оттоку денежного потока (Money A/D)\n"
            "/high_volume - Акции с повышенным объемом\n"
            "/delta — расчет дельты денежного потока для конкретной акции\n"
            "/rsi_top — Топ 10 перекупленных и перепроданных акций по RSI\n"
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

    async def receive_delta_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            days = int(update.message.text)
            if not (1 <= days <= 100):
                await update.message.reply_text("⚠️ Введите число от 1 до 100.")
                return ASK_DELTA_DAYS

            ticker_input = context.user_data['delta_ticker']
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

            if not tickers:
                await update.message.reply_text("⚠️ Не удалось распознать тикеры.")
                return ConversationHandler.END

            await update.message.reply_text(f"🔎 Обрабатываю {len(tickers)} тикеров за {days} дней...")
            
            for ticker in tickers:
                await calculate_single_delta(update, context, ticker, days)
                await asyncio.sleep(0.5)
            
            return ConversationHandler.END

        except ValueError:
            await update.message.reply_text("⚠️ Введите целое число, например: 10")
            return ASK_DELTA_DAYS

    async def calculate_single_delta(update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str, days: int):
        """Рассчитывает дельту денежного потока для одной акции"""
        await update.message.reply_text(f"🔍 Рассчитываю дельту денежного потока для {ticker} за {days} дней...")
        
        try:
            df = get_moex_data(ticker, days=100)
            if df.empty or len(df) < days + 1:
                await update.message.reply_text(f"❌ Недостаточно данных для {ticker}. Попробуйте увеличить количество дней.")
                return

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

            # Фильтр по обороту
            filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
            filter_avg_turnover = filter_turnover_series.mean()
            
            # Среднедневной оборот за период
            turnover_series = df['volume'].iloc[-days:] * df['close'].iloc[-days:]
            avg_turnover = turnover_series.mean()

            # Сегодняшний оборот
            today_volume = df['volume'].iloc[-1]
            today_close = df['close'].iloc[-1]
            today_turnover = today_volume * today_close
            
            ratio = today_turnover / avg_turnover if avg_turnover > 0 else 0

            # EMA20/EMA50 Daily
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            current_ema20 = df['EMA20'].iloc[-1]
            current_ema50 = df['EMA50'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            ema20x50_long = (current_ema20 > current_ema50) and (current_price > current_ema20)
            ema20x50_short = (current_ema20 < current_ema50) and (current_price < current_ema20)

            price_change_day = (current_price / df['close'].iloc[-2] - 1) if len(df) > 1 else 0

            # SMA30 Weekly
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
            
            if avg_turnover != 0:
                delta_pct = 100 * ad_delta / avg_turnover
            else:
                delta_pct = 0

            # Формируем сообщение
            msg = f"📊 *Анализ дельты денежного потока для {ticker}*\n"
            msg += f"📅 *Период: {date_start} – {date_end} ({days} дней)*\n\n"
            
            if filter_avg_turnover < 50_000_000:
                msg += "⚠️ *Внимание: низкий среднедневной оборот (< 50 млн ₽)*\n\n"

            if ema20x50_long:
                ema_icon = "🟢"
            elif ema20x50_short:
                ema_icon = "🔴"
            else:
                ema_icon = "⚫"

            sma_icon = "🟢" if price_above_sma30 else "🔴"
            flow_icon = "🟢" if ad_delta > 0 else "🔴"
            
            msg += f"*Δ Цены за период:* {price_pct:+.1f}%\n"
            msg += f"*Δ Потока:* {ad_delta/1_000_000:+.0f} млн ₽ {flow_icon}   *Δ / Оборот:* {delta_pct:.1f}%\n"
            msg += f"*Δ Цены 1D:* {price_change_day*100:+.1f}%   *Объём:* {ratio:.1f}x\n"
            msg += f"*EMA20x50:* {ema_icon}   *SMA30:* {sma_icon}\n\n"
            msg += f"💰 *Среднедневной оборот:* {avg_turnover/1_000_000:.1f} млн ₽\n"
            
            await update.message.reply_text(msg, parse_mode="Markdown")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при расчете дельты для {ticker}: {str(e)}")

    # Основные команды анализа
    async def chart_hv(update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [[InlineKeyboardButton(sector, callback_data=f"sector:{sector}:0")] for sector in SECTORS]
        await update.message.reply_text("Выберите отрасль:", reply_markup=InlineKeyboardMarkup(keyboard))

    async def stan(update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [[InlineKeyboardButton(sector, callback_data=f"stan_sector:{sector}:0")] for sector in SECTORS]
        await update.message.reply_text("Выберите отрасль для анализа по Штейну:", reply_markup=InlineKeyboardMarkup(keyboard))

    async def high_volume(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔍 Ищу акции с повышенным объёмом…")
        rows = []
        
        for ticker in sum(SECTORS.values(), []):
            try:
                df = get_moex_data(ticker, days=100)
                if df.empty or len(df) < 60: 
                    continue
                    
                # Расчёт среднего оборота за 10 дней
                volume_series = df['volume'].iloc[-11:-1]
                close_series = df['close'].iloc[-11:-1]
                turnover_series = volume_series * close_series
                avg_turnover = turnover_series.mean()
                
                # Сегодняшний оборот
                today_volume = df['volume'].iloc[-1]
                today_close = df['close'].iloc[-1]
                today_turnover = today_volume * today_close
                
                ratio = today_turnover / avg_turnover if avg_turnover > 0 else 0
                
                if ratio < 1.2:
                    continue
                    
                # EMA20/EMA50 Daily
                df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
                df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
                
                current_ema20 = df['EMA20'].iloc[-1]
                current_ema50 = df['EMA50'].iloc[-1]
                current_price = df['close'].iloc[-1]
                
                ema20x50_long = (current_ema20 > current_ema50) and (current_price > current_ema20)
                
                price_change = (current_price / df['close'].iloc[-2] - 1) if len(df) > 1 else 0
                
                # SMA30 Weekly
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

                # Money Flow A/D
                money_df = calculate_money_ad(df)
                ad_delta = money_df['money_ad'].iloc[-1] - money_df['money_ad'].iloc[-11]
                money_flow_icon = "🟢" if ad_delta > 0 else "🔴"
                money_flow_str = f"{ad_delta/1_000_000:+.0f}M"
                
                rows.append((
                    ticker, 
                    current_price, 
                    price_change, 
                    ratio, 
                    ema20x50_long, 
                    price_above_sma30,
                    money_flow_icon,
                    money_flow_str
                ))
                
            except Exception as e:
                print(f"Ошибка для {ticker}: {e}")
                continue
        
        rows.sort(key=lambda x: x[3], reverse=True)
        rows = rows[:15]
        
        if not rows:
            await update.message.reply_text("📊 Акций с повышенным объёмом не найдено")
            return
        
        msg = "📊 <b>Акции с повышенным объёмом</b>\n\n"
        msg += "<pre>"
        msg += f"{'Тикер':<6} {'Цена':>8} {'Δ Цены':>7} {'Объём':>6} {'ema20x50':>6} {'sma30':>6} {'Δ Потока':>10}\n"
        msg += "-" * 60 + "\n"
        
        for ticker, price, delta, ratio, ema_signal, sma_signal, mf_icon, mf_str in rows:
            ema_icon = "🟢" if ema_signal else "🔴"
            sma_icon = "🟢" if sma_signal else "🔴"
            
            msg += f"{ticker:<6} {price:>8.2f} {delta*100:>6.1f}% {ratio:>5.1f}x {ema_icon:>6} {sma_icon:>4} {mf_icon}{mf_str:>6}\n"
        
        msg += "</pre>\n\n"
        msg += "<i>EMA - пересечение EMA20x50 (D) на дневном ТФ</i>\n"
        msg += "<i>SMA - цена выше SMA30 на недельном ТФ</i>\n"
        msg += "<i>Δ Потока - приток/отток денежных средств (посл. 10 дней)</i>"
        
        await update.message.reply_text(msg, parse_mode="HTML")

    # Остальные команды (cross_ema20x50, cross_ema20x50_4h, long_moneyflow, rsi_top и т.д.)
    # ... [здесь должны быть все остальные команды из оригинального файла]

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
                levels = find_levels(df)
                patterns = detect_double_patterns(df)
                chart = plot_stock(df, ticker, levels, patterns)
                
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
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TOKEN:
        print("❌ Переменная окружения TELEGRAM_TOKEN не установлена.")
        exit()

    # Создаём приложение
    app = ApplicationBuilder().token(TOKEN).build()

    # Добавляем хендлеры
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chart_hv", chart_hv))
    app.add_handler(CommandHandler("stan", stan))
    app.add_handler(CommandHandler("high_volume", high_volume))
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
