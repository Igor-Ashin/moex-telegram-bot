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

# === ОТЛОЖЕННАЯ АКТИВАЦИЯ КЭШИРОВАНИЯ ===
try:
    if 'caching' in globals():
        success = caching.activate_caching_if_enabled()
        if success:
            print("🎯 Кэширование активировано (отложенная активация)")
        else:
            print("⚠️ Отложенная активация кэширования не удалась")
except Exception as e:
    print(f"❌ Ошибка отложенной активации: {e}")


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


async def cache_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отладочная команда для проверки кэша"""
    try:
        if 'caching' in globals():
            stats = caching.get_cache_stats()
            msg = f"🔍 **Отладка кэширования:**\n\n"
            msg += f"📊 Статистика:\n"
            msg += f"• MOEX кэш: {stats['moex_entries']} записей\n"
            msg += f"• Weekly кэш: {stats['weekly_entries']} записей\n"
            msg += f"• FIGI кэш: {stats['figi_entries']} записей\n"
            msg += f"• Общий размер: {stats['size_mb']} MB\n\n"
            
            # Проверяем, заменены ли функции
            import sys
            if 'main' in sys.modules:
                main_module = sys.modules['main']
                msg += f"🔧 Замена функций:\n"
                msg += f"• get_moex_data: {'✅' if hasattr(main_module, '_original_get_moex_data') else '❌'}\n"
                msg += f"• get_moex_weekly_data: {'✅' if hasattr(main_module, '_original_get_moex_weekly_data') else '❌'}\n"
                msg += f"• get_figi_by_ticker: {'✅' if hasattr(main_module, '_original_get_figi_by_ticker') else '❌'}\n"
        else:
            msg = "❌ Модуль caching не загружен"
            
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка отладки: {e}")
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
            "/stan_recent_d_short — акции с шорт пересечением SMA30 на 1D\n"
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


async def cross_ema20x50(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу пересечения EMA20 и EMA50 за последние 14 дней...")
    long_hits, short_hits = [], []
    today = datetime.today().date()
    
    for ticker in sum(SECTORS.values(), []):
        try:
            df = get_moex_data(ticker, days=100)  # достаточно для расчета EMA
            if df.empty or len(df) < 100:
                continue
                
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            # Получаем данные за последние 15 дней для анализа
            recent = df.tail(15)  # 14 дней + текущий
            
            # Текущие значения
            current_close = df['close'].iloc[-1]
            current_ema20 = df['EMA20'].iloc[-1]
            current_ema50 = df['EMA50'].iloc[-1]
            
            # Проверяем пересечения за последние 14 дней
            for i in range(1, len(recent)):
                prev_ema20 = recent['EMA20'].iloc[i-1]
                prev_ema50 = recent['EMA50'].iloc[i-1]
                curr_ema20 = recent['EMA20'].iloc[i]
                curr_ema50 = recent['EMA50'].iloc[i]
                curr_close = recent['close'].iloc[i]  # Цена в день пересечения
                
                # Получаем дату для текущего дня
                date = recent.index[i].strftime('%d.%m.%Y')
                
                # Лонг пересечение: EMA20 пересекает EMA50 снизу вверх + подтверждение
                if (
                    prev_ema20 <= prev_ema50
                    and curr_ema20 > curr_ema50
                    and curr_close > curr_ema20
                    and current_close > current_ema20
                    and current_ema20 > current_ema50
                ):
                    long_hits.append((ticker, date))
                    break  # Только одно пересечение за период
        
                # Шорт пересечение: EMA20 пересекает EMA50 сверху вниз + подтверждение
                elif (
                    prev_ema20 >= prev_ema50
                    and curr_ema20 < curr_ema50
                    and curr_close < curr_ema20
                    and current_close < current_ema20
                    and current_ema20 < current_ema50
                ):
                    short_hits.append((ticker, date))
                    break  # Только одно пересечение за период
                    
        except Exception as e:
            print(f"Ошибка EMA для {ticker}: {e}")
            continue
    
    # Сортировка по дате (новые вверх)
    long_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y'), reverse=True)
    short_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y'), reverse=True)
    
    # Формируем сообщение
    msg = ""
    if long_hits:
        msg += f"🟢 *Лонг пересечение EMA20×50 за последние 14 дней, всего: {len(long_hits)}:*\n"
        msg += "\n".join(f"{t} {d}" for t, d in long_hits) + "\n\n"
    else:
        msg += "🟢 *Лонг сигналов не найдено за последние 14 дней*\n\n"
        
    if short_hits:
        msg += f"🔴 *Шорт пересечение EMA20×50 за последние 14 дней, всего: {len(short_hits)}:*\n"
        msg += "\n".join(f"{t} {d}" for t, d in short_hits)+ "\n\n"
    else:
        msg += "🔴 *Шорт сигналов не найдено за последние 14 дней*\n\n"
    #msg += "\n"   
    # Добавляем итоговый список тикеров внизу
    if long_hits or short_hits:
        tickers_summary = []
        if long_hits:
            long_tickers = ", ".join(t for t, _ in long_hits)
            tickers_summary.append(f"*Лонг:* {long_tickers}")
        if short_hits:
            short_tickers = ", ".join(t for t, _ in short_hits)
            tickers_summary.append(f"*Шорт:* {short_tickers}")
        msg += "\n" + "\n".join(tickers_summary)

    await update.message.reply_text(msg, parse_mode="Markdown")


async def cross_ema20x50_4h(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🔍 Ищу пересечения EMA20 и EMA50 по 4H таймфрейму за последние 25 свечей...")
        print("▶ Запущена команда EMA CROSS")
        
        # Контроль времени выполнения
        start_time = datetime.now()
        MAX_EXECUTION_TIME = 1500  # 25 минут
        
        all_tickers = sum(SECTORS1.values(), [])
        print(f"🔁 Всего тикеров для обработки: {len(all_tickers)}")
        
        long_hits, short_hits = [], []
        processed_count = 0
        
        # Обрабатываем тикеры с ограничением по времени
        for ticker in all_tickers:
            # Проверка времени выполнения
            if (datetime.now() - start_time).seconds > MAX_EXECUTION_TIME:
                print(f"⏰ Превышено максимальное время выполнения ({MAX_EXECUTION_TIME} сек)")
                break
                
            try:
                print(f"🔁 Обрабатываем {ticker} ({processed_count + 1}/{len(all_tickers)})")
                
                # Принудительно сбрасываем буфер логов
                import sys
                sys.stdout.flush()
                
                # Добавляем timeout для каждого тикера
                print(f"📡 Запрашиваем данные для {ticker}...")
                
                # Оборачиваем ВСЮ обработку тикера в timeout
                ticker_result = await asyncio.wait_for(
                    process_single_ticker(ticker),
                    timeout=20.0  # 20 секунд на весь тикер
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
                
                # Отправляем промежуточное уведомление каждые 20 тикеров
                if processed_count % 20 == 0:
                    try:
                        progress_msg = f"⏳ Обработано {processed_count}/{len(all_tickers)} тикеров..."
                        await update.message.reply_text(progress_msg)
                        print(f"📱 Отправлено уведомление: {progress_msg}")
                    except Exception as progress_e:
                        print(f"❌ Ошибка отправки прогресса: {progress_e}")
                
                # Небольшая задержка между запросами + принудительный сброс буфера
                await asyncio.sleep(0.5)  # Увеличиваем задержку для API Tinkoff
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
        
        # Сортировка по дате (новые вверх)
        try:
            long_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y %H:%M'), reverse=True)
            short_hits.sort(key=lambda x: datetime.strptime(x[1], '%d.%m.%Y %H:%M'), reverse=True)
        except Exception as e:
            print(f"❌ Ошибка сортировки: {e}")
        
        # Ограничиваем количество результатов
        long_hits = long_hits[:30]
        short_hits = short_hits[:30]
        
        # Формируем сообщение
        execution_time = (datetime.now() - start_time).seconds
        msg = f"📊 *Анализ завершен* (обработано {processed_count} тикеров за {execution_time} сек)\n\n"
        
        if long_hits:
            msg += f"🟢 *Лонг пересечение EMA20×50 за последние 25 4Ч свечей, всего: {len(long_hits)}:*\n"
            msg += "\n".join(f"{t} {d}" for t, d in long_hits) + "\n\n"
        else:
            msg += "🟢 *Лонг сигналов не найдено за последние 25 4Ч свечей*\n\n"
            
        if short_hits:
            msg += f"🔴 *Шорт пересечение EMA20×50 за последние 25 4Ч свечей, всего: {len(short_hits)}:*\n\n"
            msg += "\n".join(f"{t} {d}" for t, d in short_hits)+ "\n\n"
        else:
            msg += "🔴 *Шорт сигналов не найдено за последние 25 4Ч свечей*\n\n"
        #msg += "\n"
        # Добавляем итоговый список тикеров внизу
        if long_hits or short_hits:
            tickers_summary = []
            if long_hits:
                long_tickers = ", ".join(t for t, _ in long_hits)
                tickers_summary.append(f"*Лонг:* {long_tickers}")
            if short_hits:
                short_tickers = ", ".join(t for t, _ in short_hits)
                tickers_summary.append(f"*Шорт:* {short_tickers}")
            msg += "\n" + "\n".join(tickers_summary)
        
        # Отправляем результат
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
    """
    Обрабатывает один тикер и возвращает найденные сигналы
    """
    try:
        # Получаем данные
        df = await asyncio.to_thread(get_moex_data_4h_tinkoff, ticker, 25)
        print(f"📊 Данные получены для {ticker}: {len(df) if not df.empty else 0} свечей")
        
        if df.empty:
            print(f"❌ Пустые данные для {ticker}")
            return None
            
        # Проверяем минимальное количество данных
        if len(df) < 50:
            print(f"❌ Недостаточно данных для {ticker}: {len(df)} свечей")
            return None
        
        print(f"🧮 Рассчитываем EMA для {ticker}...")
        # Рассчитываем EMA в отдельном потоке для избежания блокировки
        def calculate_ema(df):
            df_copy = df.copy()
            df_copy['EMA20'] = df_copy['close'].ewm(span=20, adjust=False).mean()
            df_copy['EMA50'] = df_copy['close'].ewm(span=50, adjust=False).mean()
            return df_copy
        
        df = await asyncio.to_thread(calculate_ema, df)
        
        print(f"🔍 Анализируем пересечения для {ticker}...")
        # Получаем данные за последние 26 свечей для анализа
        recent = df.tail(26)
        
        # Текущие значения
        current_close = df['close'].iloc[-1]
        current_ema20 = df['EMA20'].iloc[-1]
        current_ema50 = df['EMA50'].iloc[-1]
        
        long_signal = None
        short_signal = None
        
        # Проверяем пересечения за последний период
        for i in range(1, len(recent)):
            try:
                prev_ema20 = recent['EMA20'].iloc[i-1]
                prev_ema50 = recent['EMA50'].iloc[i-1]
                curr_ema20 = recent['EMA20'].iloc[i]
                curr_ema50 = recent['EMA50'].iloc[i]
                curr_close = recent['close'].iloc[i]
                
                # Получаем дату для текущего дня
                date = recent.index[i].strftime('%d.%m.%Y %H:%M')
                
                # Лонг пересечение: EMA20 пересекает EMA50 снизу вверх + подтверждение
                if (
                    prev_ema20 <= prev_ema50
                    and curr_ema20 > curr_ema50
                    and curr_close > curr_ema20
                    and current_close > current_ema20
                    and current_ema20 > current_ema50
                ):
                    long_signal = (ticker, date)
                    break
        
                # Шорт пересечение: EMA20 пересекает EMA50 сверху вниз + подтверждение
                elif (
                    prev_ema20 >= prev_ema50
                    and curr_ema20 < curr_ema50
                    and curr_close < curr_ema20
                    and current_close < current_ema20
                    and current_ema20 < current_ema50
                ):
                    short_signal = (ticker, date)
                    break
            except Exception as inner_e:
                print(f"❌ Ошибка при анализе пересечений для {ticker}: {inner_e}")
                continue
        
        return (long_signal, short_signal)
        
    except Exception as e:
        print(f"❌ Ошибка обработки тикера {ticker}: {e}")
        return None



async def receive_delta_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получает количество дней и выполняет расчет дельты"""
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


async def calculate_single_delta(update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str, days: int):
    """Рассчитывает дельту денежного потока для одной акции"""
    await update.message.reply_text(f"🔍 Рассчитываю дельту денежного потока для {ticker} за {days} дней...")
    
    try:
        df = get_moex_data(ticker, days=100)  # с запасом
        if df.empty or len(df) < days + 1:
            await update.message.reply_text(f"❌ Недостаточно данных для {ticker}. Попробуйте увеличить количество дней.")
            return

        
        df = df.rename(columns={'close': 'close', 'volume': 'volume'})
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

        # Условие для шорт сигнала EMA20x50
        ema20x50_short = (current_ema20 < current_ema50) and (current_price < current_ema20)

        # Изменение цены за день
        price_change_day = (current_price / df['close'].iloc[-2] - 1) if len(df) > 1 else 0

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
            delta_pct = 100 * ad_delta / avg_turnover
        else:
            delta_pct = 0

        # Формируем сообщение
        msg = f"📊 *Анализ дельты денежного потока для {ticker}*\n"
        msg += f"📅 *Период: {date_start} – {date_end} ({days} дней)*\n\n"
        
        # Добавляем предупреждение о низком обороте
        if filter_avg_turnover < 50_000_000:
            msg += "⚠️ *Внимание: низкий среднедневной оборот (< 50 млн ₽)*\n\n"

        # Иконки для сигналов
        if ema20x50_long:
            ema_icon = "🟢"
            ema_label = "Лонг"
        elif ema20x50_short:
            ema_icon = "🔴"
            ema_label = "Шорт"
        else:
            ema_icon = "⚫"
            ema_label = "Нет сигнала"

        sma_icon = "🟢" if price_above_sma30 else "🔴"
        flow_icon = "🟢" if ad_delta > 0 else "🔴"
        
        msg += f"*Δ Цены за период:* {price_pct:+.1f}%\n"
        msg += f"*Δ Потока:* {ad_delta/1_000_000:+.0f} млн ₽ {flow_icon}   *Δ / Оборот:* {delta_pct:.1f}%\n"
        #msg += f"*Δ / Оборот:* {delta_pct:.1f}%\n"
        msg += f"*Δ Цены 1D:* {price_change_day*100:+.1f}%   *Объём:* {ratio:.1f}x\n"
        #msg += f"*Объём:* {ratio:.1f}x\n"
        msg += f"*EMA20x50:* {ema_icon}   *SMA30:* {sma_icon}\n"
        #msg += f"*SMA30:* {sma_icon}\n"
        msg += "\n"
        
        # Добавляем интерпретацию результатов
        #if ad_delta > 0:
        #    msg += "Деньги приходят в акцию 🟢 \n"
        #else:
        #    msg += "Деньги уходят из акции 🔴\n"
        
        msg += f"💰 *Среднедневной оборот:* {avg_turnover/1_000_000:.1f} млн ₽\n"

        # Добавляем расшифровку сигналов
        #msg += f"EMA20x50: {ema_icon} ({ema_label})\n"
        #msg += f"SMA30 Weekly: {sma_icon} ({'Цена выше SMA30 1W' if price_above_sma30 else 'Цена ниже SMA30 1W'})"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при расчете дельты для {ticker}: {str(e)}")

# RSI TOP
async def rsi_top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда для показа топ 10 перекупленных и топ 10 перепроданных акций по RSI
    """
    await update.message.reply_text("🔍 Анализирую RSI всех акций. Это может занять некоторое время...")
    
    overbought_stocks = []  # RSI > 70
    oversold_stocks = []    # RSI < 30
    
    # Проходим по всем тикерам
    for ticker in sum(SECTORS.values(), []):
        try:
            # Получаем данные за последние 100 дней (с запасом для RSI)
            df = get_moex_data(ticker, days=100)
            if df.empty or len(df) < 15:  # Минимум 15 дней для корректного RSI
                continue
            
            # 💰 Среднедневной оборот за фиксированные 10 дней (для фильтра)
            filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
            filter_avg_turnover = filter_turnover_series.mean()
            
            # ❌ Фильтр по минимальному обороту: 50 млн руб за последние 10 дней
            if filter_avg_turnover < 50_000_000:
                continue
            
            # Вычисляем RSI
            rsi = compute_rsi(df['close'], window=14)
            if rsi.empty:
                continue
                
            # Берем последнее значение RSI
            current_rsi = rsi.iloc[-1]
            if pd.isna(current_rsi):
                continue
                
            # Текущая цена и изменение за день
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) >= 2 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0
            
            # Относительный объем (текущий объем к среднему за 10 дней)
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-10:].mean()
            relative_volume_pct = (current_volume / avg_volume * 100) if avg_volume != 0 else 100
            
            # Классификация по RSI
            if current_rsi >= 70:
                overbought_stocks.append((ticker, current_rsi, current_price, price_change_pct, relative_volume_pct))
            elif current_rsi <= 30:
                oversold_stocks.append((ticker, current_rsi, current_price, price_change_pct, relative_volume_pct))
                
        except Exception as e:
            logger.error(f"Ошибка при анализе RSI для {ticker}: {e}")
            continue
    
    # Сортируем списки
    overbought_stocks.sort(key=lambda x: x[1], reverse=True)  # По убыванию RSI
    oversold_stocks.sort(key=lambda x: x[1])                 # По возрастанию RSI
    
    # Формируем сообщение
    msg = f"📊 RSI анализ на {datetime.now().strftime('%d.%m.%Y %H:%M')}:\n\n"
    
    # 🔴 Перекупленные акции (RSI >= 70)
    if overbought_stocks:
        msg += "🔴 Топ 10 перекупленных акций (RSI ≥ 70):\n"
        msg += "<pre>\n"
        msg += f"{'Тикер':<6}  {'RSI':<4}  {'Цена':<8}  {'Изм %':<7}  {'Отн.об %':<8}\n"
        msg += f"{'─' * 6}  {'─' * 4}  {'─' * 8}  {'─' * 7}  {'─' * 8}\n"
        
        for ticker, rsi_val, price, price_change_pct, rel_volume in overbought_stocks[:10]:
            msg += f"{ticker:<6}  {rsi_val:4.0f}  {price:8.1f}  {price_change_pct:+6.1f}%  {rel_volume:7.0f}%\n"
        msg += "</pre>\n\n"
    else:
        msg += "🔴 Перекупленных акций (RSI ≥ 70) не найдено\n\n"
    
    # 🟢 Перепроданные акции (RSI <= 30)
    if oversold_stocks:
        msg += "🟢 Топ 10 перепроданных акций (RSI ≤ 30):\n"
        msg += "<pre>\n"
        msg += f"{'Тикер':<6}  {'RSI':<4}  {'Цена':<8}  {'Изм %':<7}  {'Отн.об %':<8}\n"
        msg += f"{'─' * 6}  {'─' * 4}  {'─' * 8}  {'─' * 7}  {'─' * 8}\n"
        
        for ticker, rsi_val, price, price_change_pct, rel_volume in oversold_stocks[:10]:
            msg += f"{ticker:<6}  {rsi_val:4.0f}  {price:8.1f}  {price_change_pct:+6.1f}%  {rel_volume:7.0f}%\n"
        msg += "</pre>\n\n"
    else:
        msg += "🟢 Перепроданных акций (RSI ≤ 30) не найдено\n\n"
    
    # Статистика
    total_analyzed = len(overbought_stocks) + len(oversold_stocks)
    msg += f"📈 Статистика:\n"
    msg += f"• Всего акций в зонах экстремума: {total_analyzed}\n"
    msg += f"• Перекупленных: {len(overbought_stocks)}\n"
    msg += f"• Перепроданных: {len(oversold_stocks)}\n"
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


# Получение данных для Штейн
def get_moex_weekly_data(ticker="SBER", weeks=80):
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


# Telegram команды
if Update and ContextTypes:

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
    
    async def stan_recent_week(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔍 Ищу акции с недавним long пересечением цены через SMA30...")
        
        crossovers = []
        all_tickers = sum(SECTORS.values(), [])
        
        # Проверяем каждый тикер
        for ticker in all_tickers:
            try:
                crossover_date = find_sma30_crossover_week(ticker, weeks=5)
                if crossover_date:
                    crossovers.append((ticker, crossover_date))
            except Exception as e:
                print(f"Ошибка при анализе {ticker}: {e}")
                continue
        
        if not crossovers:
            await update.message.reply_text("📊 За последние 5 недель не найдено акций с пересечением цены через SMA30 снизу вверх.")
            return
        
        # Сортируем по дате (от самого свежего к самому старому)
        crossovers.sort(key=lambda x: x[1], reverse=True)
        
        # Формируем результат
        result_text = "📈 Акции с пересечением цены через SMA30 снизу вверх за последние 5 недель:\n\n"
        
        for ticker, date in crossovers:
            formatted_date = date.strftime('%d.%m.%Y')
            result_text += f"{ticker} {formatted_date}\n"
        
        result_text += f"\n🔢 Всего найдено: {len(crossovers)} акций"
        
        await update.message.reply_text(result_text)

        # === ИНТЕГРАЦИЯ КЭШИРОВАНИЯ ===
    try:
        import caching
        print("✅ Модуль кэширования загружен успешно")
        
        # Явная активация
        if hasattr(caching, 'activate_caching_if_enabled'):
            success = caching.activate_caching_if_enabled()
            if success:
                print("🎯 Кэширование активировано")
            else:
                print("⚠️ Кэширование не активировано")
    
    except ImportError:
        print("ℹ️ Модуль кэширования не найден, работаем без кэша")


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

    # === Добавляем хендлеры ===
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chart_hv", chart_hv))
    app.add_handler(CommandHandler("cross_ema20x50", cross_ema20x50))
    app.add_handler(CommandHandler("cross_ema20x50_4h", cross_ema20x50_4h))
    app.add_handler(CommandHandler("stan", stan))
    app.add_handler(CommandHandler("stan_recent", stan_recent))
    app.add_handler(CommandHandler("stan_recent_d_short", stan_recent_d_short))
    app.add_handler(CommandHandler("stan_recent_week", stan_recent_week))
    app.add_handler(CommandHandler("long_moneyflow", long_moneyflow))
    app.add_handler(CommandHandler("high_volume", high_volume))
    app.add_handler(CommandHandler("rsi_top", rsi_top))
    app.add_handler(CommandHandler("cache_debug", cache_debug))
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
