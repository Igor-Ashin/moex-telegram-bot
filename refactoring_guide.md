# Руководство по рефакторингу MOEX-бота на модули

## Предлагаемая структура проекта

```
moex_bot/
├── main.py                     # Точка входа
├── requirements.txt            # Зависимости
├── render.yaml                 # Конфигурация развертывания
├── config/
│   ├── __init__.py
│   ├── settings.py             # Настройки приложения
│   └── sectors.py              # Секторы и акции
├── data/
│   ├── __init__.py
│   ├── moex_client.py          # Клиент MOEX API
│   ├── tinkoff_client.py       # Клиент Tinkoff API
│   └── cache.py                # Кэширование данных
├── analysis/
│   ├── __init__.py
│   ├── indicators.py           # Технические индикаторы
│   ├── patterns.py             # Паттерны анализа
│   └── charts.py               # Построение графиков
├── bot/
│   ├── __init__.py
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── basic.py            # Базовые команды (/start, /help)
│   │   ├── analysis.py         # Команды анализа (/stan, /rsi_top)
│   │   ├── trading.py          # Торговые сигналы (/cross_ema, /moneyflow)
│   │   └── callbacks.py        # Обработчики callback'ов
│   ├── services/
│   │   ├── __init__.py
│   │   ├── market_scanner.py   # Сканирование рынка
│   │   └── notifications.py    # Уведомления
│   └── utils/
│       ├── __init__.py
│       ├── decorators.py       # Декораторы (rate limiting, error handling)
│       ├── validators.py       # Валидация данных
│       └── formatters.py       # Форматирование сообщений
└── tests/
    ├── __init__.py
    ├── test_indicators.py
    ├── test_moex_client.py
    └── test_handlers.py
```

## Шаг 1: Конфигурация

### config/settings.py
```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    # Telegram
    telegram_token: str = os.getenv("TELEGRAM_TOKEN")
    webhook_url: str = os.getenv("WEBHOOK_URL", "https://moex-telegram-bot-sra8.onrender.com")
    
    # API Keys
    tinkoff_token: str = os.getenv("TINKOFF_API_TOKEN")
    
    # Rate Limiting
    max_requests_per_minute: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))
    
    # Cache
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL", "300"))  # 5 минут
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Filters
    min_turnover: float = float(os.getenv("MIN_TURNOVER", "50000000"))  # 50 млн
    
    def __post_init__(self):
        if not self.telegram_token:
            raise ValueError("TELEGRAM_TOKEN is required")

# Глобальная конфигурация
settings = Settings()
```

### config/sectors.py
```python
# Перенесено из main.py
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

# Упрощенная версия для некоторых функций
SECTORS_SIMPLIFIED = {
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

def get_all_tickers() -> list[str]:
    """Возвращает все тикеры из всех секторов"""
    return sum(SECTORS.values(), [])

def get_sector_tickers(sector: str) -> list[str]:
    """Возвращает тикеры для конкретного сектора"""
    return SECTORS.get(sector, [])
```

## Шаг 2: Работа с данными

### data/moex_client.py
```python
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
        """Получение дневных данных"""
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
        """Получение 4-часовых данных"""
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
        """Получение недельных данных"""
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
            return self._process_candles_data(data, weeks)
            
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
```

### data/tinkoff_client.py
```python
import os
from typing import Optional
import logging
from tinkoff.invest import Client, CandleInterval
import pandas as pd

logger = logging.getLogger(__name__)

class TinkoffClient:
    """Клиент для работы с Tinkoff API"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TINKOFF_API_TOKEN")
        if not self.token:
            logger.warning("Tinkoff API token not provided")
    
    def get_figi_by_ticker(self, ticker: str) -> Optional[str]:
        """Получение FIGI по тикеру"""
        if not self.token:
            return None
            
        try:
            with Client(self.token) as client:
                instruments = client.instruments.shares()
                for instrument in instruments.instruments:
                    if instrument.ticker == ticker:
                        return instrument.figi
            return None
        except Exception as e:
            logger.error(f"Error getting FIGI for {ticker}: {e}")
            return None
    
    def get_4h_data(self, ticker: str, days: int = 25) -> pd.DataFrame:
        """Получение 4-часовых данных через Tinkoff API"""
        if not self.token:
            return pd.DataFrame()
            
        try:
            figi = self.get_figi_by_ticker(ticker)
            if not figi:
                return pd.DataFrame()
                
            with Client(self.token) as client:
                from datetime import datetime, timedelta
                
                now = datetime.now()
                from_date = now - timedelta(days=days)
                
                candles = client.get_all_candles(
                    figi=figi,
                    from_=from_date,
                    to=now,
                    interval=CandleInterval.CANDLE_INTERVAL_4_HOUR
                )
                
                data = []
                for candle in candles:
                    data.append({
                        'begin': candle.time,
                        'close': candle.close.units + candle.close.nano / 1e9,
                        'high': candle.high.units + candle.high.nano / 1e9,
                        'low': candle.low.units + candle.low.nano / 1e9,
                        'volume': candle.volume
                    })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    df.set_index('begin', inplace=True)
                    
                return df
                
        except Exception as e:
            logger.error(f"Error getting 4h data from Tinkoff for {ticker}: {e}")
            return pd.DataFrame()

# Глобальный экземпляр клиента
tinkoff_client = TinkoffClient()
```

## Шаг 3: Технические индикаторы

### analysis/indicators.py
```python
import pandas as pd
import numpy as np
from typing import Optional

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисление RSI
    Перенесено из main.py
    """
    if len(series) < window + 1:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)
    
    # Используем ewm с alpha = 1/window для сглаживания Wilder's
    alpha = 1.0 / window
    avg_gain = gain_series.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss_series.ewm(alpha=alpha, adjust=False).mean()
    
    # Вычисляем RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Заменяем бесконечные значения на NaN
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.round(0)

def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Вычисление EMA"""
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """Вычисление SMA"""
    return df['close'].rolling(window=period).mean()

def calculate_money_ad(df: pd.DataFrame) -> pd.Series:
    """
    Вычисление Money Flow (A/D Line)
    Перенесено из main.py
    """
    if df.empty or len(df) < 2:
        return pd.Series([], dtype=float)
    
    # Money Flow Multiplier
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    
    # Заменяем NaN на 0 (когда high == low)
    mf_multiplier = mf_multiplier.fillna(0)
    
    # Money Flow Volume
    mf_volume = mf_multiplier * df['volume']
    
    # Accumulation/Distribution Line
    ad_line = mf_volume.cumsum()
    
    return ad_line

def analyze_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление всех индикаторов к DataFrame
    Перенесено из main.py
    """
    if df.empty:
        return df
    
    # RSI
    df['RSI'] = compute_rsi(df['close'], window=14)
    
    # Volume analysis
    df['Volume_Mean'] = df['volume'].rolling(window=10).mean()
    df['Anomaly'] = df['volume'] > 1.5 * df['Volume_Mean']
    df['Volume_Multiplier'] = df['volume'] / df['Volume_Mean']
    
    # EMA
    df['EMA9'] = calculate_ema(df, 9)
    df['EMA20'] = calculate_ema(df, 20)
    df['EMA50'] = calculate_ema(df, 50)
    df['EMA100'] = calculate_ema(df, 100)
    df['EMA200'] = calculate_ema(df, 200)
    
    # SMA
    df['SMA30'] = calculate_sma(df, 30)
    
    # Money Flow
    df['Money_AD'] = calculate_money_ad(df)
    
    return df

def find_ema_crossover(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> Optional[str]:
    """Поиск пересечения EMA"""
    if df.empty or len(df) < max(fast_period, slow_period) + 1:
        return None
    
    fast_ema = calculate_ema(df, fast_period)
    slow_ema = calculate_ema(df, slow_period)
    
    # Проверяем последние 2 точки
    if len(fast_ema) < 2 or len(slow_ema) < 2:
        return None
    
    # Текущее положение
    current_fast = fast_ema.iloc[-1]
    current_slow = slow_ema.iloc[-1]
    
    # Предыдущее положение
    prev_fast = fast_ema.iloc[-2]
    prev_slow = slow_ema.iloc[-2]
    
    # Пересечение снизу вверх (бычий сигнал)
    if prev_fast <= prev_slow and current_fast > current_slow:
        return "bullish"
    
    # Пересечение сверху вниз (медвежий сигнал)
    if prev_fast >= prev_slow and current_fast < current_slow:
        return "bearish"
    
    return None
```

## Шаг 4: Обработчики команд

### bot/handlers/basic.py
```python
from telegram import Update
from telegram.ext import ContextTypes
import logging

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    text = (
        "Привет! Это бот от команды @TradeAnsh для анализа акций Мосбиржи.\n"
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
    
    # Логирование
    logger.info(f"User {update.effective_user.id} started the bot")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    await start(update, context)
```

### bot/handlers/analysis.py
```python
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from config.sectors import SECTORS
from data.moex_client import moex_client
from analysis.indicators import analyze_indicators
from analysis.charts import plot_stock_chart
from analysis.patterns import find_levels, detect_double_patterns
import logging

logger = logging.getLogger(__name__)

async def chart_hv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /chart_hv - выбор акций через кнопки"""
    keyboard = [
        [InlineKeyboardButton(sector, callback_data=f"sector:{sector}:0")] 
        for sector in SECTORS
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите отрасль:", reply_markup=reply_markup)
    
    logger.info(f"User {update.effective_user.id} requested chart_hv")

async def stan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /stan - анализ по методу Стэна Вайнштейна"""
    keyboard = [
        [InlineKeyboardButton(sector, callback_data=f"stan_sector:{sector}:0")] 
        for sector in SECTORS
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Выберите отрасль для анализа по Стэну Вайнштейну:", 
        reply_markup=reply_markup
    )
    
    logger.info(f"User {update.effective_user.id} requested stan analysis")

async def rsi_top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /rsi_top - топ по RSI"""
    await update.message.reply_text("🔄 Анализирую RSI для всех акций...")
    
    try:
        from bot.services.market_scanner import MarketScanner
        scanner = MarketScanner()
        
        overbought, oversold = await scanner.scan_rsi_extremes()
        
        # Формируем сообщение
        message = "📊 **RSI Анализ**\n\n"
        
        if overbought:
            message += "🔴 **Перекупленные акции (RSI > 70):**\n"
            for ticker, rsi in overbought[:10]:
                message += f"• {ticker}: {rsi:.0f}\n"
        
        if oversold:
            message += "\n🟢 **Перепроданные акции (RSI < 30):**\n"
            for ticker, rsi in oversold[:10]:
                message += f"• {ticker}: {rsi:.0f}\n"
        
        if not overbought and not oversold:
            message += "ℹ️ Нет акций в экстремальных зонах RSI"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Error in rsi_top: {e}")
        await update.message.reply_text("❌ Ошибка при анализе RSI")
```

## Шаг 5: Утилиты

### bot/utils/decorators.py
```python
import functools
import logging
from typing import Callable, Dict, List
from collections import defaultdict
import time
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[int, List[float]] = defaultdict(list)
    
    def is_allowed(self, user_id: int) -> bool:
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Удаляем старые запросы
        self.requests[user_id] = [
            req_time for req_time in user_requests 
            if now - req_time < self.window
        ]
        
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        self.requests[user_id].append(now)
        return True

# Глобальный rate limiter
rate_limiter = RateLimiter()

def rate_limit(func: Callable) -> Callable:
    """Декоратор для ограничения частоты запросов"""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        
        if not rate_limiter.is_allowed(user_id):
            await update.message.reply_text(
                "⏳ Слишком много запросов. Попробуйте через минуту."
            )
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
            await update.message.reply_text(
                "❌ Произошла ошибка. Попробуйте позже или обратитесь к администратору."
            )
    
    return wrapper

def log_command(func: Callable) -> Callable:
    """Декоратор для логирования команд"""
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        command = func.__name__
        
        logger.info(f"Command '{command}' executed by user {user_id} ({username})")
        
        result = await func(update, context, *args, **kwargs)
        
        logger.info(f"Command '{command}' completed for user {user_id}")
        return result
    
    return wrapper

# Комбинированный декоратор
def telegram_handler(func: Callable) -> Callable:
    """Комбинированный декоратор для обработчиков команд"""
    return log_command(handle_errors(rate_limit(func)))
```

## Шаг 6: Новый main.py

### main.py
```python
import os
import logging
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ConversationHandler, MessageHandler, filters

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импорты обработчиков
from bot.handlers.basic import start, help_command
from bot.handlers.analysis import chart_hv, stan, rsi_top
from bot.handlers.trading import cross_ema20x50, cross_ema20x50_4h, high_volume, moneyflow
from bot.handlers.callbacks import handle_callback
from bot.handlers.conversations import delta_conversation, moneyflow_conversation

# Конфигурация
from config.settings import settings

def setup_handlers(app):
    """Настройка обработчиков команд"""
    
    # Основные команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("chart_hv", chart_hv))
    app.add_handler(CommandHandler("stan", stan))
    app.add_handler(CommandHandler("rsi_top", rsi_top))
    
    # Торговые сигналы
    app.add_handler(CommandHandler("cross_ema20x50", cross_ema20x50))
    app.add_handler(CommandHandler("cross_ema20x50_4h", cross_ema20x50_4h))
    app.add_handler(CommandHandler("high_volume", high_volume))
    
    # Callback обработчики
    app.add_handler(CallbackQueryHandler(handle_callback))
    
    # Диалоги
    app.add_handler(delta_conversation)
    app.add_handler(moneyflow_conversation)
    
    logger.info("All handlers registered successfully")

def main():
    """Основная функция запуска бота"""
    
    if not settings.telegram_token:
        logger.error("TELEGRAM_TOKEN not found in environment variables")
        return
    
    # Создаем приложение
    app = ApplicationBuilder().token(settings.telegram_token).build()
    
    # Настраиваем обработчики
    setup_handlers(app)
    
    # Запуск через webhook
    logger.info("Starting bot with webhook...")
    
    app.run_webhook(
        listen="0.0.0.0",
        port=8080,
        url_path=settings.telegram_token,
        webhook_url=f"{settings.webhook_url}/{settings.telegram_token}"
    )

if __name__ == '__main__':
    main()
```

## Шаг 7: Миграция

### Порядок миграции:

1. **Создайте новую структуру папок**
2. **Скопируйте код по модулям** (начните с config и data)
3. **Протестируйте каждый модуль** отдельно
4. **Постепенно переносите обработчики**
5. **Обновите main.py**
6. **Протестируйте всю систему**

### Команды для миграции:

```bash
# Создание структуры
mkdir -p config data analysis bot/{handlers,services,utils} tests

# Создание __init__.py файлов
touch config/__init__.py data/__init__.py analysis/__init__.py
touch bot/__init__.py bot/handlers/__init__.py bot/services/__init__.py bot/utils/__init__.py

# Бэкап оригинального файла
cp main.py main_backup.py
```

## Преимущества новой структуры:

1. **Читаемость**: Каждый модуль отвечает за свою область
2. **Тестируемость**: Легко писать unit-тесты
3. **Масштабируемость**: Легко добавлять новые функции
4. **Переиспользование**: Код можно использовать в других проектах
5. **Поддержка**: Легче находить и исправлять ошибки

Такая структура сделает ваш код более профессиональным и удобным для разработки!