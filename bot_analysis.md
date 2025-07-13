# Анализ MOEX Telegram-бота

## Общая информация

**Название**: MOEX Stock Bot  
**Размер кода**: 1814 строк (85KB)  
**Платформа развертывания**: Render (render.com)  
**Команда разработки**: @TradeAnsh  

## Архитектура и технологии

### Основные технологии
- **Python 3.x** - основной язык разработки
- **python-telegram-bot v20.7** - работа с Telegram API
- **pandas, numpy** - обработка и анализ данных
- **matplotlib** - создание графиков
- **scipy** - математические вычисления
- **tinkoff-investments** - получение данных от Tinkoff API
- **flask** - веб-сервер для webhook'ов
- **requests** - HTTP-запросы к MOEX API

### Структура развертывания
```yaml
# render.yaml
services:
  - type: web
    name: moex-bot
    env: python
    startCommand: "python main.py"
    plan: free
```

## Функциональность бота

### Основные команды

1. **`/start`** - Приветствие и список команд
2. **`/chart_hv`** - Выбор акций через секторальные кнопки
3. **`/stan`** - Анализ по методу Стэна Вайнштейна
4. **`/cross_ema20x50`** - Поиск пересечений EMA 20x50 (1D)
5. **`/cross_ema20x50_4h`** - Поиск пересечений EMA 20x50 (4H)
6. **`/stan_recent`** - Акции с лонг пересечением SMA30 (1D)
7. **`/stan_recent_short`** - Акции с шорт пересечением SMA30 (1D)
8. **`/stan_recent_week`** - Акции с лонг пересечением SMA30 (1W)
9. **`/moneyflow`** - Анализ денежного потока (Money A/D)
10. **`/high_volume`** - Акции с повышенным объемом
11. **`/delta`** - Расчет дельты денежного потока
12. **`/rsi_top`** - Топ по RSI (перекупленность/перепроданность)

### Покрытие акций

**Всего акций**: ~150 российских акций, разбитых на 12 секторов:
- Финансы (15 акций)
- Нефтегаз (11 акций)
- Металлы и добыча (13 акций)
- IT (13 акций)
- Телеком (4 акции)
- Строители (3 акции)
- Ритейл (11 акций)
- Электро (10 акций)
- Транспорт и логистика (5 акций)
- Агро (5 акций)
- Медицина (5 акций)
- Машиностроение (4 акции)

## Технический анализ

### Индикаторы
- **RSI (14)** - Relative Strength Index
- **EMA (9, 20, 50, 100, 200)** - Exponential Moving Average
- **SMA (30)** - Simple Moving Average
- **Money A/D** - Accumulation/Distribution Line
- **Volume Analysis** - Анализ объемов торгов

### Анализ графиков
- Построение candlestick charts
- Определение уровней поддержки/сопротивления
- Поиск паттернов "двойная вершина/дно"
- Выделение аномальных объемов
- Наложение технических индикаторов

### Источники данных

1. **MOEX API** - основной источник данных
   - Дневные данные (24H интервал)
   - 4-часовые данные (4H интервал)
   - Недельные данные

2. **Tinkoff Investments API** - дополнительные данные
   - Получение FIGI кодов
   - Альтернативные данные по акциям

## Архитектурные решения

### Положительные аспекты

1. **Webhook Architecture**: Использование webhook'ов для стабильной работы
2. **Асинхронность**: Правильное использование async/await
3. **Модульность**: Разделение функций по назначению
4. **Обработка ошибок**: try/except блоки в критических местах
5. **Фильтрация данных**: Минимальные требования к обороту (50 млн руб.)

### Проблемы архитектуры

1. **Монолитность**: Весь код в одном файле (1814 строк)
2. **Дублирование**: Похожие функции `get_moex_data` и `get_moex_data_4h`
3. **Смешанная логика**: Telegram handlers и бизнес-логика в одном месте
4. **Жестко закодированные данные**: Списки акций в коде
5. **Отсутствие кэширования**: Каждый запрос идет к API
6. **Нет логирования**: Отсутствует система логирования

## Производительность и масштабирование

### Текущие ограничения

1. **Rate Limiting**: Нет контроля частоты запросов к API
2. **Memory Usage**: Нет оптимизации использования памяти
3. **Concurrent Requests**: Нет ограничений на одновременные запросы
4. **Data Persistence**: Отсутствует сохранение данных между запросами

### Производительность функций

```python
# Пример неэффективного кода
async def all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for ticker in sum(SECTORS.values(), []):  # ~150 акций
        try:
            df = get_moex_data(ticker)  # Запрос к API
            # ... обработка ...
```

## Безопасность

### Положительные аспекты
- Использование переменных окружения для токенов
- Validation пользовательского ввода в некоторых местах

### Уязвимости
- Отсутствие rate limiting для пользователей
- Нет авторизации/аутентификации
- Отсутствует логирование действий пользователей
- Нет защиты от DOS атак

## Рекомендации по улучшению

### 1. Рефакторинг архитектуры

```python
# Предлагаемая структура
project/
├── main.py                 # Точка входа
├── bot/
│   ├── __init__.py
│   ├── handlers/           # Telegram handlers
│   ├── services/           # Бизнес-логика
│   └── utils/             # Утилиты
├── data/
│   ├── moex_client.py     # MOEX API client
│   ├── tinkoff_client.py  # Tinkoff API client
│   └── cache.py           # Кэширование
├── analysis/
│   ├── indicators.py      # Технические индикаторы
│   ├── patterns.py        # Паттерны
│   └── charts.py          # Графики
└── config/
    ├── settings.py        # Конфигурация
    └── sectors.py         # Секторы и акции
```

### 2. Кэширование данных

```python
import redis
from datetime import timedelta

class DataCache:
    def __init__(self):
        self.redis = redis.Redis()
    
    def get_market_data(self, ticker: str, interval: str):
        key = f"market_data:{ticker}:{interval}"
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        
        # Получение данных...
        data = fetch_from_api(ticker, interval)
        
        # Кэширование на 5 минут
        self.redis.setex(key, 300, json.dumps(data))
        return data
```

### 3. Обработка ошибок

```python
import logging
from functools import wraps

def handle_errors(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            return await func(update, context)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            await update.message.reply_text("❌ Произошла ошибка. Попробуйте позже.")
    return wrapper
```

### 4. Rate Limiting

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
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
```

### 5. Конфигурация

```python
# config/settings.py
import os
from dataclasses import dataclass

@dataclass
class Settings:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN")
    tinkoff_token: str = os.getenv("TINKOFF_API_TOKEN")
    webhook_url: str = os.getenv("WEBHOOK_URL")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_requests_per_minute: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))
```

### 6. Тестирование

```python
# tests/test_analysis.py
import pytest
from analysis.indicators import compute_rsi

def test_rsi_calculation():
    # Тестовые данные
    prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89]
    
    result = compute_rsi(pd.Series(prices))
    
    assert len(result) == len(prices)
    assert not result.isna().all()
    assert 0 <= result.iloc[-1] <= 100
```

## Мониторинг и метрики

### Предлагаемые метрики

1. **Производительность**
   - Время ответа API
   - Количество запросов в секунду
   - Использование памяти

2. **Пользовательские**
   - Количество активных пользователей
   - Популярные команды
   - Частота ошибок

3. **Бизнес-метрики**
   - Самые популярные акции
   - Время использования бота
   - Конверсия пользователей

### Система логирования

```python
import logging
import structlog

# Структурированное логирование
logger = structlog.get_logger()

async def track_command_usage(update: Update, command: str):
    logger.info(
        "command_executed",
        user_id=update.effective_user.id,
        username=update.effective_user.username,
        command=command,
        timestamp=datetime.now().isoformat()
    )
```

## Итоговая оценка

### Сильные стороны (8/10)
- ✅ Богатая функциональность
- ✅ Правильное использование технических индикаторов
- ✅ Хорошее покрытие российского рынка
- ✅ Стабильная работа через webhook'и
- ✅ Интуитивный интерфейс

### Области для улучшения (6/10)
- ❌ Монолитная архитектура
- ❌ Отсутствие кэширования
- ❌ Нет системы логирования
- ❌ Слабая обработка ошибок
- ❌ Нет rate limiting

### Общая оценка: 7/10

Бот представляет собой функциональный и полезный инструмент для анализа российского фондового рынка. Основные алгоритмы работают корректно, покрытие рынка хорошее. Однако требуется серьезный рефакторинг архитектуры для улучшения производительности, надежности и масштабируемости.

## Приоритетные шаги по улучшению

1. **Высокий приоритет**
   - Разделение кода на модули
   - Добавление кэширования
   - Система логирования

2. **Средний приоритет**
   - Rate limiting
   - Обработка ошибок
   - Тестирование

3. **Низкий приоритет**
   - Мониторинг метрик
   - Оптимизация производительности
   - Расширение функциональности