# Практическое руководство по миграции на модули

## Текущая структура проекта

```
workspace/
├── main.py (1814 строк)
├── requirements.txt
├── render.yaml
├── config/
│   ├── __init__.py
│   ├── settings.py           ✅ Создано
│   └── sectors.py            ✅ Создано
├── data/
│   ├── __init__.py
│   ├── moex_client.py        ✅ Создано
│   └── tinkoff_client.py     (нужно создать)
├── analysis/
│   ├── __init__.py
│   ├── indicators.py         ✅ Создано
│   ├── patterns.py           ✅ Создано
│   └── charts.py             ✅ Создано
├── bot/
│   ├── __init__.py
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── basic.py          ✅ Создано
│   │   ├── analysis.py       (нужно создать)
│   │   ├── trading.py        (нужно создать)
│   │   └── callbacks.py      (нужно создать)
│   ├── services/
│   │   ├── __init__.py
│   │   └── market_scanner.py (нужно создать)
│   └── utils/
│       ├── __init__.py
│       └── decorators.py     (нужно создать)
├── tests/
│   └── __init__.py
├── main_refactored.py        ✅ Создано (пример)
└── refactoring_guide.md      ✅ Создано
```

## Шаг 1: Тестирование созданных модулей

### Тест конфигурации
```bash
python3 -c "from config.settings import settings; print(f'Token: {settings.telegram_token[:10]}...')"
python3 -c "from config.sectors import get_all_tickers; print(f'Всего тикеров: {len(get_all_tickers())}')"
```

### Тест MOEX клиента
```bash
python3 -c "
from data.moex_client import moex_client
import pandas as pd
df = moex_client.get_daily_data('SBER', 10)
print(f'Данные по SBER: {len(df)} строк')
print(df.head())
"
```

### Тест индикаторов
```bash
python3 -c "
from data.moex_client import moex_client
from analysis.indicators import analyze_indicators
df = moex_client.get_daily_data('SBER', 50)
df = analyze_indicators(df)
print(f'RSI: {df["RSI"].iloc[-1]:.1f}')
print(f'EMA20: {df["EMA20"].iloc[-1]:.2f}')
"
```

## Шаг 2: Создание недостающих модулей

### data/tinkoff_client.py
```python
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TinkoffClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TINKOFF_API_TOKEN")
        if not self.token:
            logger.warning("Tinkoff API token not provided")
    
    def get_figi_by_ticker(self, ticker: str) -> Optional[str]:
        # Перенести логику из main.py
        return None

# Глобальный экземпляр
tinkoff_client = TinkoffClient()
```

### bot/handlers/analysis.py
```python
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from config.sectors import SECTORS
import logging

logger = logging.getLogger(__name__)

async def chart_hv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /chart_hv"""
    keyboard = [
        [InlineKeyboardButton(sector, callback_data=f"sector:{sector}:0")] 
        for sector in SECTORS
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите отрасль:", reply_markup=reply_markup)

async def stan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /stan"""
    # Логика из main.py
    pass

async def rsi_top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /rsi_top"""
    # Логика из main.py
    pass
```

### bot/services/market_scanner.py
```python
from typing import List, Tuple
from data.moex_client import moex_client
from analysis.indicators import analyze_indicators
from config.sectors import get_all_tickers

class MarketScanner:
    """Сканер рынка для поиска торговых сигналов"""
    
    async def scan_rsi_extremes(self) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Поиск перекупленных и перепроданных акций"""
        overbought = []
        oversold = []
        
        for ticker in get_all_tickers():
            try:
                df = moex_client.get_daily_data(ticker, 30)
                if df.empty:
                    continue
                
                df = analyze_indicators(df)
                rsi = df['RSI'].iloc[-1]
                
                if rsi > 70:
                    overbought.append((ticker, rsi))
                elif rsi < 30:
                    oversold.append((ticker, rsi))
                    
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
        
        # Сортировка по RSI
        overbought.sort(key=lambda x: x[1], reverse=True)
        oversold.sort(key=lambda x: x[1])
        
        return overbought, oversold
```

## Шаг 3: Постепенная миграция

### Вариант 1: Постепенная замена

1. **Создайте main_new.py** - копию оригинального main.py
2. **Начните замену импортов** по одному:
   ```python
   # Было:
   SECTORS = {...}
   
   # Стало:
   from config.sectors import SECTORS
   ```

3. **Заменяйте функции** одну за другой:
   ```python
   # Было:
   def get_moex_data(ticker, days):
       # ... 50 строк кода
   
   # Стало:
   from data.moex_client import moex_client
   # используйте moex_client.get_daily_data()
   ```

4. **Тестируйте** каждое изменение

### Вариант 2: Создание нового проекта

1. **Создайте новую папку** `moex_bot_v2/`
2. **Скопируйте модули** из текущего проекта
3. **Создайте новый main.py** с нуля
4. **Тестируйте** весь функционал
5. **Замените** старый проект новым

## Шаг 4: Обновление зависимостей

### requirements.txt
```
python-telegram-bot[webhooks]==20.7
pandas
numpy
matplotlib
scipy
requests
flask
tinkoff-investments
redis  # для кэширования
pytest  # для тестирования
```

## Шаг 5: Тестирование

### Создание tests/test_basic.py
```python
import pytest
from unittest.mock import Mock, patch
from bot.handlers.basic import start

@pytest.mark.asyncio
async def test_start_command():
    update = Mock()
    context = Mock()
    
    await start(update, context)
    
    update.message.reply_text.assert_called_once()
    args = update.message.reply_text.call_args[0][0]
    assert "Привет!" in args
    assert "/start" in args
```

### Запуск тестов
```bash
pytest tests/ -v
```

## Шаг 6: Развертывание

### Обновление render.yaml
```yaml
services:
  - type: web
    name: moex-bot
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python main.py"  # или main_refactored.py
    plan: free
    envVars:
      - key: TELEGRAM_TOKEN
        sync: false
      - key: TINKOFF_API_TOKEN
        sync: false
      - key: LOG_LEVEL
        value: INFO
```

## Шаг 7: Мониторинг и отладка

### Добавление логирования
```python
import logging
import structlog

# Структурированные логи
logger = structlog.get_logger()

# В каждой функции
logger.info("Processing request", ticker=ticker, user_id=user_id)
```

### Отслеживание ошибок
```python
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[LoggingIntegration(level=logging.INFO)]
)
```

## Контрольный список миграции

- [ ] Создана структура папок
- [ ] Созданы базовые модули (config, data, analysis)
- [ ] Протестированы модули работы с данными
- [ ] Созданы обработчики команд
- [ ] Создан новый main.py
- [ ] Протестированы основные команды
- [ ] Добавлены тесты
- [ ] Обновлены зависимости
- [ ] Протестировано развертывание
- [ ] Добавлен мониторинг
- [ ] Создана документация

## Полезные команды

### Проверка импортов
```bash
python3 -c "import sys; sys.path.append('.'); from config.settings import settings; print('OK')"
```

### Проверка структуры
```bash
find . -name "*.py" -exec python3 -m py_compile {} \;
```

### Запуск с дебагом
```bash
export LOG_LEVEL=DEBUG
python3 main_refactored.py
```

## Преимущества после миграции

1. **Читаемость**: Легче понимать код
2. **Поддержка**: Быстрее находить и исправлять ошибки
3. **Тестирование**: Можно тестировать каждый модуль отдельно
4. **Расширяемость**: Легко добавлять новые функции
5. **Переиспользование**: Можно использовать модули в других проектах
6. **Производительность**: Возможность добавить кэширование
7. **Мониторинг**: Лучше отслеживать работу бота

Такая структура сделает ваш проект более профессиональным и готовым к масштабированию!