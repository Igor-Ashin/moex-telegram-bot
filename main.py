# moex_stock_bot.py

import matplotlib
matplotlib.use('Agg')  # Включаем "безголовый" режим для matplotlib
import requests
import pandas as pd
import numpy as np
import os  # ← должен быть на самом верху, на уровне других импортов
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import argrelextrema
import asyncio

# Заменяем telegram на условный заглушку или комментарий, чтобы избежать ошибки в окружении
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
except ModuleNotFoundError:
    print("Библиотека 'python-telegram-bot' не установлена.")
    Update = None
    ApplicationBuilder = None
    CommandHandler = None
    CallbackQueryHandler = None
    ContextTypes = None

SECTORS = {
    "Финансы": ["SBER", "T", "VTBR", "MOEX", "SPBE", "RENI", "BSPB", "SVCB", "MBNK", "LEAS", "SFIN", "AFKS", "CARM", "ZAYM", "CBOM"],
    "Нефтегаз": ["GAZP", "NVTK", "LKOH", "ROSN", "TATNP", "TATN", "SNGS", "SNGSP", "BANE", "BANEP", "RNFT"],
    "Металлы и добыча": ["ALRS", "GMKN", "RUAL", "TRMK", "MAGN", "NLMK", "CHMF", "MTLRP", "MTLR", "VSMO", "RASP", "SELG", "PLZL", "UGLD"],
    "IT": ["YDEX", "DATA", "HEAD", "POSI", "VKCO", "ASTR", "IVAT", "DELI", "WUSH", "CNRU", "DIAS", "SOFL"],
    "Телеком": ["MTSS", "RTKMP", "RTKM", "MGTSP"],
    "Строители": ["ETLN", "SMLT", "LSRG", "PIKK"],
    "Ритейл": ["X5", "MGNT", "BELU", "LENT", "OZON", "EUTR", "ABRD", "GCHE", "AQUA", "HNFG", "MVID", "VSEH", "FIXP"],
    "Электро": ["IRAO", "UPRO", "LSNGP", "MSRS", "MRKZ", "MRKU", "MRKC", "MRKP", "FEES", "HYDR", "DVEC", "TGKA", "TGKN", "TGKB", "MSNG", "ELFV"],
    "Транспорт и логистика": ["TRNFP", "AFLT", "FESH", "NMTP", "FLOT"],
    "Агро": ["PHOR", "RAGR", "KZOS", "NKNC", "UFOSP", "KAZT", "AKRN", "NKHP"],
    "Медицина": ["MDMG", "OZPH", "PRMD", "GECO", "APTK", "LIFE", "ABIO", "GEMC"],
    "Машиностроение": ["UWGN", "SVAV", "KMAZ", "UNAC", "IRKT", "ZILLP"]
}

TICKERS_PER_PAGE = 10

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
    await update.message.reply_text("🔍 Ищу Топ по росту денежного потока (Money A/D)...")
    result = []
    for ticker in sum(SECTORS.values(), []):
        try:
            df = get_moex_data(ticker, days=30)
            if df.empty or len(df) < 15:
                continue

            df = df.rename(columns={'CLOSE': 'close', 'VOLUME': 'volume'})  # если еще не переименовано
            df = calculate_money_ad(df)

            ad_start = df['money_ad'].iloc[-10]
            ad_end = df['money_ad'].iloc[-1]
            ad_delta = ad_end - ad_start

            price_start = df['close'].iloc[-10]
            price_end = df['close'].iloc[-1]
            date_start = df.index[-10].strftime('%d.%m.%y')
            date_end = df.index[-1].strftime('%d.%m.%y')

            price_delta = price_end - price_start
            price_pct = 100 * price_delta / price_start

            # Логирование для отладки
            print(f"{ticker} — moneyAD start: {ad_start:.2f}, end: {ad_end:.2f}, Δ: {ad_delta:.2f}, price %: {price_pct:.2f}")

            if ad_delta > 0 or ad_delta < 0 :
                result.append((ticker, round(price_pct, 2), round(ad_delta, 2), date_start, date_end))
        except Exception as e:
            print(f"Ошибка Money A/D для {ticker}: {e}")
            continue

    if not result:
        await update.message.reply_text("Не найдено активов с ростом денежного потока (Money A/D)")
        return

    # Сортировка по дельте денежного потока (рубли)
    result.sort(key=lambda x: x[2], reverse=True)
    result = result[:10]  # топ-10

    msg = "🏦 Топ по росту денежного потока (Money A/D за 2 недели):\n\n"
    for ticker, price_pct, ad_delta, date_start, date_end in result:
        msg += (f"{ticker}: Цена {price_pct:.2f}%, Денежный поток {ad_delta/1000000:.2f} Млн ₽ "
                f"(Дата отсчета {date_start}, Текущая дата {date_end})\n")

    await update.message.reply_text(msg)

# Получение данных для Штейн
def get_moex_weekly_data(ticker="SBER", weeks=100):
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
        df = df.rename(columns={'close': 'CLOSE'})
        df = df[['CLOSE']].dropna()
        return df.tail(weeks)
    except Exception as e:
        print(f"Ошибка получения данных для {ticker}: {e}")
        return pd.DataFrame()

#построение графика штейн
def plot_stan_chart(df, ticker):
    if df.empty:
        return None
    
    try:
        df['SMA30'] = df['CLOSE'].rolling(window=30).mean()
        df['Upper'] = df['SMA30'] + 2 * df['CLOSE'].rolling(window=30).std()
        df['Lower'] = df['SMA30'] - 2 * df['CLOSE'].rolling(window=30).std()

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['CLOSE'], label='Цена', color='blue')
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

# Получение данных с MOEX
def get_moex_data(ticker="SBER", days=100):
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
        df = df.sort_values('begin')  # сортировка по дате
        df.set_index('begin', inplace=True)
        df = df.rename(columns={'close': 'CLOSE', 'volume': 'VOLUME'})
        df = df[['CLOSE', 'VOLUME']].dropna()
        return df.tail(days)
    except Exception as e:
        print(f"Ошибка получения данных для {ticker}: {e}")
        return pd.DataFrame()

# Вычисление RSI вручную
def compute_rsi(series, window=14):
    if len(series) < window:
        return pd.Series([np.nan] * len(series), index=series.index)
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window=window).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(0)

# RSI и аномальные объемы
def analyze_indicators(df):
    if df.empty:
        return df
    
    df['RSI'] = compute_rsi(df['CLOSE'], window=14)
    df['Volume_Mean'] = df['VOLUME'].rolling(window=10).mean()
    df['Anomaly'] = df['VOLUME'] > 1.5 * df['Volume_Mean']
    df['Volume_Multiplier'] = df['VOLUME'] / df['Volume_Mean']
    df['EMA9'] = df['CLOSE'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['CLOSE'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['CLOSE'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['CLOSE'].ewm(span=100, adjust=False).mean()
    df['EMA200'] = df['CLOSE'].ewm(span=200, adjust=False).mean()
    return df

# Поддержка и сопротивление
def find_levels(df):
    if df.empty:
        return []
    
    levels = []
    closes = df['CLOSE'].values
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

# Двойная вершина и дно
def detect_double_patterns(df):
    if df.empty or len(df) < 5:
        return []
    
    closes = df['CLOSE'].values
    patterns = []
    for i in range(2, len(closes) - 2):
        if closes[i-2] < closes[i-1] < closes[i] and closes[i] > closes[i+1] > closes[i+2]:
            patterns.append(('Double Top', df.index[i], closes[i]))
        if closes[i-2] > closes[i-1] > closes[i] and closes[i] < closes[i+1] < closes[i+2]:
            patterns.append(('Double Bottom', df.index[i], closes[i]))
    return patterns

# Построение графика
def plot_stock(df, ticker, levels=[], patterns=[]):
    if df.empty:
        return None
    
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['CLOSE'], label='Цена')

        plt.plot(df.index, df['EMA9'], label='EMA9', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['EMA20'], label='EMA20', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['EMA50'], label='EMA50', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['EMA100'], label='EMA100', linestyle='--', alpha=0.7)
        plt.plot(df.index, df['EMA200'], label='EMA200', linestyle='--', alpha=0.7)

        # Аномальные объемы
        for idx in df[df['Anomaly']].index:
            volume_ratio = df.loc[idx, 'Volume_Multiplier']
            plt.scatter(idx, df.loc[idx, 'CLOSE'], color='red')
            plt.text(idx, df.loc[idx, 'CLOSE'], f"{volume_ratio:.1f}x", color='red', fontsize=8, ha='left')

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

# Telegram команды
if Update and ContextTypes:
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            "Привет! Это бот от команды @TradeAnsh для анализа акций Мосбиржи.\n"
            "Команды:\n"
            "/a — выбрать акцию через кнопки\n"
            "/all — анализ голубых фишек Мосбиржи\n"
            "/stan — анализ акции по методу Стэна Вайнштейна\n"
            "/stan_recent — акции с недавним пересечением SMA30 снизу вверх\n"
            "/long_moneyflow - Топ по росту денежного потока (Money A/D) за 2 недели\n"
        )
        await update.message.reply_text(text)

    async def a(update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [[InlineKeyboardButton(sector, callback_data=f"sector:{sector}:0")] for sector in SECTORS]
        await update.message.reply_text("Выберите отрасль:", reply_markup=InlineKeyboardMarkup(keyboard))

    async def stan(update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [[InlineKeyboardButton(sector, callback_data=f"stan_sector:{sector}:0")] for sector in SECTORS]
        await update.message.reply_text("Выберите отрасль для анализа по Штейну:", reply_markup=InlineKeyboardMarkup(keyboard))

    async def all(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Начинаю анализ всех акций. Это может занять некоторое время...")
        
        for ticker in sum(SECTORS.values(), []):
            try:
                df = get_moex_data(ticker)
                if df.empty:
                    await update.message.reply_text(f"❌ Не удалось получить данные для {ticker}")
                    continue
                    
                df = analyze_indicators(df)
                levels = find_levels(df)
                patterns = detect_double_patterns(df)
                chart = plot_stock(df, ticker, levels, patterns)
                
                if chart is None:
                    await update.message.reply_text(f"❌ Ошибка при создании графика для {ticker}")
                    continue
                
                rsi_series = df['RSI'].dropna()
                rsi_value = rsi_series.iloc[-1] if not rsi_series.empty else "Недостаточно данных для RSI"
                latest_date = df.index.max().strftime('%Y-%m-%d')
                text_summary = f"\nПоследний RSI: {rsi_value}\nАктуальность данных: до {latest_date}\n"
                
                with open(chart, 'rb') as photo:
                    await update.message.reply_photo(photo=photo)
                await update.message.reply_text(f"{ticker}\n{text_summary}")
                
                # Удаляем файл после отправки
                if os.path.exists(chart):
                    os.remove(chart)
                    
            except Exception as e:
                await update.message.reply_text(f"❌ Ошибка при анализе {ticker}: {str(e)}")

    def find_sma30_crossover(ticker, days=7):
        """
        Находит пересечение цены снизу вверх через SMA30 за последние дни
        И проверяет, что на текущий момент цена находится выше SMA30
        Возвращает дату пересечения или None
        """
        try:
            df = get_moex_data(ticker, days=60)  # Берём больше данных для расчета SMA30
            if df.empty or len(df) < 35:  # Нужно минимум 35 дней для SMA30 + проверка
                return None
            
            # Вычисляем SMA30
            df['SMA30'] = df['CLOSE'].rolling(window=30).mean()
            
            # Проверяем, что текущая цена выше SMA30
            current_close = df['CLOSE'].iloc[-1]
            current_sma30 = df['SMA30'].iloc[-1]
            
            if current_close <= current_sma30:
                return None  # Текущая цена не выше SMA30
            
            # Берём только последние days дней для поиска пересечений
            recent_df = df.tail(days + 1)  # +1 для сравнения с предыдущим днём
            
            crossover_date = None
            
            # Ищем пересечение снизу вверх
            for i in range(1, len(recent_df)):
                prev_close = recent_df['CLOSE'].iloc[i-1]
                curr_close = recent_df['CLOSE'].iloc[i]
                prev_sma = recent_df['SMA30'].iloc[i-1]
                curr_sma = recent_df['SMA30'].iloc[i]
                
                # Проверяем пересечение: вчера цена была ниже SMA30, сегодня выше
                if (prev_close < prev_sma and curr_close > curr_sma):
                    crossover_date = recent_df.index[i]
                    break
            
            return crossover_date
            
        except Exception as e:
            print(f"Ошибка при поиске пересечения SMA30 для {ticker}: {e}")
            return None  
            
    async def stan_recent(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔍 Ищу акции с недавним пересечением цены через SMA30 снизу вверх...")
        
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

    async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data

        try:
            # === Обработка обычной команды /a ===
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
                
                # Удаляем файл после отправки
                if os.path.exists(chart):
                    os.remove(chart)

            # === Обработка команды /stan ===
            elif data.startswith("stan_sector:"):
                _, sector, page = data.split(":")
                page = int(page)
                tickers = SECTORS.get(sector, [])
                start = page * TICKERS_PER_PAGE
                end = start + TICKERS_PER_PAGE
                visible = tickers[start:end]

                keyboard = [[InlineKeyboardButton(t, callback_data=f"stan_ticker:{t}")] for t in visible]
                nav = []
                if start > 0:
                    nav.append(InlineKeyboardButton("⬅️", callback_data=f"stan_sector:{sector}:{page-1}"))
                if end < len(tickers):
                    nav.append(InlineKeyboardButton("➡️", callback_data=f"stan_sector:{sector}:{page+1}"))
                if nav:
                    keyboard.append(nav)
                keyboard.append([InlineKeyboardButton("🔙 Назад к отраслям", callback_data="stan_back")])

                await query.edit_message_text(f"Вы выбрали отрасль: {sector}. Теперь выберите тикер:", reply_markup=InlineKeyboardMarkup(keyboard))

            elif data == "stan_back":
                keyboard = [[InlineKeyboardButton(sector, callback_data=f"stan_sector:{sector}:0")] for sector in SECTORS]
                await query.edit_message_text("Выберите отрасль для анализа по Штейну:", reply_markup=InlineKeyboardMarkup(keyboard))

            elif data.startswith("stan_ticker:"):
                ticker = data.split(":", 1)[1]
                await query.edit_message_text(f"Вы выбрали тикер: {ticker}. Выполняется анализ по Штейну...")

                df = get_moex_weekly_data(ticker)
                if df.empty:
                    await context.bot.send_message(chat_id=query.message.chat.id, text=f"❌ Не удалось получить данные для {ticker}")
                    return

                chart = plot_stan_chart(df, ticker)
                if chart is None:
                    await context.bot.send_message(chat_id=query.message.chat.id, text=f"❌ Ошибка при создании графика для {ticker}")
                    return

                latest_date = df.index.max().strftime('%Y-%m-%d')
                with open(chart, 'rb') as photo:
                    await context.bot.send_photo(chat_id=query.message.chat.id, photo=photo)
                await context.bot.send_message(chat_id=query.message.chat.id, text=f"График построен по данным на {latest_date}")
                
                # Удаляем файл после отправки
                if os.path.exists(chart):
                    os.remove(chart)

        except Exception as e:
            await context.bot.send_message(chat_id=query.message.chat.id, text=f"❌ Произошла ошибка: {str(e)}")

# ==== Flask сервер для поддержки работы 24/7 ====
from flask import Flask
from threading import Thread

app_web = Flask('')

@app_web.route('/')
def home():
    return "Бот работает!"

def run():
    app_web.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()

# ==== Запуск Telegram-бота с веб-сервером ====
if ApplicationBuilder:
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    if TOKEN is None:
        print("Ошибка: переменная окружения TELEGRAM_TOKEN не установлена.")
    else:
        keep_alive()  # ← запуск Flask
        app = ApplicationBuilder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("a", a))
        app.add_handler(CommandHandler("all", all))  # Используем функцию all
        app.add_handler(CommandHandler("stan", stan))
        app.add_handler(CommandHandler("stan_recent", stan_recent))
        app.add_handler(CommandHandler("long_moneyflow", long_moneyflow))
        app.add_handler(CallbackQueryHandler(handle_callback))
        print("✅ Бот запущен и поддерживается Flask-сервером.")
        app.run_polling()
else:
    print("Функциональность Telegram-бота отключена из-за отсутствия библиотеки 'telegram'.")
