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
    "Финансы": ["T", "LEAS", "CARM", "ZAYM", "MBNK", "SBERP", "MOEX", "RENI", "BSPB", "SVCB", "VTBR", "SBER", "CBOM"],
    "Нефтегаз": ["BANE", "TATN", "RNFT", "BANEP", "NVTK", "LKOH", "ROSN", "TATNP", "GAZP", "SNGS", "SNGSP"],
    "Металлы и добыча": ["TRMK", "MTLRP", "AMEZ", "ROLO", "RUAL", "VSMO", "LNZL", "RASP", "ALRS", "GMKN", "SELG", "PLZL", "UGLD", "MTLR", "MAGN", "NLMK", "CHMF"],
    "IT": ["CNRU", "YDEX", "DATA", "HEAD", "IVAT", "DELI", "WUSH", "POSI", "VKCO", "ASTR", "DIAS", "SOFL"],
    "Телеком": ["NSVZ", "RTKMP", "VEON-RX", "TTLK", "MGTSP", "MTSS", "RTKM"],
    "Строители": ["ETLN", "SMLT", "LSRG", "PIKK"],
    "Ритейл": ["VSEH", "EUTR", "KROT", "ABRD", "GCHE", "AQUA", "HNFG", "FIXP", "BELU", "LENT", "OKEY", "OZON", "MVID", "MGNT"],
    "Электро": ["HYDR", "DVEC", "TGKA", "TGKN", "TGKB", "LSNG", "MSNG", "ELFV", "IRAO", "UPRO", "MSRS", "MRKZ", "MRKU", "MRKC", "MRKP", "FEES"],
    "Транспорт и логистика": ["TRNFP", "AFLT", "FESH", "NMTP", "FLOT"],
    "Агро": ["KZOS", "NKNC", "UFOSP", "KAZT", "AKRN", "NKHP", "PHOR"],
    "Медицина": ["OZPH", "PRMD", "GECO", "APTK", "LIFE", "ABIO", "GEMC"],
    "Машиностроение": ["IRKT", "ZILLP", "UWGN", "SVAV", "KMAZ", "UNAC"]
}

TICKERS_PER_PAGE = 5

# Получение данных с MOEX

def get_moex_data(ticker="SBER", days=100):
    till = datetime.today().strftime('%Y-%m-%d')
    from_date = (datetime.today() - pd.Timedelta(days=days * 1.5)).strftime('%Y-%m-%d')
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?interval=24&from={from_date}&till={till}"
    r = requests.get(url)
    data = r.json()
    candles = data['candles']['data']
    columns = data['candles']['columns']
    df = pd.DataFrame(candles, columns=columns)
    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    df = df.rename(columns={'close': 'CLOSE', 'volume': 'VOLUME'})
    df = df[['CLOSE', 'VOLUME']].dropna()
    return df.tail(days)

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
    levels = []
    closes = df['CLOSE'].values
    indexes = np.arange(len(closes))
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
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['CLOSE'], label='Цена')

    plt.plot(df.index, df['EMA9'], label='EMA9', linestyle='--', alpha=0.7)
    plt.plot(df.index, df['EMA20'], label='EMA20', linestyle='--', alpha=0.7)
    plt.plot(df.index, df['EMA50'], label='EMA50', linestyle='--', alpha=0.7)
    plt.plot(df.index, df['EMA100'], label='EMA100', linestyle='--', alpha=0.7)  # EMA100
    plt.plot(df.index, df['EMA200'], label='EMA200', linestyle='--', alpha=0.7)  # EMA200

    # Горизонтальные объемы
  # plt.bar(df.index, df['VOLUME'], width=0.8, color='gray', alpha=0.3, label="Объем")
    
    for idx in df[df['Anomaly']].index:
        volume_ratio = df.loc[idx, 'Volume_Multiplier']
        plt.scatter(idx, df.loc[idx, 'CLOSE'], color='red')
        plt.text(idx, df.loc[idx, 'CLOSE'], f"{volume_ratio:.1f}x", color='red', fontsize=8, ha='left')

    for date, price in levels:
        plt.axhline(price, linestyle='--', alpha=0.3)

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

# Telegram команда

if Update and ContextTypes:
    async def a(update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [[InlineKeyboardButton(sector, callback_data=f"sector:{sector}:0")] for sector in SECTORS]
        await update.message.reply_text("Выберите отрасль:", reply_markup=InlineKeyboardMarkup(keyboard))

    async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data
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
            df = analyze_indicators(df)
            levels = find_levels(df)
            patterns = detect_double_patterns(df)
            chart = plot_stock(df, ticker, levels, patterns)
            rsi_series = df['RSI'].dropna()
            rsi_value = rsi_series.iloc[-1] if not rsi_series.empty else "Недостаточно данных для RSI"
            latest_date = df.index.max().strftime('%Y-%m-%d')
            text_summary = f"\nПоследний RSI: {rsi_value}\nАктуальность данных: до {latest_date}\n"
            await context.bot.send_photo(chat_id=query.message.chat.id, photo=open(chart, 'rb'))
            await context.bot.send_message(chat_id=query.message.chat.id, text=text_summary)

   #     if patterns:
    #        text_summary += "\nОбнаружены паттерны:\n"
     #       for p in patterns:
      #          text_summary += f"- {p[0]} на {p[1].date()} по цене {p[2]:.2f}\n"



        async def all(update: Update, context: ContextTypes.DEFAULT_TYPE):
        for ticker in sum(SECTORS.values(), []):
            try:
                df = get_moex_data(ticker)
                df = analyze_indicators(df)
                levels = find_levels(df)
                patterns = detect_double_patterns(df)
                chart = plot_stock(df, ticker, levels, patterns)
                rsi_series = df['RSI'].dropna()
                rsi_value = rsi_series.iloc[-1] if not rsi_series.empty else "Недостаточно данных для RSI"
                latest_date = df.index.max().strftime('%Y-%m-%d')
                text_summary = f"\nПоследний RSI: {rsi_value}\nАктуальность данных: до {latest_date}\n"
                

           #     if patterns:
            #        text_summary += "\nОбнаружены паттерны:\n"
             #       for p in patterns:
              #          text_summary += f"- {p[0]} на {p[1].date()} по цене {p[2]:.2f}\n"

                await update.message.reply_photo(photo=open(chart, 'rb'))
                await update.message.reply_text(f"{ticker}\n{text_summary}")
            except Exception as e:
                await update.message.reply_text(f"Ошибка при анализе {ticker}: {e}")
                continue

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            "Привет! Я это бот от команды @TradeAnsh для анализа акций Мосбиржи.\n"
            "Команды:\n"
            "/a <тикер> — анализ одной акции (например: /a SBER)\n"
            "/all — анализ всех голубых фишек Мосбиржи\n\n"
            "Популярные тикеры:\n"
            "SBER, GAZP, LKOH, GMKN, ROSN, TATN, YDEX"
        )
        await update.message.reply_text(text)





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
        app.add_handler(CommandHandler("all", all))
        app.add_handler(CallbackQueryHandler(handle_callback))
        print("✅ Бот запущен и поддерживается Flask-сервером.")
        app.run_polling()
else:
    print("Функциональность Telegram-бота отключена из-за отсутствия библиотеки 'telegram'.")
