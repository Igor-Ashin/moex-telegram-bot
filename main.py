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
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove
    from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, ConversationHandler, MessageHandler, filters
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

ASK_DAYS = 1  # состояние для выбора дней
ASK_TICKER = 2
ASK_DELTA_DAYS = 3

async def ask_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📅 Введите количество дней для расчета дельты денежного потока (например, 10):")
    return ASK_DAYS

async def receive_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        days = int(update.message.text)
        if not (1 <= days <= 60):
            await update.message.reply_text("⚠️ Введите число от 1 до 30.")
            return ASK_DAYS

        context.user_data['days'] = days
        await long_moneyflow(update, context)
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text("⚠️ Введите целое число, например: 10")
        return ASK_DAYS

# Добавьте эти функции после функции receive_days:

async def ask_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запрашивает тикер у пользователя"""
    await update.message.reply_text("📊 Введите тикер акции (например, SBER):")
    return ASK_TICKER

async def receive_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получает тикер и запрашивает количество дней"""
    ticker = update.message.text.strip().upper()
    
    # Проверяем, что тикер существует в наших секторах
    all_tickers = sum(SECTORS.values(), [])
    if ticker not in all_tickers:
        await update.message.reply_text(f"⚠️ Тикер '{ticker}' не найден в базе данных. Проверьте правильность написания.")
        return ASK_TICKER
    
    context.user_data['delta_ticker'] = ticker
    await update.message.reply_text(f"✅ Вы ввели тикер: {ticker}\n\n📅 Введите количество дней для расчета дельты денежного потока (например, 10):")
    return ASK_DELTA_DAYS


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
            
            # Получаем данные за последние 8 дней для анализа
            recent = df.tail(15)  # 7 дней + текущий
            
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
        msg += "\n".join(f"{t} {d}" for t, d in short_hits)
    else:
        msg += "🔴 *Шорт сигналов не найдено за последние 14 дней*"
    
    await update.message.reply_text(msg, parse_mode="Markdown")

async def receive_delta_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получает количество дней и выполняет расчет дельты"""
    try:
        days = int(update.message.text)
        if not (1 <= days <= 100):
            await update.message.reply_text("⚠️ Введите число от 1 до 100.")
            return ASK_DELTA_DAYS

        ticker = context.user_data['delta_ticker']
        await calculate_single_delta(update, context, ticker, days)
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text("⚠️ Введите целое число, например: 10")
        return ASK_DELTA_DAYS

async def calculate_single_delta(update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str, days: int):
    """Рассчитывает дельту денежного потока для одной акции"""
    await update.message.reply_text(f"🔍 Рассчитываю дельту денежного потока для {ticker} за {days} дней...")
    
    try:
        df = get_moex_data(ticker, days=days + 5)  # с запасом
        if df.empty or len(df) < days + 1:
            await update.message.reply_text(f"❌ Недостаточно данных для {ticker}. Попробуйте уменьшить количество дней.")
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
        
        # 📊 Отношение дельты потока к обороту (%)
        if avg_turnover != 0:
            delta_vs_turnover = 100 * ad_delta / avg_turnover
        else:
            delta_vs_turnover = 0

        # Формируем сообщение
        msg = f"📊 Анализ дельты денежного потока для {ticker}\n"
        msg += f"📅 Период: {date_start} – {date_end} ({days} дней)\n\n"
        
        # Добавляем предупреждение о низком обороте
        if filter_avg_turnover < 50_000_000:
            msg += "⚠️ Внимание: низкий среднедневной оборот (< 50 млн ₽)\n\n"
        
        msg += "<pre>\n"
        msg += f"{'Тикер':<6}  {'Изм. цены':<9}  {'Δ Потока':>19}  {'Δ / Оборот':>12}\n"
        msg += f"{ticker:<6}  {price_pct:+8.2f}%  {ad_delta/1_000_000:13,.2f} млн ₽  {delta_vs_turnover:9.1f}%\n"
        msg += "</pre>\n\n"
        
        # Добавляем интерпретацию результатов
        if ad_delta > 0:
            msg += "📈 Положительная дельта потока - деньги притекают в акцию\n"
        else:
            msg += "📉 Отрицательная дельта потока - деньги оттекают из акции\n"
        
        msg += f"💰 Среднедневной оборот: {avg_turnover/1_000_000:.1f} млн ₽"
        
        await update.message.reply_text(msg, parse_mode="HTML")
        
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
    await update.message.reply_text(f"🔍 Ищу Топ по росту и оттоку денежного потока за {days} дней...")
    
    result = []
    for ticker in sum(SECTORS.values(), []):
        try:
            df = get_moex_data(ticker, days=days + 5)  # с запасом
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
                    round(delta_vs_turnover, 2)
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
        msg += "📈 Топ 10 по росту:\n"
        msg += "<pre>\n"
        msg += f"{'Тикер':<6}  {'Изм. цены':<9}  {'Δ Потока':>19}  {'Δ / Оборот':>12}\n"
        # Убираем линию с дефисами, как просил
        for ticker, price_pct, ad_delta, _, _, delta_pct in result_up[:10]:
            msg += f"{ticker:<6}  {price_pct:+8.2f}%  {ad_delta/1_000_000:13,.2f} млн ₽  {delta_pct:9.1f}%\n"
        msg += "</pre>\n\n"
    
    # 📉 Падение
    if result_down:
        msg += "📉 Топ 10 по оттоку:\n"
        msg += "<pre>\n"
        msg += f"{'Тикер':<6}  {'Изм. цены':<9}  {'Δ Потока':>19}  {'Δ / Оборот':>12}\n"
        # Линию тоже убираем
        for ticker, price_pct, ad_delta, _, _, delta_pct in result_down[:10]:
            msg += f"{ticker:<6}  {price_pct:+8.2f}%  {ad_delta/1_000_000:13,.2f} млн ₽  {delta_pct:9.1f}%\n"
        msg += "</pre>\n"
    
    await update.message.reply_text(msg, parse_mode="HTML")




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
        df = df.rename(columns={'close': 'close'})
        df = df[['close']].dropna()
        return df.tail(weeks)
    except Exception as e:
        print(f"Ошибка получения данных для {ticker}: {e}")
        return pd.DataFrame()

#построение графика штейн
def plot_stan_chart(df, ticker):
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

# Получение данных с MOEX
def get_moex_data(ticker="SBER", days=120):
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

# Поддержка и сопротивление
def find_levels(df):
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

# Двойная вершина и дно
def detect_double_patterns(df):
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

# Построение графика
def plot_stock(df, ticker, levels=[], patterns=[]):
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

# Telegram команды
if Update and ContextTypes:
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            "Привет! Это бот от команды @TradeAnsh для анализа акций Мосбиржи.\n"
            "Команды:\n"
            "/chart_hv — выбрать акцию через кнопки\n"
            "/stan — анализ акции по методу Стэна Вайнштейна\n"
            "/cross_ema20x50 — акции с пересечением EMA 20x50 на 1D\n"
            "/stan_recent — акции с лонг пересечением SMA30 на 1D\n"
            "/stan_recent_short — акции с шорт пересечением SMA30 на 1D\n"
            "/stan_recent_week — акции с лонг пересечением SMA30 на 1W\n"
            "/moneyflow - Топ по росту и оттоку денежного потока (Money A/D)\n"
            "/delta — расчет дельты денежного потока для конкретной акции\n"
            "/rsi_top — Топ 10 перекупленных и перепроданных акций по RSI\n"
        )
        await update.message.reply_text(text)

    async def chart_hv(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

            # 💰 Среднедневной оборот за фиксированные 10 дней (для фильтра)
            filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
            filter_avg_turnover = filter_turnover_series.mean()
            
            # ❌ Фильтр по минимальному обороту: 50 млн руб за последние 10 дней
            if filter_avg_turnover < 50_000_000:
                return None  

            
            # Вычисляем SMA30
            df['SMA30'] = df['close'].rolling(window=30).mean()
            
            # Проверяем, что текущая цена выше SMA30
            current_close = df['close'].iloc[-1]
            current_sma30 = df['SMA30'].iloc[-1]
            
            if current_close <= current_sma30:
                return None  # Текущая цена не выше SMA30
            
            # Берём только последние days дней для поиска пересечений
            recent_df = df.tail(days + 1)  # +1 для сравнения с предыдущим днём
            
            crossover_date = None
            
            # Ищем пересечение снизу вверх
            for i in range(1, len(recent_df)):
                prev_close = recent_df['close'].iloc[i-1]
                curr_close = recent_df['close'].iloc[i]
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


    def find_sma30_crossover_short(ticker, days=7):
        """
        Находит пересечение цены снизу вверх через SMA30 за последние дни
        И проверяет, что на текущий момент цена находится выше SMA30
        Возвращает дату пересечения или None
        """
        try:
            df = get_moex_data(ticker, days=60)  # Берём больше данных для расчета SMA30
            if df.empty or len(df) < 35:  # Нужно минимум 35 дней для SMA30 + проверка
                return None

            # 💰 Среднедневной оборот за фиксированные 10 дней (для фильтра)
            filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
            filter_avg_turnover = filter_turnover_series.mean()
            
            # ❌ Фильтр по минимальному обороту: 50 млн руб за последние 10 дней
            if filter_avg_turnover < 50_000_000:
                return None  

            
            # Вычисляем SMA30
            df['SMA30'] = df['close'].rolling(window=30).mean()
            
            # Проверяем, что текущая цена ниже SMA30
            current_close = df['close'].iloc[-1]
            current_sma30 = df['SMA30'].iloc[-1]
            
            if current_close >= current_sma30:
                return None  # Текущая цена не ниже SMA30
            
            # Берём только последние days дней для поиска пересечений
            recent_df = df.tail(days + 1)  # +1 для сравнения с предыдущим днём
            
            crossover_date = None
            
            # Ищем пересечение снизу вверх
            for i in range(1, len(recent_df)):
                prev_close = recent_df['close'].iloc[i-1]
                curr_close = recent_df['close'].iloc[i]
                prev_sma = recent_df['SMA30'].iloc[i-1]
                curr_sma = recent_df['SMA30'].iloc[i]
                
                # Проверяем пересечение: вчера цена была выше SMA30, сегодня ниже
                if (prev_close > prev_sma and curr_close < curr_sma):
                    crossover_date = recent_df.index[i]
                    break
            
            return crossover_date
            
        except Exception as e:
            print(f"Ошибка при поиске пересечения SMA30 для {ticker}: {e}")
            return None  
    
    def find_sma30_crossover_week(ticker, weeks=5):
        """
        Находит пересечение цены снизу вверх через SMA30 за последние дни
        И проверяет, что на текущий момент цена находится выше SMA30
        Возвращает дату пересечения или None
        """
        try:
            df = get_moex_weekly_data(ticker, weeks=60)  # Берём больше данных для расчета SMA30
            if df.empty or len(df) < 35:  # Нужно минимум 35 недель для SMA30 + проверка 
                return None

            dfd = get_moex_data(ticker, days=20)  # Берём больше данных для расчета среднего объема
            if dfd.empty or len(dfd) < 15:  # Нужно минимум 15 дней для расчета среднего объема
                return None

            # 💰 Среднедневной оборот за фиксированные 10 дней (для фильтра)
            filter_turnover_series = dfd['volume'].iloc[-10:] * dfd['close'].iloc[-10:]
            filter_avg_turnover = filter_turnover_series.mean()
            
            # ❌ Фильтр по минимальному обороту: 50 млн руб за последние 10 дней
            if filter_avg_turnover < 50_000_000:
                return None  

            
            # Вычисляем SMA30
            df['SMA30'] = df['close'].rolling(window=30).mean()
            
            # Проверяем, что текущая цена выше SMA30
            current_close = df['close'].iloc[-1]
            current_sma30 = df['SMA30'].iloc[-1]
            
            if current_close <= current_sma30:
                return None  # Текущая цена не выше SMA30
            
            # Берём только последние days дней для поиска пересечений
            recent_df = df.tail(weeks + 1)  # +1 для сравнения с предыдущим днём
            
            crossover_date = None
            
            # Ищем пересечение снизу вверх
            for i in range(1, len(recent_df)):
                prev_close = recent_df['close'].iloc[i-1]
                curr_close = recent_df['close'].iloc[i]
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


    async def stan_recent_d_short(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔍 Ищу акции с недавним пересечением цены через SMA30 снизу вверх...")
        
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
        await update.message.reply_text("🔍 Ищу акции с недавним пересечением цены через SMA30 снизу вверх...")
        
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
        app.add_handler(CommandHandler("chart_hv", chart_hv))
        app.add_handler(CommandHandler("cross_ema20x50", cross_ema20x50))
        app.add_handler(CommandHandler("stan", stan))
        app.add_handler(CommandHandler("stan_recent", stan_recent))
        app.add_handler(CommandHandler("stan_recent_d_short", stan_recent_d_short))
        app.add_handler(CommandHandler("stan_recent_week", stan_recent_week))
        app.add_handler(CommandHandler("long_moneyflow", long_moneyflow))
        app.add_handler(CommandHandler("rsi_top", rsi_top))
        app.add_handler(CallbackQueryHandler(handle_callback))
        delta_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("delta", ask_ticker)],
            states={
                ASK_TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_ticker)],
                ASK_DELTA_DAYS: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_delta_days)]
            },
            fallbacks=[],
        )
        app.add_handler(delta_conv_handler)

        # === Хендлер с диалогом выбора дней ===
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler("moneyflow", ask_days)],
            states={
                ASK_DAYS: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_days)]
            },
            fallbacks=[],
        )
        app.add_handler(conv_handler)
        
        print("✅ Бот запущен и поддерживается Flask-сервером.")
        app.run_polling()
else:
    print("Функциональность Telegram-бота отключена из-за отсутствия библиотеки 'telegram'.")
