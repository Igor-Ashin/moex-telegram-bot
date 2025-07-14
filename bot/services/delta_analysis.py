from telegram import Update
from telegram.ext import ContextTypes
import pandas as pd
import logging
from typing import Optional

from data.optimized_moex_client import optimized_moex_client
from analysis.indicators import calculate_money_ad
from bot.utils.decorators import telegram_handler, typing_action, performance_monitor

logger = logging.getLogger(__name__)

@telegram_handler
@typing_action 
@performance_monitor("calculate_single_delta")
async def calculate_single_delta(update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str, days: int):
    """
    Рассчитывает дельту денежного потока для одной акции
    Актуализированная версия из main.py
    """
    await update.message.reply_text(f"🔍 Рассчитываю дельту денежного потока для {ticker} за {days} дней...")
    
    try:
        # Получаем дневные данные с запасом
        df = optimized_moex_client.get_daily_data(ticker, days=100)
        if df.empty or len(df) < days + 1:
            await update.message.reply_text(f"❌ Недостаточно данных для {ticker}. Попробуйте увеличить количество дней.")
            return

        # Переименовываем колонки для совместимости
        df = df.rename(columns={'close': 'close', 'volume': 'volume'})
        
        # Вычисляем Money A/D используя актуальный алгоритм
        df = calculate_money_ad(df)

        # Рассчитываем дельту A/D Line
        ad_start = df['money_ad'].iloc[-(days+1)]
        ad_end = df['money_ad'].iloc[-1]
        ad_delta = ad_end - ad_start

        # Данные по ценам
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
            wdf = optimized_moex_client.get_weekly_data(ticker, weeks=80)  # Больше недель для SMA30
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
        msg = f"📊 Анализ дельты денежного потока для {ticker}\n"
        msg += f"📅 Период: {date_start} – {date_end} ({days} дней)\n\n"
        
        # Добавляем предупреждение о низком обороте
        if filter_avg_turnover < 50_000_000:
            msg += "⚠️ Внимание: низкий среднедневной оборот (< 50 млн ₽)\n\n"

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
        
        msg += "<pre>\n"
        msg += f"{'Тикер':<6} {'Δ Цены':<9} {flow_icon}{'Δ Потока':>11} {'Δ / Оборот':>8} {'Δ Цены 1D':>8} {'Объём':>8} {'ema20х50':>7} {'sma30':>4}\n"
        msg += f"{ticker:<6} {price_pct:5.1f}% {ad_delta/1_000_000:11,.0f} млн ₽ {delta_pct:8.1f}%  {price_change_day*100:>8.1f}%  {ratio:>6.1f}x {ema_icon:>5} {sma_icon:>4}\n"
        msg += "</pre>\n"
        
        msg += f"💰 Среднедневной оборот: {avg_turnover/1_000_000:.1f} млн ₽\n"
        
        await update.message.reply_text(msg, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error calculating delta for {ticker}: {e}")
        await update.message.reply_text(f"❌ Ошибка при расчете дельты для {ticker}: {str(e)}")

class DeltaAnalyzer:
    """Класс для анализа дельты денежного потока"""
    
    def __init__(self):
        self.min_turnover = 50_000_000  # 50 млн руб
    
    def calculate_delta_metrics(self, ticker: str, days: int) -> Optional[dict]:
        """
        Возвращает метрики дельты без отправки сообщений
        Полезно для использования в других модулях
        """
        try:
            # Получаем данные
            df = optimized_moex_client.get_daily_data(ticker, days=100)
            if df.empty or len(df) < days + 1:
                return None

            # Вычисляем Money A/D
            df = calculate_money_ad(df)

            # Основные метрики
            ad_start = df['money_ad'].iloc[-(days+1)]
            ad_end = df['money_ad'].iloc[-1]
            ad_delta = ad_end - ad_start

            price_start = df['close'].iloc[-(days+1)]
            price_end = df['close'].iloc[-1]
            price_pct = 100 * (price_end - price_start) / price_start

            # Оборот
            turnover_series = df['volume'].iloc[-days:] * df['close'].iloc[-days:]
            avg_turnover = turnover_series.mean()
            
            filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
            filter_avg_turnover = filter_turnover_series.mean()

            # EMA сигналы
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            current_ema20 = df['EMA20'].iloc[-1]
            current_ema50 = df['EMA50'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            ema20x50_long = (current_ema20 > current_ema50) and (current_price > current_ema20)
            ema20x50_short = (current_ema20 < current_ema50) and (current_price < current_ema20)

            # Weekly SMA30
            try:
                wdf = optimized_moex_client.get_weekly_data(ticker, weeks=80)
                if len(wdf) >= 30:
                    wdf['SMA30'] = wdf['close'].rolling(window=30).mean()
                    weekly_sma30 = wdf['SMA30'].iloc[-1]
                    weekly_price = wdf['close'].iloc[-1]
                    price_above_sma30 = weekly_price > weekly_sma30 if pd.notna(weekly_sma30) else False
                else:
                    price_above_sma30 = False
            except:
                price_above_sma30 = False

            return {
                'ticker': ticker,
                'days': days,
                'ad_delta': ad_delta,
                'ad_delta_millions': ad_delta / 1_000_000,
                'price_change_pct': price_pct,
                'avg_turnover': avg_turnover,
                'avg_turnover_millions': avg_turnover / 1_000_000,
                'filter_avg_turnover': filter_avg_turnover,
                'low_turnover': filter_avg_turnover < self.min_turnover,
                'delta_to_turnover_pct': 100 * ad_delta / avg_turnover if avg_turnover > 0 else 0,
                'ema20x50_long': ema20x50_long,
                'ema20x50_short': ema20x50_short,
                'price_above_sma30_weekly': price_above_sma30,
                'current_price': current_price,
                'date_start': df.index[-(days+1)].strftime('%d.%m.%y'),
                'date_end': df.index[-1].strftime('%d.%m.%y')
            }

        except Exception as e:
            logger.error(f"Error calculating delta metrics for {ticker}: {e}")
            return None

    def scan_delta_opportunities(self, tickers: list, days: int = 10) -> list:
        """
        Сканирует список тикеров для поиска интересных сигналов по дельте
        """
        opportunities = []
        
        for ticker in tickers:
            try:
                metrics = self.calculate_delta_metrics(ticker, days)
                if metrics and not metrics['low_turnover']:
                    # Критерии для интересных сигналов
                    if (metrics['ad_delta'] > 0 and  # Приток денег
                        metrics['ema20x50_long'] and  # Лонг сигнал EMA
                        metrics['price_above_sma30_weekly']):  # Цена выше SMA30 weekly
                        
                        opportunities.append(metrics)
                        
            except Exception as e:
                logger.warning(f"Error scanning {ticker}: {e}")
        
        # Сортируем по силе сигнала (дельта к обороту)
        opportunities.sort(key=lambda x: x['delta_to_turnover_pct'], reverse=True)
        
        return opportunities

# Глобальный экземпляр анализатора
delta_analyzer = DeltaAnalyzer()