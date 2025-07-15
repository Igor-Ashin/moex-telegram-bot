from telegram import Update
from telegram.ext import ContextTypes
import pandas as pd
import logging
from typing import List, Tuple

from bot.utils.decorators import telegram_handler, fast_telegram_handler, typing_action, performance_monitor
from data.optimized_moex_client import optimized_moex_client
from analysis.indicators import analyze_indicators
from config.sectors import get_all_tickers, get_sector_tickers
from data.cache import cache
from bot.services.delta_analysis import calculate_single_delta, delta_analyzer

logger = logging.getLogger(__name__)

@fast_telegram_handler
@typing_action
async def start_optimized(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Оптимизированная команда /start - быстрая, без rate limiting
    """
    text = (
        "🚀 **MOEX Bot 2.0** - Оптимизированная версия\n\n"
        "**Быстрые команды:**\n"
        "/scan_fast — Быстрое сканирование топ-акций\n"
        "/cache_stats — Статистика кэша\n"
        "/performance — Статистика производительности\n\n"
        
        "**Анализ рынка:**\n"
        "/rsi_top — RSI анализ (оптимизированный)\n"
        "/volume_scan — Поиск аномальных объемов\n"
        "/delta_scan — Дельта-сигналы денежного потока\n"
        "/ema_signals — Сигналы по EMA пересечениям\n\n"
        
        "**Настройки:**\n"
        "/clear_cache — Очистить кэш\n"
        "/warm_cache — Прогреть кэш популярных акций\n\n"
        
        "✨ Улучшения: кэширование, параллельная обработка, rate limiting"
    )
    
    await update.message.reply_text(text, parse_mode='Markdown')

@fast_telegram_handler
async def cache_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает статистику кэша"""
    try:
        # Статистика кэша
        cache_stats = cache.get_cache_stats()
        
        # Статистика MOEX клиента
        client_stats = optimized_moex_client.get_performance_stats()
        
        message = (
            "📊 **Статистика производительности**\n\n"
            
            "**Кэш:**\n"
            f"• Всего запросов: {cache_stats['total_requests']}\n"
            f"• Попадания: {cache_stats['hits']}\n"
            f"• Промахи: {cache_stats['misses']}\n"
            f"• Hit Rate: {cache_stats['hit_rate']}\n"
            f"• Размер кэша: {cache_stats['memory_cache_size']} записей\n\n"
            
            "**MOEX API:**\n"
            f"• API вызовов: {client_stats['api_calls']}\n"
            f"• Кэш попаданий: {client_stats['cache_hits']}\n"
            f"• Hit Rate: {client_stats['cache_hit_rate']}\n"
            f"• Среднее время API: {client_stats['avg_api_response_time']}\n"
            f"• Общее время API: {client_stats['total_api_time']}\n"
        )
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        await update.message.reply_text("❌ Ошибка получения статистики")

@telegram_handler
@typing_action
@performance_monitor("scan_fast")
async def scan_fast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Быстрое сканирование популярных акций
    Использует кэш для мгновенного ответа
    """
    # Популярные тикеры - часто запрашиваемые пользователями
    popular_tickers = [
        "SBER", "GAZP", "LKOH", "YDEX", "MGNT", "ROSN", 
        "NVTK", "VTBR", "ALRS", "MTSS", "MOEX", "PIKK"
    ]
    
    message = await update.message.reply_text("🚀 Быстрое сканирование рынка...")
    
    try:
        # Получаем данные (будут из кэша если доступны)
        data_dict = optimized_moex_client.get_multiple_daily_data(popular_tickers, days=50)
        
        results = []
        for ticker, df in data_dict.items():
            if df.empty:
                continue
                
            # Анализируем индикаторы
            df = analyze_indicators(df)
            
            current_price = df['close'].iloc[-1]
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]) else None
            
            # Определяем сигнал по EMA
            signal = "📊"
            if 'EMA20' in df.columns and 'EMA50' in df.columns:
                ema20 = df['EMA20'].iloc[-1]
                ema50 = df['EMA50'].iloc[-1]
                
                if current_price > ema20 > ema50:
                    signal = "🟢"  # Восходящий тренд
                elif current_price < ema20 < ema50:
                    signal = "🔴"  # Нисходящий тренд
                elif ema20 > current_price > ema50:
                    signal = "🟡"  # Коррекция в тренде
            
            # Анализ RSI
            rsi_signal = ""
            if rsi:
                if rsi > 70:
                    rsi_signal = "🔥"  # Перекупленность
                elif rsi < 30:
                    rsi_signal = "❄️"   # Перепроданность
            
            results.append({
                'ticker': ticker,
                'price': current_price,
                'rsi': rsi,
                'signal': signal,
                'rsi_signal': rsi_signal
            })
        
        # Сортируем по силе сигнала
        results.sort(key=lambda x: (x['signal'] == "🟢", x['rsi'] or 50), reverse=True)
        
        # Формируем ответ
        response = "🚀 **Быстрое сканирование** (топ-12 акций)\n\n"
        
        for item in results[:8]:  # Показываем топ-8
            rsi_text = f"RSI:{item['rsi']:.0f}" if item['rsi'] else "RSI:N/A"
            response += f"{item['signal']}{item['rsi_signal']} **{item['ticker']}**: {item['price']:.2f} | {rsi_text}\n"
        
        response += "\n🟢 - Тренд вверх | 🔴 - Тренд вниз | 🟡 - Коррекция\n"
        response += "🔥 - Перекупленность | ❄️ - Перепроданность"
        
        await message.edit_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in scan_fast: {e}")
        await message.edit_text("❌ Ошибка при сканировании рынка")

@telegram_handler
@typing_action
@performance_monitor("rsi_top_optimized")
async def rsi_top_optimized(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Оптимизированный анализ RSI с батчевой обработкой
    """
    message = await update.message.reply_text("🔄 Анализирую RSI для всех акций... (оптимизированный режим)")
    
    try:
        tickers = get_all_tickers()
        
        # Разбиваем на батчи для оптимальной обработки
        batch_size = 25
        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        
        overbought = []
        oversold = []
        processed_count = 0
        
        for i, batch in enumerate(batches):
            # Обновляем прогресс
            if i % 2 == 0:  # Каждый второй батч
                progress = f"🔄 Обработано: {processed_count}/{len(tickers)} акций..."
                await message.edit_text(progress)
            
            # Получаем данные для батча (используя кэш)
            data_dict = optimized_moex_client.get_multiple_daily_data(batch, days=30)
            
            # Анализируем RSI
            for ticker, df in data_dict.items():
                if df.empty:
                    continue
                
                try:
                    df = analyze_indicators(df)
                    rsi = df['RSI'].iloc[-1]
                    
                    if pd.notna(rsi):
                        if rsi > 70:
                            overbought.append((ticker, rsi))
                        elif rsi < 30:
                            oversold.append((ticker, rsi))
                            
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing RSI for {ticker}: {e}")
        
        # Сортируем результаты
        overbought.sort(key=lambda x: x[1], reverse=True)
        oversold.sort(key=lambda x: x[1])
        
        # Формируем ответ
        response = f"📊 **RSI Анализ** ({processed_count} акций)\n\n"
        
        if overbought:
            response += "🔴 **Перекупленные (RSI > 70):**\n"
            for ticker, rsi in overbought[:10]:
                response += f"• **{ticker}**: {rsi:.0f}\n"
        
        if oversold:
            response += "\n🟢 **Перепроданные (RSI < 30):**\n"
            for ticker, rsi in oversold[:10]:
                response += f"• **{ticker}**: {rsi:.0f}\n"
        
        if not overbought and not oversold:
            response += "ℹ️ Нет акций в экстремальных зонах RSI"
        
        # Добавляем статистику
        cache_stats = cache.get_cache_stats()
        response += f"\n📈 Cache Hit Rate: {cache_stats['hit_rate']}"
        
        await message.edit_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in rsi_top_optimized: {e}")
        await message.edit_text("❌ Ошибка при анализе RSI")

@telegram_handler
async def warm_cache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Прогрев кэша популярных акций"""
    message = await update.message.reply_text("🔥 Прогреваю кэш...")
    
    try:
        popular_tickers = [
            "SBER", "GAZP", "LKOH", "YDEX", "MGNT", "ROSN", 
            "NVTK", "VTBR", "ALRS", "MTSS", "MOEX", "PIKK",
            "T", "NLMK", "CHMF", "PHOR", "RUAL", "GMKN"
        ]
        
        # Запускаем прогрев в фоне
        optimized_moex_client.warm_up_cache(popular_tickers)
        
        await message.edit_text(
            f"✅ Кэш прогрет для {len(popular_tickers)} популярных акций\n"
            "Теперь команды будут работать быстрее!"
        )
        
    except Exception as e:
        logger.error(f"Error warming cache: {e}")
        await message.edit_text("❌ Ошибка при прогреве кэша")

@telegram_handler
async def clear_cache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очистка кэша"""
    try:
        cache.clear_all()
        
        await update.message.reply_text(
            "🗑️ Кэш очищен!\n"
            "Следующие запросы будут медленнее, но с актуальными данными."
        )
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        await update.message.reply_text("❌ Ошибка при очистке кэша")

@telegram_handler
@typing_action
async def volume_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сканирование аномальных объемов"""
    message = await update.message.reply_text("🔍 Ищу аномальные объемы...")
    
    try:
        # Используем средний размер набора для оптимизации
        scan_tickers = [
            "SBER", "GAZP", "LKOH", "YDEX", "MGNT", "ROSN", "NVTK", "VTBR", 
            "ALRS", "MTSS", "MOEX", "PIKK", "T", "NLMK", "CHMF", "PHOR",
            "RUAL", "GMKN", "TATN", "SNGS"
        ]
        
        data_dict = optimized_moex_client.get_multiple_daily_data(scan_tickers, days=20)
        
        volume_alerts = []
        
        for ticker, df in data_dict.items():
            if df.empty:
                continue
                
            try:
                df = analyze_indicators(df)
                
                # Проверяем аномальные объемы за последние 3 дня
                recent_anomalies = df[df['Anomaly']].tail(3)
                
                if not recent_anomalies.empty:
                    latest_anomaly = recent_anomalies.iloc[-1]
                    volume_multiplier = latest_anomaly['Volume_Multiplier']
                    price = latest_anomaly['close']
                    date = latest_anomaly.name.strftime('%Y-%m-%d')
                    
                    volume_alerts.append({
                        'ticker': ticker,
                        'multiplier': volume_multiplier,
                        'price': price,
                        'date': date
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing volume for {ticker}: {e}")
        
        # Сортируем по множителю объема
        volume_alerts.sort(key=lambda x: x['multiplier'], reverse=True)
        
        response = "🔍 **Аномальные объемы** (последние 3 дня)\n\n"
        
        if volume_alerts:
            for alert in volume_alerts[:10]:
                response += f"📊 **{alert['ticker']}**: {alert['multiplier']:.1f}x объем\n"
                response += f"   Цена: {alert['price']:.2f} | {alert['date']}\n\n"
        else:
            response += "ℹ️ Аномальных объемов не обнаружено"
        
        await message.edit_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in volume_scan: {e}")
        await message.edit_text("❌ Ошибка при сканировании объемов")

@telegram_handler
@typing_action
async def delta_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сканирование дельта-сигналов по денежному потоку"""
    message = await update.message.reply_text("🔍 Ищу сигналы по дельте денежного потока...")
    
    try:
        # Используем популярные тикеры для быстрого сканирования
        scan_tickers = [
            "SBER", "GAZP", "LKOH", "YDEX", "MGNT", "ROSN", "NVTK", "VTBR", 
            "ALRS", "MTSS", "MOEX", "PIKK", "T", "NLMK", "CHMF", "PHOR",
            "RUAL", "GMKN", "TATN", "SNGS", "X5", "OZON"
        ]
        
        # Сканируем возможности за 10 дней
        opportunities = delta_analyzer.scan_delta_opportunities(scan_tickers, days=10)
        
        response = "💰 **Дельта-сигналы денежного потока** (10 дней)\n\n"
        
        if opportunities:
            response += "🟢 **Найденные возможности:**\n"
            response += "`Тикер  Δ%   Поток млн  EMA  SMA30W`\n"
            
            for opp in opportunities[:8]:  # Топ-8 сигналов
                ema_signal = "🟢" if opp['ema20x50_long'] else "🔴"
                sma_signal = "🟢" if opp['price_above_sma30_weekly'] else "🔴"
                
                response += f"`{opp['ticker']:<6} {opp['price_change_pct']:>4.1f}% {opp['ad_delta_millions']:>8.0f}    {ema_signal}   {sma_signal}`\n"
            
            response += "\n🟢 - Позитивный сигнал | 🔴 - Негативный сигнал\n"
            response += "EMA - тренд EMA20x50 | SMA30W - позиция относительно SMA30 недельной"
        else:
            response += "ℹ️ Сильных дельта-сигналов не найдено\n"
            response += "Попробуйте позже или используйте /delta для анализа конкретной акции"
        
        await message.edit_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in delta_scan: {e}")
        await message.edit_text("❌ Ошибка при сканировании дельта-сигналов")