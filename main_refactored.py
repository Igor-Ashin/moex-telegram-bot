#!/usr/bin/env python3
"""
Refactored MOEX Telegram Bot - новая модульная архитектура
"""

import os
import logging
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler
from telegram.ext import ConversationHandler, MessageHandler, filters

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импорты конфигурации
from config.settings import settings

# Импорты обработчиков
from bot.handlers.basic import start, help_command

# Импорты для будущего использования (когда создадим остальные модули)
# from bot.handlers.analysis import chart_hv, stan, rsi_top
# from bot.handlers.trading import cross_ema20x50, high_volume, moneyflow
# from bot.handlers.callbacks import handle_callback
# from bot.handlers.conversations import delta_conversation, moneyflow_conversation

def setup_handlers(app):
    """
    Настройка обработчиков команд
    """
    # Основные команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    
    # TODO: Добавить остальные обработчики по мере создания модулей
    # app.add_handler(CommandHandler("chart_hv", chart_hv))
    # app.add_handler(CommandHandler("stan", stan))
    # app.add_handler(CommandHandler("rsi_top", rsi_top))
    # app.add_handler(CommandHandler("cross_ema20x50", cross_ema20x50))
    # app.add_handler(CommandHandler("high_volume", high_volume))
    # app.add_handler(CallbackQueryHandler(handle_callback))
    # app.add_handler(delta_conversation)
    # app.add_handler(moneyflow_conversation)
    
    logger.info("Handlers registered successfully")

def main():
    """
    Основная функция запуска бота
    """
    try:
        # Проверяем конфигурацию
        if not settings.telegram_token:
            logger.error("TELEGRAM_TOKEN not found in environment variables")
            return
        
        logger.info(f"Starting bot with webhook URL: {settings.webhook_url}")
        
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
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise

if __name__ == '__main__':
    main()