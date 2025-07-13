from telegram import Update
from telegram.ext import ContextTypes
import logging

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /start (перенесено из main.py)
    """
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
    """
    Команда /help
    """
    await start(update, context)