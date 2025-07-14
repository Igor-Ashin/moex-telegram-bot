import pandas as pd
import numpy as np
from typing import Optional

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисление RSI (перенесено из main.py)
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

def calculate_money_ad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление Money Flow (A/D Line) - актуализированная версия из main.py
    """
    df = df.copy()
    df['TYP'] = (df['high'] + df['low'] + df['close']) / 3
    df['CLV'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['CLV'] = df['CLV'].fillna(0)
    df['money_flow'] = df['CLV'] * df['volume'] * df['TYP']
    df['money_ad'] = df['money_flow'].cumsum()
    return df

def analyze_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление всех индикаторов к DataFrame (перенесено из main.py)
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
    df = calculate_money_ad(df)
    
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

def find_sma30_crossover(df: pd.DataFrame, ticker: str, days: int = 7) -> Optional[str]:
    """
    Поиск пересечения цены с SMA30 (перенесено из main.py)
    """
    if df.empty or len(df) < 35:
        return None
    
    # Фильтр по минимальному обороту
    from config.settings import settings
    filter_turnover_series = df['volume'].iloc[-10:] * df['close'].iloc[-10:]
    filter_avg_turnover = filter_turnover_series.mean()
    
    if filter_avg_turnover < settings.min_turnover:
        return None
    
    # Вычисляем SMA30
    df['SMA30'] = calculate_sma(df, 30)
    
    # Проверяем, что текущая цена выше SMA30
    current_close = df['close'].iloc[-1]
    current_sma30 = df['SMA30'].iloc[-1]
    
    if current_close <= current_sma30:
        return None
    
    # Берём последние days дней для поиска пересечений
    recent_df = df.tail(days + 1)
    
    for i in range(1, len(recent_df)):
        prev_close = recent_df['close'].iloc[i-1]
        curr_close = recent_df['close'].iloc[i]
        prev_sma30 = recent_df['SMA30'].iloc[i-1]
        curr_sma30 = recent_df['SMA30'].iloc[i]
        
        # Пересечение снизу вверх
        if prev_close <= prev_sma30 and curr_close > curr_sma30:
            return recent_df.index[i].strftime('%Y-%m-%d')
    
    return None