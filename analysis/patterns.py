import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import List, Tuple

def find_levels(df: pd.DataFrame) -> List[Tuple]:
    """
    Поиск уровней поддержки и сопротивления (перенесено из main.py)
    """
    if df.empty:
        return []
    
    levels = []
    closes = df['close'].values
    local_max = argrelextrema(closes, np.greater)[0]
    local_min = argrelextrema(closes, np.less)[0]

    extrema = sorted(
        [(i, closes[i]) for i in np.concatenate((local_max, local_min))], 
        key=lambda x: x[1]
    )
    
    if len(extrema) > 0:
        grouped = pd.Series([round(p[1], 1) for p in extrema]).value_counts()
        strong_levels = grouped[grouped > 1].index.tolist()
        
        for level in strong_levels:
            for i, val in extrema:
                if abs(val - level) < 0.5:
                    levels.append((df.index[i], val))
                    break
    
    return levels

def detect_double_patterns(df: pd.DataFrame) -> List[Tuple]:
    """
    Поиск паттернов двойной вершины и дна (перенесено из main.py)
    """
    if df.empty or len(df) < 5:
        return []
    
    closes = df['close'].values
    patterns = []
    
    for i in range(2, len(closes) - 2):
        # Двойная вершина
        if (closes[i-2] < closes[i-1] < closes[i] and 
            closes[i] > closes[i+1] > closes[i+2]):
            patterns.append(('Double Top', df.index[i], closes[i]))
        
        # Двойное дно
        if (closes[i-2] > closes[i-1] > closes[i] and 
            closes[i] < closes[i+1] < closes[i+2]):
            patterns.append(('Double Bottom', df.index[i], closes[i]))
    
    return patterns

def detect_volume_anomalies(df: pd.DataFrame) -> List[Tuple]:
    """
    Обнаружение аномалий объема
    """
    if df.empty or 'Volume_Mean' not in df.columns or 'Volume_Multiplier' not in df.columns:
        return []
    
    anomalies = []
    for idx in df[df['Anomaly']].index:
        volume_ratio = df.loc[idx, 'Volume_Multiplier']
        price = df.loc[idx, 'close']
        anomalies.append((idx, price, volume_ratio))
    
    return anomalies

def analyze_trend(df: pd.DataFrame) -> str:
    """
    Анализ тренда на основе EMA
    """
    if df.empty or len(df) < 50:
        return "Недостаточно данных"
    
    # Проверяем последние значения EMA
    if 'EMA20' not in df.columns or 'EMA50' not in df.columns:
        return "Недостаточно данных"
    
    current_price = df['close'].iloc[-1]
    ema20 = df['EMA20'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    
    if current_price > ema20 > ema50:
        return "Восходящий тренд"
    elif current_price < ema20 < ema50:
        return "Нисходящий тренд"
    else:
        return "Боковое движение"

def get_support_resistance_levels(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
    """
    Определение ближайших уровней поддержки и сопротивления
    """
    if df.empty or len(df) < window:
        return None, None
    
    recent_data = df.tail(window)
    current_price = df['close'].iloc[-1]
    
    # Поддержка - максимальная цена ниже текущей
    support_levels = recent_data[recent_data['close'] < current_price]['close']
    support = support_levels.max() if not support_levels.empty else None
    
    # Сопротивление - минимальная цена выше текущей  
    resistance_levels = recent_data[recent_data['close'] > current_price]['close']
    resistance = resistance_levels.min() if not resistance_levels.empty else None
    
    return support, resistance