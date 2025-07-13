import matplotlib
matplotlib.use('Agg')  # Безголовый режим для matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Optional, List, Tuple
from analysis.patterns import find_levels, detect_double_patterns, detect_volume_anomalies

def plot_stock_chart(df: pd.DataFrame, ticker: str, 
                    levels: List[Tuple] = None, 
                    patterns: List[Tuple] = None) -> Optional[str]:
    """
    Построение графика акции (перенесено из main.py plot_stock)
    """
    if df.empty:
        return None
    
    try:
        plt.figure(figsize=(12, 6))
        
        # Основная линия цены
        plt.plot(df.index, df['close'], label='Цена', color='blue', linewidth=2)

        # EMA линии
        if 'EMA9' in df.columns:
            plt.plot(df.index, df['EMA9'], label='EMA9', linestyle='--', alpha=0.7, color='orange')
        if 'EMA20' in df.columns:
            plt.plot(df.index, df['EMA20'], label='EMA20', linestyle='--', alpha=0.7, color='green')
        if 'EMA50' in df.columns:
            plt.plot(df.index, df['EMA50'], label='EMA50', linestyle='--', alpha=0.7, color='red')
        if 'EMA100' in df.columns:
            plt.plot(df.index, df['EMA100'], label='EMA100', linestyle='--', alpha=0.7, color='purple')
        if 'EMA200' in df.columns:
            plt.plot(df.index, df['EMA200'], label='EMA200', linestyle='--', alpha=0.7, color='brown')

        # Аномальные объемы
        anomalies = detect_volume_anomalies(df)
        for idx, price, volume_ratio in anomalies:
            plt.scatter(idx, price, color='red', s=50, alpha=0.7)
            plt.text(idx, price, f"{volume_ratio:.1f}x", 
                    color='red', fontsize=8, ha='left', va='bottom')

        # Уровни поддержки/сопротивления
        if levels:
            for date, price in levels:
                plt.axhline(price, linestyle='--', alpha=0.3, color='gray')

        # Паттерны
        if patterns:
            plotted_top = False
            plotted_bottom = False
            
            for name, date, price in patterns:
                if name == 'Double Top' and not plotted_top:
                    plt.scatter(date, price, color='red', marker='v', s=100, 
                              label='Double Top', alpha=0.7)
                    plotted_top = True
                elif name == 'Double Bottom' and not plotted_bottom:
                    plt.scatter(date, price, color='green', marker='^', s=100, 
                              label='Double Bottom', alpha=0.7)
                    plotted_bottom = True
                elif name == 'Double Top':
                    plt.scatter(date, price, color='red', marker='v', s=100, alpha=0.7)
                elif name == 'Double Bottom':
                    plt.scatter(date, price, color='green', marker='^', s=100, alpha=0.7)

        plt.title(f'{ticker} - Технический анализ')
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохранение файла
        filename = f'{ticker}_chart.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"Ошибка при создании графика для {ticker}: {e}")
        return None

def plot_stan_chart(df: pd.DataFrame, ticker: str) -> Optional[str]:
    """
    Построение графика по методу Стэна Вайнштейна (перенесено из main.py)
    """
    if df.empty:
        return None
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Основные данные
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Цена', color='blue', linewidth=2)
        
        # SMA 30
        if 'SMA30' in df.columns:
            plt.plot(df.index, df['SMA30'], label='SMA30', color='red', linewidth=2)
        
        # EMA
        if 'EMA200' in df.columns:
            plt.plot(df.index, df['EMA200'], label='EMA200', color='orange', linewidth=1)
        
        plt.title(f'{ticker} - Анализ по методу Стэна Вайнштейна')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Объем
        plt.subplot(2, 1, 2)
        plt.bar(df.index, df['volume'], alpha=0.7, color='lightblue')
        
        # Выделяем аномальные объемы
        anomalies = detect_volume_anomalies(df)
        for idx, _, volume_ratio in anomalies:
            plt.bar(idx, df.loc[idx, 'volume'], color='red', alpha=0.8)
        
        plt.ylabel('Объем')
        plt.xlabel('Дата')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение файла
        filename = f'{ticker}_stan_chart.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"Ошибка при создании Stan графика для {ticker}: {e}")
        return None

def plot_rsi_chart(df: pd.DataFrame, ticker: str) -> Optional[str]:
    """
    Построение графика RSI
    """
    if df.empty or 'RSI' not in df.columns:
        return None
    
    try:
        plt.figure(figsize=(12, 8))
        
        # График цены
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Цена', color='blue', linewidth=2)
        plt.title(f'{ticker} - Анализ RSI')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График RSI
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
        plt.axhline(70, color='red', linestyle='--', alpha=0.7, label='Перекупленность')
        plt.axhline(30, color='green', linestyle='--', alpha=0.7, label='Перепроданность')
        plt.axhline(50, color='gray', linestyle='-', alpha=0.5)
        
        plt.ylabel('RSI')
        plt.xlabel('Дата')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение файла
        filename = f'{ticker}_rsi_chart.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"Ошибка при создании RSI графика для {ticker}: {e}")
        return None

def create_summary_chart(ticker: str, df: pd.DataFrame) -> Optional[str]:
    """
    Создание сводного графика с основными индикаторами
    """
    if df.empty:
        return None
    
    try:
        # Анализируем данные
        from analysis.indicators import analyze_indicators
        df = analyze_indicators(df)
        
        levels = find_levels(df)
        patterns = detect_double_patterns(df)
        
        # Создаем график
        chart_file = plot_stock_chart(df, ticker, levels, patterns)
        
        return chart_file
        
    except Exception as e:
        print(f"Ошибка при создании сводного графика для {ticker}: {e}")
        return None