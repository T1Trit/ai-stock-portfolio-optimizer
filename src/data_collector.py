"""
Модуль для сбора финансовых данных
Автор: Мекеда Богдан Сергеевич
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    Класс для сбора и предобработки финансовых данных
    """
    
    def __init__(self, tickers: List[str], period: str = "5y"):
        """
        Инициализация коллектора данных
        
        Args:
            tickers: Список тикеров акций
            period: Период для загрузки данных (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        self.tickers = tickers
        self.period = period
        
    def fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Получение исторических данных для одной акции
        
        Args:
            ticker: Тикер акции
            
        Returns:
            DataFrame с историческими данными
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=self.period)
            
            if data.empty:
                logger.warning(f"Нет данных для {ticker}")
                return pd.DataFrame()
            
            # Добавляем технические индикаторы
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка при получении данных для {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление технических индикаторов
        
        Args:
            data: DataFrame с базовыми данными
            
        Returns:
            DataFrame с добавленными индикаторами
        """
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = rolling_mean + (rolling_std * 2)
        data['BB_Lower'] = rolling_mean - (rolling_std * 2)
        data['BB_Middle'] = rolling_mean
        
        # Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(window=30).std()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        return data
    
    def fetch_fundamental_data(self, ticker: str) -> Dict:
        """
        Получение фундаментальных показателей компании
        
        Args:
            ticker: Тикер акции
            
        Returns:
            Словарь с фундаментальными показателями
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Извлекаем ключевые показатели
            fundamentals = {
                'ticker': ticker,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'dividend_yield': info.get('dividendYield', 0),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'profit_margin': info.get('profitMargins', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'current_ratio': info.get('currentRatio', None),
                'free_cash_flow': info.get('freeCashflow', 0),
                'beta': info.get('beta', 1),
                'target_price': info.get('targetMeanPrice', None),
                'recommendation': info.get('recommendationKey', 'none')
            }
            
            # Получаем финансовые отчеты
            try:
                fundamentals['quarterly_earnings'] = stock.quarterly_earnings.to_dict('records') if not stock.quarterly_earnings.empty else []
                fundamentals['quarterly_revenue'] = stock.quarterly_financials.loc['Total Revenue'].to_dict() if 'Total Revenue' in stock.quarterly_financials.index else {}
            except:
                fundamentals['quarterly_earnings'] = []
                fundamentals['quarterly_revenue'] = {}
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Ошибка при получении фундаментальных данных для {ticker}: {str(e)}")
            return {}
    
    def fetch_all_data(self, n_workers: int = 5) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        Параллельная загрузка данных для всех тикеров
        
        Args:
            n_workers: Количество потоков для параллельной загрузки
            
        Returns:
            Кортеж из словарей с историческими и фундаментальными данными
        """
        historical_data = {}
        fundamental_data = {}
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Запускаем задачи для исторических данных
            hist_futures = {executor.submit(self.fetch_stock_data, ticker): ticker 
                           for ticker in self.tickers}
            
            # Запускаем задачи для фундаментальных данных
            fund_futures = {executor.submit(self.fetch_fundamental_data, ticker): ticker 
                           for ticker in self.tickers}
            
            # Собираем результаты для исторических данных
            for future in as_completed(hist_futures):
                ticker = hist_futures[future]
                try:
                    data = future.result()
                    if not data.empty:
                        historical_data[ticker] = data
                except Exception as e:
                    logger.error(f"Ошибка при получении исторических данных для {ticker}: {str(e)}")
            
            # Собираем результаты для фундаментальных данных
            for future in as_completed(fund_futures):
                ticker = fund_futures[future]
                try:
                    data = future.result()
                    if data:
                        fundamental_data[ticker] = data
                except Exception as e:
                    logger.error(f"Ошибка при получении фундаментальных данных для {ticker}: {str(e)}")
        
        logger.info(f"Загружено данных: {len(historical_data)} исторических, {len(fundamental_data)} фундаментальных")
        return historical_data, fundamental_data