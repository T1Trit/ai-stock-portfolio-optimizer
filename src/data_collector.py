"""
Модуль для сбора финансовых данных
Автор: Мекеда Богдан Сергеевич
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Класс для сбора и обработки финансовых данных"""
    
    def __init__(self, tickers: List[str], period: str = "2y"):
        """
        Инициализация сборщика данных
        
        Args:
            tickers: Список тикеров акций
            period: Период данных (1y, 2y, 5y, max)
        """
        self.tickers = tickers
        self.period = period
        
    def fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Получение исторических данных для одного тикера
        
        Args:
            ticker: Тикер акции
            
        Returns:
            DataFrame с историческими данными
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=self.period)
            
            if data.empty:
                logger.warning(f"Нет данных для тикера {ticker}")
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
            data: DataFrame с ценами
            
        Returns:
            DataFrame с добавленными индикаторами
        """
        # Скользящие средние
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # EMA
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (std * 2)
        
        # Волатильность
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
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