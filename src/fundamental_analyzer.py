"""
Модуль для фундаментального анализа акций
Автор: Мекеда Богдан Сергеевич
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """Класс для анализа фундаментальных показателей акций"""
    
    def __init__(self):
        # Веса для различных показателей в итоговом скоре
        self.weights = {
            'valuation': 0.3,  # Оценка стоимости
            'profitability': 0.25,  # Прибыльность
            'growth': 0.2,  # Рост
            'financial_health': 0.15,  # Финансовое здоровье
            'dividend': 0.1  # Дивиденды
        }
    
    def analyze_stock(self, fundamental_data: Dict) -> Dict:
        """
        Анализ одной акции
        
        Args:
            fundamental_data: Фундаментальные данные акции
            
        Returns:
            Словарь с результатами анализа
        """
        if not fundamental_data:
            return {}
        
        scores = {}
        
        # Анализ оценки стоимости
        scores['valuation'] = self._analyze_valuation(fundamental_data)
        
        # Анализ прибыльности
        scores['profitability'] = self._analyze_profitability(fundamental_data)
        
        # Анализ роста
        scores['growth'] = self._analyze_growth(fundamental_data)
        
        # Анализ финансового здоровья
        scores['financial_health'] = self._analyze_financial_health(fundamental_data)
        
        # Анализ дивидендов
        scores['dividend'] = self._analyze_dividend(fundamental_data)
        
        # Расчет общего скора
        total_score = sum(scores[key] * self.weights[key] for key in scores)
        
        # Определение рейтинга
        rating = self._get_rating(total_score)
        
        return {
            'ticker': fundamental_data.get('ticker', 'N/A'),
            'company_name': fundamental_data.get('company_name', 'N/A'),
            'sector': fundamental_data.get('sector', 'N/A'),
            'scores': scores,
            'total_score': round(total_score, 2),
            'rating': rating,
            'recommendation': self._get_recommendation(total_score, fundamental_data)
        }
    
    def _analyze_valuation(self, data: Dict) -> float:
        """Анализ оценки стоимости"""
        score = 5.0  # Базовый скор
        
        # P/E ratio
        pe = data.get('pe_ratio')
        if pe and pe > 0:
            if pe < 15:
                score += 2
            elif pe < 25:
                score += 1
            elif pe > 40:
                score -= 2
        
        # P/B ratio
        pb = data.get('price_to_book')
        if pb and pb > 0:
            if pb < 1.5:
                score += 1.5
            elif pb < 3:
                score += 0.5
            elif pb > 5:
                score -= 1.5
        
        # PEG ratio
        peg = data.get('peg_ratio')
        if peg and peg > 0:
            if peg < 1:
                score += 1.5
            elif peg > 2:
                score -= 1
        
        return max(0, min(10, score))
    
    def _analyze_profitability(self, data: Dict) -> float:
        """Анализ прибыльности"""
        score = 5.0
        
        # ROE
        roe = data.get('roe')
        if roe and roe > 0:
            if roe > 0.15:
                score += 2
            elif roe > 0.10:
                score += 1
            elif roe < 0.05:
                score -= 1
        
        # ROA
        roa = data.get('roa')
        if roa and roa > 0:
            if roa > 0.08:
                score += 1.5
            elif roa > 0.05:
                score += 0.5
            elif roa < 0.02:
                score -= 1
        
        # Profit Margin
        margin = data.get('profit_margin')
        if margin and margin > 0:
            if margin > 0.15:
                score += 1.5
            elif margin > 0.10:
                score += 1
            elif margin < 0.05:
                score -= 1
        
        return max(0, min(10, score))
    
    def _analyze_growth(self, data: Dict) -> float:
        """Анализ роста"""
        score = 5.0
        
        # Revenue Growth
        revenue_growth = data.get('revenue_growth')
        if revenue_growth:
            if revenue_growth > 0.15:
                score += 2.5
            elif revenue_growth > 0.10:
                score += 1.5
            elif revenue_growth > 0.05:
                score += 0.5
            elif revenue_growth < 0:
                score -= 2
        
        # Анализ квартальной динамики прибыли
        earnings = data.get('quarterly_earnings', [])
        if len(earnings) >= 4:
            try:
                # Проверяем рост за последние кварталы
                recent_growth = []
                for i in range(1, min(4, len(earnings))):
                    if earnings[i-1] and earnings[i]:
                        growth = (earnings[i] - earnings[i-1]) / abs(earnings[i-1])
                        recent_growth.append(growth)
                
                if recent_growth:
                    avg_growth = np.mean(recent_growth)
                    if avg_growth > 0.1:
                        score += 1.5
                    elif avg_growth > 0:
                        score += 0.5
                    elif avg_growth < -0.1:
                        score -= 1.5
            except:
                pass
        
        return max(0, min(10, score))
    
    def _analyze_financial_health(self, data: Dict) -> float:
        """Анализ финансового здоровья"""
        score = 5.0
        
        # Debt to Equity
        de_ratio = data.get('debt_to_equity')
        if de_ratio is not None:
            if de_ratio < 0.3:
                score += 2
            elif de_ratio < 0.6:
                score += 1
            elif de_ratio > 1.5:
                score -= 2
        
        # Current Ratio
        current_ratio = data.get('current_ratio')
        if current_ratio and current_ratio > 0:
            if current_ratio > 2:
                score += 1.5
            elif current_ratio > 1.5:
                score += 1
            elif current_ratio < 1:
                score -= 2
        
        # Free Cash Flow
        fcf = data.get('free_cash_flow', 0)
        if fcf > 0:
            score += 1.5
        elif fcf < 0:
            score -= 1
        
        return max(0, min(10, score))
    
    def _analyze_dividend(self, data: Dict) -> float:
        """Анализ дивидендов"""
        score = 5.0
        
        div_yield = data.get('dividend_yield', 0)
        if div_yield > 0:
            if 0.02 <= div_yield <= 0.06:  # 2-6% считается хорошим
                score += 2
            elif 0.01 <= div_yield < 0.02:
                score += 1
            elif div_yield > 0.08:  # Слишком высокий может быть подозрительным
                score -= 1
        else:
            # Не все компании платят дивиденды, особенно растущие
            score = 5.0
        
        return max(0, min(10, score))
    
    def _get_rating(self, score: float) -> str:
        """Определение текстового рейтинга"""
        if score >= 8:
            return "Сильная покупка"
        elif score >= 7:
            return "Покупка"
        elif score >= 6:
            return "Умеренная покупка"
        elif score >= 5:
            return "Держать"
        elif score >= 4:
            return "Слабая продажа"
        else:
            return "Продажа"
    
    def _get_recommendation(self, score: float, data: Dict) -> str:
        """Генерация рекомендации"""
        ticker = data.get('ticker', 'акция')
        
        if score >= 7.5:
            return f"{ticker} показывает отличные фундаментальные показатели. Рекомендуется к покупке."
        elif score >= 6.5:
            return f"{ticker} имеет хорошие показатели. Можно рассмотреть для включения в портфель."
        elif score >= 5.5:
            return f"{ticker} показывает умеренные результаты. Требует дополнительного анализа."
        elif score >= 4.5:
            return f"{ticker} имеет некоторые слабые места в фундаментальных показателях."
        else:
            return f"{ticker} показывает слабые фундаментальные показатели. Не рекомендуется к покупке."
    
    def analyze_portfolio(self, fundamental_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Анализ портфеля акций
        
        Args:
            fundamental_data: Словарь с фундаментальными данными по тикерам
            
        Returns:
            DataFrame с результатами анализа
        """
        results = []
        
        for ticker, data in fundamental_data.items():
            analysis = self.analyze_stock(data)
            if analysis:
                results.append(analysis)
        
        if not results:
            return pd.DataFrame()
        
        # Создаем DataFrame
        df = pd.DataFrame(results)
        
        # Сортируем по общему скору
        df = df.sort_values('total_score', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_sector_analysis(self, df: pd.DataFrame) -> Dict:
        """Анализ по секторам"""
        if df.empty:
            return {}
        
        sector_stats = df.groupby('sector').agg({
            'total_score': ['mean', 'count'],
            'ticker': 'count'
        }).round(2)
        
        return sector_stats.to_dict()