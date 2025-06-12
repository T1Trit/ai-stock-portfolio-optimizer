"""
Модуль для фундаментального анализа компаний
Автор: Мекеда Богдан Сергеевич
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """
    Класс для проведения фундаментального анализа компаний
    """
    
    def __init__(self, fundamental_data: Dict[str, Dict]):
        """
        Инициализация анализатора
        
        Args:
            fundamental_data: Словарь с фундаментальными данными компаний
        """
        self.fundamental_data = fundamental_data
        
        # Пороговые значения для оценки
        self.THRESHOLDS = {
            'pe_ratio': {
                'excellent': 15,
                'good': 20,
                'fair': 25,
                'poor': 30
            },
            'roe': {
                'excellent': 0.15,
                'good': 0.12,
                'fair': 0.08,
                'poor': 0.05
            },
            'debt_to_equity': {
                'excellent': 0.3,
                'good': 0.5,
                'fair': 0.8,
                'poor': 1.2
            },
            'current_ratio': {
                'excellent': 2.0,
                'good': 1.5,
                'fair': 1.2,
                'poor': 1.0
            },
            'revenue_growth': {
                'excellent': 0.20,
                'good': 0.15,
                'fair': 0.10,
                'poor': 0.05
            },
            'profit_margin': {
                'excellent': 0.20,
                'good': 0.15,
                'fair': 0.10,
                'poor': 0.05
            }
        }
        
        # Веса для итогового скора
        self.WEIGHTS = {
            'pe_score': 0.20,
            'roe_score': 0.20,
            'growth_score': 0.15,
            'profitability_score': 0.15,
            'financial_health_score': 0.15,
            'valuation_score': 0.10,
            'dividend_score': 0.05
        }
    
    def calculate_scores(self) -> pd.DataFrame:
        """
        Расчет скоров для всех компаний
        
        Returns:
            DataFrame с рассчитанными скорами
        """
        results = []
        
        for ticker, data in self.fundamental_data.items():
            try:
                # Расчет отдельных скоров
                pe_score = self._calculate_pe_score(data.get('pe_ratio'))
                roe_score = self._calculate_roe_score(data.get('roe'))
                growth_score = self._calculate_growth_score(data.get('revenue_growth'))
                profitability_score = self._calculate_profitability_score(
                    data.get('profit_margin'), 
                    data.get('roa')
                )
                financial_health_score = self._calculate_financial_health_score(
                    data.get('debt_to_equity'), 
                    data.get('current_ratio')
                )
                valuation_score = self._calculate_valuation_score(
                    data.get('price_to_book'), 
                    data.get('peg_ratio')
                )
                dividend_score = self._calculate_dividend_score(data.get('dividend_yield'))
                
                # Итоговый скор
                total_score = (
                    pe_score * self.WEIGHTS['pe_score'] +
                    roe_score * self.WEIGHTS['roe_score'] +
                    growth_score * self.WEIGHTS['growth_score'] +
                    profitability_score * self.WEIGHTS['profitability_score'] +
                    financial_health_score * self.WEIGHTS['financial_health_score'] +
                    valuation_score * self.WEIGHTS['valuation_score'] +
                    dividend_score * self.WEIGHTS['dividend_score']
                ) * 100
                
                # Определение рейтинга
                rating = self._get_rating(total_score)
                
                results.append({
                    'ticker': ticker,
                    'company_name': data.get('company_name', 'N/A'),
                    'sector': data.get('sector', 'N/A'),
                    'market_cap': data.get('market_cap', 0),
                    'pe_score': pe_score * 100,
                    'roe_score': roe_score * 100,
                    'growth_score': growth_score * 100,
                    'profitability_score': profitability_score * 100,
                    'financial_health_score': financial_health_score * 100,
                    'valuation_score': valuation_score * 100,
                    'dividend_score': dividend_score * 100,
                    'total_score': total_score,
                    'rating': rating,
                    'pe_ratio': data.get('pe_ratio'),
                    'roe': data.get('roe'),
                    'revenue_growth': data.get('revenue_growth'),
                    'debt_to_equity': data.get('debt_to_equity'),
                    'dividend_yield': data.get('dividend_yield')
                })
                
            except Exception as e:
                logger.error(f"Ошибка при расчете скоров для {ticker}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        return df.sort_values('total_score', ascending=False)
    
    def _calculate_pe_score(self, pe_ratio: float) -> float:
        """Расчет скора на основе P/E ratio"""
        if pe_ratio is None or pe_ratio <= 0:
            return 0.5
        
        thresholds = self.THRESHOLDS['pe_ratio']
        
        # Для P/E меньше - лучше
        if pe_ratio <= thresholds['excellent']:
            return 1.0
        elif pe_ratio <= thresholds['good']:
            return 0.8
        elif pe_ratio <= thresholds['fair']:
            return 0.6
        elif pe_ratio <= thresholds['poor']:
            return 0.4
        else:
            return 0.2
    
    def _calculate_roe_score(self, roe: float) -> float:
        """Расчет скора на основе ROE"""
        if roe is None:
            return 0.5
        
        thresholds = self.THRESHOLDS['roe']
        
        if roe >= thresholds['excellent']:
            return 1.0
        elif roe >= thresholds['good']:
            return 0.8
        elif roe >= thresholds['fair']:
            return 0.6
        elif roe >= thresholds['poor']:
            return 0.4
        else:
            return 0.2
    
    def _calculate_growth_score(self, revenue_growth: float) -> float:
        """Расчет скора на основе роста выручки"""
        if revenue_growth is None:
            return 0.5
        
        thresholds = self.THRESHOLDS['revenue_growth']
        
        if revenue_growth >= thresholds['excellent']:
            return 1.0
        elif revenue_growth >= thresholds['good']:
            return 0.8
        elif revenue_growth >= thresholds['fair']:
            return 0.6
        elif revenue_growth >= thresholds['poor']:
            return 0.4
        else:
            return 0.2
    
    def _calculate_profitability_score(self, profit_margin: float, roa: float) -> float:
        """Расчет скора прибыльности"""
        scores = []
        
        if profit_margin is not None:
            thresholds = self.THRESHOLDS['profit_margin']
            if profit_margin >= thresholds['excellent']:
                scores.append(1.0)
            elif profit_margin >= thresholds['good']:
                scores.append(0.8)
            elif profit_margin >= thresholds['fair']:
                scores.append(0.6)
            elif profit_margin >= thresholds['poor']:
                scores.append(0.4)
            else:
                scores.append(0.2)
        
        if roa is not None:
            # ROA > 10% - отлично, > 5% - хорошо
            if roa > 0.10:
                scores.append(1.0)
            elif roa > 0.05:
                scores.append(0.7)
            elif roa > 0:
                scores.append(0.4)
            else:
                scores.append(0.2)
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_financial_health_score(self, debt_to_equity: float, current_ratio: float) -> float:
        """Расчет скора финансового здоровья"""
        scores = []
        
        if debt_to_equity is not None:
            thresholds = self.THRESHOLDS['debt_to_equity']
            # Для debt_to_equity меньше - лучше
            if debt_to_equity <= thresholds['excellent']:
                scores.append(1.0)
            elif debt_to_equity <= thresholds['good']:
                scores.append(0.8)
            elif debt_to_equity <= thresholds['fair']:
                scores.append(0.6)
            elif debt_to_equity <= thresholds['poor']:
                scores.append(0.4)
            else:
                scores.append(0.2)
        
        if current_ratio is not None:
            thresholds = self.THRESHOLDS['current_ratio']
            if current_ratio >= thresholds['excellent']:
                scores.append(1.0)
            elif current_ratio >= thresholds['good']:
                scores.append(0.8)
            elif current_ratio >= thresholds['fair']:
                scores.append(0.6)
            elif current_ratio >= thresholds['poor']:
                scores.append(0.4)
            else:
                scores.append(0.2)
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_valuation_score(self, price_to_book: float, peg_ratio: float) -> float:
        """Расчет скора оценки стоимости"""
        scores = []
        
        if price_to_book is not None and price_to_book > 0:
            # P/B < 1 - потенциально недооценена
            if price_to_book < 1:
                scores.append(1.0)
            elif price_to_book < 2:
                scores.append(0.8)
            elif price_to_book < 3:
                scores.append(0.6)
            elif price_to_book < 5:
                scores.append(0.4)
            else:
                scores.append(0.2)
        
        if peg_ratio is not None and peg_ratio > 0:
            # PEG < 1 - недооценена относительно роста
            if peg_ratio < 0.5:
                scores.append(1.0)
            elif peg_ratio < 1:
                scores.append(0.8)
            elif peg_ratio < 1.5:
                scores.append(0.6)
            elif peg_ratio < 2:
                scores.append(0.4)
            else:
                scores.append(0.2)
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_dividend_score(self, dividend_yield: float) -> float:
        """Расчет скора дивидендной доходности"""
        if dividend_yield is None:
            dividend_yield = 0
        
        # Конвертируем в проценты если нужно
        if dividend_yield < 0.1:
            dividend_yield = dividend_yield * 100
        
        if dividend_yield >= 4:
            return 1.0
        elif dividend_yield >= 3:
            return 0.8
        elif dividend_yield >= 2:
            return 0.6
        elif dividend_yield >= 1:
            return 0.4
        else:
            return 0.2
    
    def _get_rating(self, score: float) -> str:
        """Определение текстового рейтинга на основе скора"""
        if score >= 80:
            return "Strong Buy"
        elif score >= 70:
            return "Buy"
        elif score >= 60:
            return "Hold"
        elif score >= 50:
            return "Weak Hold"
        else:
            return "Sell"
    
    def get_sector_analysis(self) -> pd.DataFrame:
        """
        Анализ по секторам
        
        Returns:
            DataFrame с анализом по секторам
        """
        scores_df = self.calculate_scores()
        
        if scores_df.empty:
            return pd.DataFrame()
        
        sector_analysis = scores_df.groupby('sector').agg({
            'total_score': ['mean', 'std', 'count'],
            'pe_score': 'mean',
            'roe_score': 'mean',
            'growth_score': 'mean',
            'market_cap': 'sum'
        }).round(2)
        
        sector_analysis.columns = ['avg_score', 'score_std', 'count', 
                                  'avg_pe_score', 'avg_roe_score', 
                                  'avg_growth_score', 'total_market_cap']
        
        return sector_analysis.sort_values('avg_score', ascending=False)
    
    def get_top_picks(self, n: int = 10) -> pd.DataFrame:
        """
        Получение топ N компаний по скору
        
        Args:
            n: Количество компаний
            
        Returns:
            DataFrame с топ компаниями
        """
        scores_df = self.calculate_scores()
        return scores_df.head(n)
    
    def get_undervalued_stocks(self, min_score: float = 60) -> pd.DataFrame:
        """
        Получение недооцененных акций
        
        Args:
            min_score: Минимальный скор для отбора
            
        Returns:
            DataFrame с недооцененными акциями
        """
        scores_df = self.calculate_scores()
        undervalued = scores_df[
            (scores_df['total_score'] >= min_score) & 
            (scores_df['valuation_score'] >= 70)
        ]
        return undervalued.sort_values('valuation_score', ascending=False)