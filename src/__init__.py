"""
AI Stock Portfolio Optimizer

Модули для анализа и оптимизации инвестиционного портфеля акций.

Автор: Мекеда Богдан Сергеевич
Университет: ИТМО, ФТМИ, Бизнес-информатика
Год: 2025
"""

__version__ = "1.0.0"
__author__ = "Мекеда Богдан Сергеевич"

from .data_collector import DataCollector
from .fundamental_analyzer import FundamentalAnalyzer
from .ml_predictor import MLPredictor
from .portfolio_optimizer import PortfolioOptimizer, OptimalPortfolio

__all__ = [
    'DataCollector',
    'FundamentalAnalyzer', 
    'MLPredictor',
    'PortfolioOptimizer',
    'OptimalPortfolio'
]