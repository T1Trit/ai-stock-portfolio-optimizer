"""
Модуль для оптимизации инвестиционного портфеля
Автор: Мекеда Богдан Сергеевич
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
import logging
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    """Класс для хранения метрик портфеля"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float = 0.0

class PortfolioOptimizer:
    """
    Класс для оптимизации инвестиционного портфеля
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Инициализация оптимизатора
        
        Args:
            risk_free_rate: Безрисковая ставка (по умолчанию 2% годовых)
        """
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.tickers = None
        self.optimal_portfolio = None
        
    def prepare_returns_data(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Подготовка данных доходностей для оптимизации
        
        Args:
            historical_data: Словарь с историческими данными
            
        Returns:
            DataFrame с доходностями
        """
        returns_dict = {}
        
        for ticker, data in historical_data.items():
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                returns_dict[ticker] = returns
        
        self.returns_data = pd.DataFrame(returns_dict).dropna()
        self.tickers = list(self.returns_data.columns)
        
        logger.info(f"Подготовлены данные доходностей для {len(self.tickers)} активов")
        return self.returns_data
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> PortfolioMetrics:
        """
        Расчет метрик портфеля
        
        Args:
            weights: Веса активов в портфеле
            
        Returns:
            Объект с метриками портфеля
        """
        # Ожидаемая доходность (аннуализированная)
        expected_return = np.sum(self.returns_data.mean() * weights) * 252
        
        # Волатильность (аннуализированная)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns_data.cov() * 252, weights)))
        
        # Коэффициент Шарпа
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        
        # Коэффициент Сортино
        portfolio_returns = (self.returns_data * weights).sum(axis=1)
        downside_std = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        sortino_ratio = (expected_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Value at Risk (95%)
        portfolio_returns_annual = portfolio_returns * np.sqrt(252)
        var_95 = np.percentile(portfolio_returns_annual, 5)
        
        # Conditional Value at Risk (95%)
        cvar_95 = portfolio_returns_annual[portfolio_returns_annual <= var_95].mean()
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def optimize_sharpe_ratio(self, min_weight: float = 0.05, max_weight: float = 0.25) -> PortfolioMetrics:
        """
        Оптимизация портфеля по максимизации коэффициента Шарпа
        
        Args:
            min_weight: Минимальный вес актива
            max_weight: Максимальный вес актива
            
        Returns:
            Оптимальный портфель
        """
        n_assets = len(self.tickers)
        
        # Начальные веса (равные)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Целевая функция (отрицательный коэффициент Шарпа для минимизации)
        def negative_sharpe(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            return -metrics.sharpe_ratio
        
        # Ограничения
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Сумма весов = 1
        ]
        
        # Границы весов
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Оптимизация
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            optimal_metrics = self.calculate_portfolio_metrics(result.x)
            self.optimal_portfolio = optimal_metrics
            logger.info(f"Оптимизация успешна. Коэффициент Шарпа: {optimal_metrics.sharpe_ratio:.3f}")
            return optimal_metrics
        else:
            logger.error("Ошибка оптимизации")
            return None
    
    def optimize_min_volatility(self, target_return: float = None) -> PortfolioMetrics:
        """
        Оптимизация портфеля по минимизации волатильности
        
        Args:
            target_return: Целевая доходность (если не задана, берется без ограничений)
            
        Returns:
            Портфель с минимальной волатильностью
        """
        n_assets = len(self.tickers)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        def portfolio_volatility(weights):
            return self.calculate_portfolio_metrics(weights).volatility
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: self.calculate_portfolio_metrics(x).expected_return - target_return
            })
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            return self.calculate_portfolio_metrics(result.x)
        else:
            logger.error("Ошибка оптимизации минимальной волатильности")
            return None
    
    def calculate_efficient_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """
        Построение эффективной границы
        
        Args:
            n_portfolios: Количество портфелей для построения
            
        Returns:
            DataFrame с портфелями на эффективной границе
        """
        # Диапазон целевых доходностей
        min_vol_portfolio = self.optimize_min_volatility()
        max_sharpe_portfolio = self.optimize_sharpe_ratio()
        
        min_return = min_vol_portfolio.expected_return
        max_return = max_sharpe_portfolio.expected_return
        
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Оптимизация для заданной доходности
            n_assets = len(self.tickers)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.returns_data.cov() * 252, weights)))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: (self.returns_data.mean() * x).sum() * 252 - target_return}
            ]
            
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            result = minimize(
                portfolio_volatility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False}
            )
            
            if result.success:
                metrics = self.calculate_portfolio_metrics(result.x)
                efficient_portfolios.append({
                    'Return': metrics.expected_return,
                    'Volatility': metrics.volatility,
                    'Sharpe': metrics.sharpe_ratio,
                    'Weights': metrics.weights
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def monte_carlo_simulation(self, n_simulations: int = 10000) -> pd.DataFrame:
        """
        Monte Carlo симуляция случайных портфелей
        
        Args:
            n_simulations: Количество симуляций
            
        Returns:
            DataFrame с результатами симуляций
        """
        n_assets = len(self.tickers)
        results = []
        
        for _ in range(n_simulations):
            # Генерация случайных весов
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            # Расчет метрик
            metrics = self.calculate_portfolio_metrics(weights)
            
            results.append({
                'Return': metrics.expected_return,
                'Volatility': metrics.volatility,
                'Sharpe': metrics.sharpe_ratio
            })
        
        simulation_df = pd.DataFrame(results)
        return simulation_df
    
    def get_portfolio_recommendations(self, 
                                    fundamental_scores: pd.DataFrame = None,
                                    ml_predictions: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Генерация рекомендаций по портфелю с учетом всех факторов
        
        Args:
            fundamental_scores: Фундаментальные скоры компаний
            ml_predictions: ML предсказания цен
            
        Returns:
            DataFrame с рекомендациями
        """
        if self.optimal_portfolio is None:
            self.optimize_sharpe_ratio()
        
        # Создание DataFrame с рекомендациями
        recommendations = pd.DataFrame({
            'Ticker': self.tickers,
            'Weight': self.optimal_portfolio.weights
        })
        
        # Добавление фундаментальных скоров
        if fundamental_scores is not None:
            for ticker in recommendations['Ticker']:
                if ticker in fundamental_scores.index:
                    recommendations.loc[recommendations['Ticker'] == ticker, 'Fundamental_Score'] = \
                        fundamental_scores.loc[ticker, 'total_score']
                    recommendations.loc[recommendations['Ticker'] == ticker, 'Rating'] = \
                        fundamental_scores.loc[ticker, 'rating']
        
        # Добавление ML предсказаний
        if ml_predictions is not None:
            for ticker in recommendations['Ticker']:
                if ticker in ml_predictions and not ml_predictions[ticker].empty:
                    expected_return = (ml_predictions[ticker]['Predicted_Price'].iloc[-1] / 
                                     ml_predictions[ticker]['Predicted_Price'].iloc[0] - 1)
                    recommendations.loc[recommendations['Ticker'] == ticker, 'ML_Expected_Return'] = expected_return
        
        # Сортировка по весу
        recommendations = recommendations.sort_values('Weight', ascending=False)
        
        # Добавление сигналов покупки/продажи
        recommendations['Signal'] = recommendations.apply(
            lambda row: 'Strong Buy' if row['Weight'] > 0.15 and row.get('Fundamental_Score', 0) > 70
            else 'Buy' if row['Weight'] > 0.10
            else 'Hold' if row['Weight'] > 0.05
            else 'Underweight',
            axis=1
        )
        
        return recommendations
    
    def backtest_strategy(self, start_date: str, end_date: str) -> Dict:
        """
        Бэктестинг стратегии
        
        Args:
            start_date: Дата начала
            end_date: Дата окончания
            
        Returns:
            Результаты бэктестинга
        """
        # Фильтрация данных по датам
        backtest_returns = self.returns_data.loc[start_date:end_date]
        
        if self.optimal_portfolio is None:
            self.optimize_sharpe_ratio()
        
        # Расчет доходности портфеля
        portfolio_returns = (backtest_returns * self.optimal_portfolio.weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Расчет метрик
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility
        
        # Максимальная просадка
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }
    
    def optimize_risk_parity(self) -> PortfolioMetrics:
        """
        Оптимизация портфеля по принципу равного риска
        
        Returns:
            Портфель с равным вкладом риска
        """
        n_assets = len(self.tickers)
        
        def risk_parity_objective(weights):
            # Ковариационная матрица
            cov_matrix = self.returns_data.cov() * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Маргинальный вклад в риск
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Целевой вклад в риск (равный)
            target_contrib = np.ones(n_assets) / n_assets
            
            # Минимизируем разность между фактическим и целевым вкладом
            return np.sum((contrib - target_contrib) ** 2)
        
        # Ограничения
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        bounds = tuple((0.01, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            return self.calculate_portfolio_metrics(result.x)
        else:
            logger.error("Ошибка оптимизации risk parity")
            return None
    
    def get_portfolio_summary(self) -> Dict:
        """
        Получение сводки по портфелю
        
        Returns:
            Словарь с основными метриками
        """
        if self.optimal_portfolio is None:
            return {}
        
        return {
            'expected_return': f"{self.optimal_portfolio.expected_return:.2%}",
            'volatility': f"{self.optimal_portfolio.volatility:.2%}",
            'sharpe_ratio': f"{self.optimal_portfolio.sharpe_ratio:.3f}",
            'sortino_ratio': f"{self.optimal_portfolio.sortino_ratio:.3f}",
            'var_95': f"{self.optimal_portfolio.var_95:.2%}",
            'cvar_95': f"{self.optimal_portfolio.cvar_95:.2%}",
            'num_assets': len(self.tickers),
            'top_holding': self.tickers[np.argmax(self.optimal_portfolio.weights)],
            'max_weight': f"{np.max(self.optimal_portfolio.weights):.1%}"
        }