"""
Демонстрационный скрипт для AI Stock Portfolio Optimizer
Автор: Мекеда Богдан Сергеевич
"""

import sys
import os
import pandas as pd
import numpy as np

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collector import DataCollector
from fundamental_analyzer import FundamentalAnalyzer
from ml_predictor import MLPredictor
from portfolio_optimizer import PortfolioOptimizer

def main():
    """Основная демонстрационная функция"""
    
    print("🤖 AI Stock Portfolio Optimizer - Демонстрация")
    print("=" * 50)
    
    # Настройки
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    period = "1y"
    
    print(f"📊 Анализируемые акции: {', '.join(tickers)}")
    print(f"📅 Период данных: {period}")
    print()
    
    # 1. Загрузка данных
    print("1️⃣ Загрузка финансовых данных...")
    collector = DataCollector(tickers, period)
    
    try:
        historical_data, fundamental_data = collector.fetch_all_data()
        print(f"✅ Загружено данных: {len(historical_data)} исторических, {len(fundamental_data)} фундаментальных")
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return
    
    if len(historical_data) < 2:
        print("❌ Недостаточно данных для анализа")
        return
    
    # 2. Фундаментальный анализ
    print("\n2️⃣ Фундаментальный анализ...")
    analyzer = FundamentalAnalyzer()
    
    try:
        analysis_results = analyzer.analyze_portfolio(fundamental_data)
        
        if not analysis_results.empty:
            print("✅ Фундаментальный анализ завершен")
            print("\n🏆 Топ-3 акции по фундаментальному скору:")
            
            for i, (_, row) in enumerate(analysis_results.head(3).iterrows()):
                print(f"   {i+1}. {row['ticker']}: {row['total_score']:.1f}/10 ({row['rating']})")
        else:
            print("⚠️ Не удалось провести фундаментальный анализ")
            
    except Exception as e:
        print(f"❌ Ошибка при фундаментальном анализе: {e}")
    
    # 3. Оптимизация портфеля
    print("\n3️⃣ Оптимизация портфеля...")
    optimizer = PortfolioOptimizer()
    
    try:
        returns_df = optimizer.calculate_returns(historical_data)
        print(f"✅ Рассчитаны доходности для {len(returns_df.columns)} акций")
        
        # Оптимизация
        optimal_result = optimizer.optimize_portfolio("max_sharpe")
        
        if optimal_result.get('optimization_success', False):
            print("✅ Портфель успешно оптимизирован")
            print(f"\n📊 Метрики оптимального портфеля:")
            print(f"   • Ожидаемая доходность: {optimal_result['expected_return']:.1%}")
            print(f"   • Волатильность: {optimal_result['volatility']:.1%}")
            print(f"   • Коэффициент Шарпа: {optimal_result['sharpe_ratio']:.2f}")
            
            print(f"\n💼 Распределение портфеля:")
            weights = optimal_result['weights']
            for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.01:  # Показываем только веса больше 1%
                    print(f"   • {ticker}: {weight:.1%}")
        else:
            print("❌ Не удалось оптимизировать портфель")
            
    except Exception as e:
        print(f"❌ Ошибка при оптимизации портфеля: {e}")
    
    # 4. ML прогнозирование (упрощенная версия)
    print("\n4️⃣ Машинное обучение...")
    try:
        predictor = MLPredictor(sequence_length=30, prediction_days=7)  # Упрощенные параметры для демо
        
        # Выберем одну акцию для демонстрации
        demo_ticker = list(historical_data.keys())[0]
        demo_data = historical_data[demo_ticker]
        
        print(f"🔮 Создание прогноза для {demo_ticker}...")
        
        # Подготовка данных
        X, y = predictor.prepare_data(demo_data)
        
        if X.size > 0 and y.size > 0:
            # Обучение модели
            metrics = predictor.train_model(X, y)
            
            if metrics:
                print(f"✅ Модель обучена (MAPE: {metrics['mape']:.1f}%)")
                
                # Прогноз
                prediction_result = predictor.predict_prices(demo_data)
                
                if prediction_result:
                    current_price = prediction_result['current_price']
                    predicted_price = prediction_result['predictions'][-1]
                    expected_return = prediction_result['predicted_return']
                    
                    print(f"📈 Прогноз для {demo_ticker}:")
                    print(f"   • Текущая цена: ${current_price:.2f}")
                    print(f"   • Прогноз цены (7 дней): ${predicted_price:.2f}")
                    print(f"   • Ожидаемая доходность: {expected_return:.1%}")
                else:
                    print("⚠️ Не удалось создать прогноз")
            else:
                print("⚠️ Не удалось обучить модель")
        else:
            print("⚠️ Недостаточно данных для ML модели")
            
    except Exception as e:
        print(f"❌ Ошибка при ML прогнозировании: {e}")
    
    # 5. Бэктестинг (упрощенная версия)
    print("\n5️⃣ Бэктестинг стратегии...")
    try:
        if 'optimal_result' in locals() and optimal_result.get('optimization_success', False):
            from datetime import datetime, timedelta
            
            # Бэктест за последние 6 месяцев
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            backtest_results = optimizer.backtest_strategy(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            if backtest_results:
                print("✅ Бэктестинг завершен")
                print(f"📊 Результаты за 6 месяцев:")
                print(f"   • Общая доходность: {backtest_results['total_return']:.1%}")
                print(f"   • Годовая доходность: {backtest_results['annual_return']:.1%}")
                print(f"   • Коэффициент Шарпа: {backtest_results['sharpe_ratio']:.2f}")
                print(f"   • Максимальная просадка: {backtest_results['max_drawdown']:.1%}")
            else:
                print("⚠️ Не удалось выполнить бэктестинг")
        else:
            print("⚠️ Пропускаем бэктестинг - нет оптимизированного портфеля")
            
    except Exception as e:
        print(f"❌ Ошибка при бэктестинге: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Демонстрация завершена!")
    print("💡 Для полного интерфейса запустите: streamlit run app.py")
    print("⚠️  Помните: это только образовательный проект!")

if __name__ == "__main__":
    main()