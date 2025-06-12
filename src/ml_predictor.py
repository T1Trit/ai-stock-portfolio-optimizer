"""
Модуль для предсказания цен акций с использованием LSTM
Автор: Мекеда Богдан Сергеевич
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class StockPricePredictor:
    """
    Класс для предсказания цен акций с использованием LSTM нейронных сетей
    """
    
    def __init__(self, lookback_days: int = 60, forecast_days: int = 30):
        """
        Инициализация предсказателя
        
        Args:
            lookback_days: Количество дней для обучения модели
            forecast_days: Количество дней для предсказания
        """
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        
    def prepare_data(self, data: pd.DataFrame, features: List[str] = None) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Подготовка данных для обучения LSTM модели
        
        Args:
            data: DataFrame с историческими данными
            features: Список признаков для использования
            
        Returns:
            Кортеж (X, y, scaler)
        """
        if features is None:
            features = ['Close', 'Volume', 'RSI', 'MACD', 'Volatility']
        
        # Фильтруем доступные признаки
        available_features = [f for f in features if f in data.columns]
        if not available_features:
            available_features = ['Close']
        
        # Убираем NaN значения
        dataset = data[available_features].dropna()
        
        # Нормализация данных
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset.values)
        
        # Создание последовательностей для LSTM
        X, y = [], []
        
        for i in range(self.lookback_days, len(scaled_data) - self.forecast_days + 1):
            X.append(scaled_data[i-self.lookback_days:i])
            # Предсказываем цену закрытия (первый столбец)
            y.append(scaled_data[i:i+self.forecast_days, 0])
        
        return np.array(X), np.array(y), scaler
    
    def build_model(self, input_shape: tuple) -> Sequential:
        """
        Создание LSTM модели
        
        Args:
            input_shape: Форма входных данных
            
        Returns:
            Скомпилированная модель
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            
            Dense(units=25),
            Dense(units=self.forecast_days)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, ticker: str, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Обучение модели для конкретной акции
        
        Args:
            ticker: Тикер акции
            X: Обучающие данные
            y: Целевые значения
            
        Returns:
            Словарь с метриками обучения
        """
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Создание модели
        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.001
        )
        
        # Обучение модели
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Сохранение модели
        self.models[ticker] = model
        
        # Оценка модели
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Расчет метрик
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        
        results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'val_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'epochs': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        
        logger.info(f"Модель для {ticker} обучена. Val RMSE: {test_rmse:.4f}")
        
        return results
    
    def predict_future_prices(self, ticker: str, current_data: pd.DataFrame, 
                            features: List[str] = None) -> pd.DataFrame:
        """
        Предсказание будущих цен для акции
        
        Args:
            ticker: Тикер акции
            current_data: Текущие данные
            features: Список признаков
            
        Returns:
            DataFrame с предсказаниями
        """
        if ticker not in self.models:
            logger.error(f"Модель для {ticker} не обучена")
            return pd.DataFrame()
        
        if features is None:
            features = ['Close', 'Volume', 'RSI', 'MACD', 'Volatility']
        
        # Подготовка данных
        available_features = [f for f in features if f in current_data.columns]
        data = current_data[available_features].dropna()
        
        # Используем сохраненный scaler
        if ticker not in self.scalers:
            logger.error(f"Scaler для {ticker} не найден")
            return pd.DataFrame()
        
        scaler = self.scalers[ticker]
        scaled_data = scaler.transform(data)
        
        # Берем последние lookback_days дней для предсказания
        last_sequence = scaled_data[-self.lookback_days:]
        last_sequence = last_sequence.reshape(1, self.lookback_days, len(available_features))
        
        # Предсказание
        model = self.models[ticker]
        scaled_prediction = model.predict(last_sequence, verbose=0)
        
        # Обратное масштабирование для цены
        # Создаем массив нужной размерности для обратного преобразования
        dummy_array = np.zeros((scaled_prediction.shape[1], len(available_features)))
        dummy_array[:, 0] = scaled_prediction[0]
        
        prediction = scaler.inverse_transform(dummy_array)[:, 0]
        
        # Создание DataFrame с предсказаниями
        last_date = current_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.forecast_days,
            freq='D'
        )
        
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': prediction,
            'Ticker': ticker
        })
        predictions_df.set_index('Date', inplace=True)
        
        # Добавляем доверительные интервалы (упрощенный подход)
        std_dev = prediction.std()
        predictions_df['Lower_Bound'] = prediction - 1.96 * std_dev
        predictions_df['Upper_Bound'] = prediction + 1.96 * std_dev
        
        return predictions_df
    
    def train_all_models(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Обучение моделей для всех акций
        
        Args:
            historical_data: Словарь с историческими данными
            
        Returns:
            Словарь с результатами обучения
        """
        results = {}
        
        for ticker, data in historical_data.items():
            try:
                logger.info(f"Подготовка данных для {ticker}...")
                
                # Подготовка данных
                X, y, scaler = self.prepare_data(data)
                
                if len(X) < 100:
                    logger.warning(f"Недостаточно данных для {ticker}, пропускаем")
                    continue
                
                # Сохранение scaler
                self.scalers[ticker] = scaler
                
                # Обучение модели
                train_results = self.train_model(ticker, X, y)
                results[ticker] = train_results
                
                # Генерация предсказаний
                predictions = self.predict_future_prices(ticker, data)
                self.predictions[ticker] = predictions
                
            except Exception as e:
                logger.error(f"Ошибка при обучении модели для {ticker}: {str(e)}")
                continue
        
        return results
    
    def get_prediction_confidence(self, ticker: str) -> Dict[str, float]:
        """
        Расчет уверенности в предсказаниях
        
        Args:
            ticker: Тикер акции
            
        Returns:
            Словарь с метриками уверенности
        """
        if ticker not in self.predictions:
            return {}
        
        predictions = self.predictions[ticker]
        
        # Расчет метрик уверенности
        price_volatility = predictions['Predicted_Price'].std() / predictions['Predicted_Price'].mean()
        prediction_range = (predictions['Upper_Bound'] - predictions['Lower_Bound']).mean()
        confidence_score = 1 / (1 + price_volatility)  # Чем меньше волатильность, тем выше уверенность
        
        return {
            'volatility': price_volatility,
            'avg_prediction_range': prediction_range,
            'confidence_score': confidence_score,
            'expected_return': (predictions['Predicted_Price'].iloc[-1] / predictions['Predicted_Price'].iloc[0] - 1)
        }
    
    def get_all_predictions(self) -> Dict[str, pd.DataFrame]:
        """
        Получение всех предсказаний
        
        Returns:
            Словарь с предсказаниями для всех акций
        """
        return self.predictions
    
    def save_models(self, path: str):
        """
        Сохранение обученных моделей
        
        Args:
            path: Путь для сохранения
        """
        import pickle
        import os
        
        os.makedirs(path, exist_ok=True)
        
        # Сохранение моделей Keras
        for ticker, model in self.models.items():
            model.save(f"{path}/{ticker}_model.h5")
        
        # Сохранение scalers
        with open(f"{path}/scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        logger.info(f"Модели сохранены в {path}")
    
    def load_models(self, path: str):
        """
        Загрузка сохраненных моделей
        
        Args:
            path: Путь к сохраненным моделям
        """
        import pickle
        import os
        from tensorflow.keras.models import load_model
        
        # Загрузка моделей Keras
        for file in os.listdir(path):
            if file.endswith('_model.h5'):
                ticker = file.replace('_model.h5', '')
                self.models[ticker] = load_model(f"{path}/{file}")
        
        # Загрузка scalers
        scalers_path = f"{path}/scalers.pkl"
        if os.path.exists(scalers_path):
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
        
        logger.info(f"Модели загружены из {path}")
    
    def get_model_summary(self) -> Dict:
        """
        Получение сводки по моделям
        
        Returns:
            Словарь с информацией о моделях
        """
        summary = {
            'total_models': len(self.models),
            'tickers': list(self.models.keys()),
            'forecast_days': self.forecast_days,
            'lookback_days': self.lookback_days
        }
        
        return summary