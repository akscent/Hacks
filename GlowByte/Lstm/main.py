import re
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, \
    mean_absolute_error, r2_score, mean_squared_error
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import pandas as pd
import numpy as np

from utils import dct_to_replace, unique_words


class PreprocessorPredict:
    def __init__(self, filepath,
                 path_to_test='./test_dataset.csv',
                 path_to_model='models/model_lstm.h5',
                 path_to_model_weights='models/model_lstm_weights.h5',
                 path_to_save_csv='./predicted.csv',
                 xLen=150):

        if os.path.exists(filepath) and os.path.exists(path_to_test) and os.path.exists(path_to_model) and os.path.exists(path_to_model_weights):
            self.df_test = pd.read_csv(path_to_test)
            self.df_private = pd.read_csv(filepath)
            self.data = pd.concat([self.df_test, self.df_private])
            self.data = self.data.reset_index(drop = True)
            self.loaded_model = load_model(path_to_model)
            self.loaded_model.load_weights(path_to_model_weights)
            self.path_to_save = path_to_save_csv
        else:
            raise Exception('Укажите путь к тестовому и приватному датасету')
        
        self.xLen = xLen
        self.xTest = None
        self.yTest = None
        self.xScaler = None
        self.yScaler = None
    
    @staticmethod
    def extract_numbers(text):
        """Ищем числа в строке (вероятности возникновения погодного события)."""
        numbers = re.findall(r'\d+', text)
        lst = [int(num) for num in numbers]
        if len(lst) == 0:
            return 0
        return max(lst)

    @staticmethod
    def replace_strange_chars(text):
        """Удаляем все символы, оставляем только слова, удаляем лишние пробелы, переводим в lowercase."""
        cleaned_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip().lower()
    
    @staticmethod
    def replaced_words(row):
        """Заменяем слова из строки по совпадению ключей словаря на значения."""
        for pattern, replacement in dct_to_replace.items():
            row = re.sub(r'\b(?:' + pattern + r')\b', replacement, row)
        return row
    
    @staticmethod
    def fill_empty_cells(df):
        print('INFO: Заполнение пустых значений')
        df['temp_pred'].fillna(method='ffill', inplace=True)
        df['weather_pred'].fillna(method='ffill', inplace=True)
        df['weather_fact'].fillna(method='ffill', inplace=True)
        print('Пропущенных значений:', df.isna().sum().sum(), '\n')
        return df
    
    @staticmethod
    def add_features(df):
        df['date'] = pd.to_datetime(df['date'])                               # Столбец Date в формат datetime
        df['day_of_week'] = df['date'].dt.dayofweek                           # Номер дня в неделе
        df['month'] = df['date'].dt.month                                     # Номер месяца
        df['year'] = df['date'].dt.year                                       # Номер года
        df['date'] = (df['date'] - df['date'].min()).dt.days                  # Дату в количество прошедших дней с начальной даты
        
        print('INFO: Добавление полезных фичей')
        df['target_t_1'] = df['target'].shift(1)                              # Значение target на предыдущий час
        df['target_t_24'] = df['target'].shift(24)                            # Значение target на предыдущий день
        df['target_avg5'] = df['target'].rolling(window=5).mean()             # Среднее значение target за предыдущие 5 часов
        df['target_avg24'] = df['target'].rolling(window=24).mean()           # Среднее значение target за предыдущие 24 часа

        df['temp_pred_avg5'] = df['temp_pred'].rolling(window=5).mean()       # Среднее значение temp_pred за предыдущие 5 часов
        df['temp_pred_avg24'] = df['temp_pred'].rolling(window=24).mean()     # Среднее значение temp_pred за предыдущие 24 часа
        df['temp_avg5'] = df['temp'].rolling(window=5).mean()                 # Среднее значение temp за предыдущие 5 часов
        df['temp_avg24'] = df['temp'].rolling(window=24).mean()               # Среднее значение temp за предыдущие 24 часа

        for i in ['target', 'temp', 'temp_pred']:
            df[f'{i}_derivative_1'] = df[i].diff()                            # Производная 1 порядка
            df[f'{i}_derivative_2'] = df[i].diff()                            # Производная 2 порядка
            df[f'{i}_inverse'] = 1 / (df[i] + 0.01)                           # Обратное значение

        print('Размер датафрейма:', df.shape, '\n')
        return df

    def parsing_cat_cols(self, df):
        print('INFO: Парсим и кодируем в OHE столбцы погодных условиях')

        df['probability_pred'] = df['weather_pred'].apply(self.extract_numbers)
        df['probability_fact'] = df['weather_fact'].apply(self.extract_numbers)

        df['weather_pred'] = df['weather_pred'].apply(self.replace_strange_chars)
        df['weather_fact'] = df['weather_fact'].apply(self.replace_strange_chars)

        df['weather_pred'] = df['weather_pred'].apply(self.replaced_words)
        df['weather_fact'] = df['weather_fact'].apply(self.replaced_words)

        df['weather_pred'] = df['weather_pred'].str.split()
        df['weather_fact'] = df['weather_fact'].str.split()

        # OHE
        for word in unique_words:
            df[word + '_pred'] = df['weather_pred'].apply(lambda x: int(word in x))
            df[word + '_fact'] = df['weather_fact'].apply(lambda x: int(word in x))

        # Удаляем исходные категориальные переменные
        df = df.drop(['weather_pred', 'weather_fact'], axis=1)
        print('Размер датафрейма:', df.shape)
        return df

    def preprocessing_1(self):
        self.data = self.fill_empty_cells(self.data)
        self.data = self.add_features(self.data)
        self.data = self.parsing_cat_cols(self.data)
        self.data.fillna(value=0, inplace=True)
        self.data = np.array(self.data)
    
    def preprocessing_2(self):
        # Формирование параметров загрузки данных
        valLen = self.df_private.shape[0] + (self.xLen * 2)
        trainLen = self.data.shape[0] - valLen # Размер тренировочной выборки

        xTrain,xTest = self.data[:trainLen], self.data[trainLen+self.xLen:]
        xScaler = MinMaxScaler()
        xScaler.fit(xTrain)
        xTrain = xScaler.transform(xTrain)
        xTest = xScaler.transform(xTest)
        
        yTrain,yTest = np.reshape(self.data[:trainLen,2],(-1,1)), np.reshape(self.data[trainLen+self.xLen:,2],(-1,1))
        yScaler = MinMaxScaler()
        yScaler.fit(yTrain)
        yTrain = yScaler.transform(yTrain)
        yTest = yScaler.transform(yTest)
        
        self.xTest = xTest
        self.yTest = yTest
        self.xScaler = xScaler
        self.yScaler = yScaler

    @staticmethod
    def getPred(currModel, xVal, yVal, yScaler):
        predVal = yScaler.inverse_transform(currModel.predict(xVal))
        yValUnscaled = yScaler.inverse_transform(yVal)
        return (predVal, yValUnscaled)

    @staticmethod
    # бейзлайн - предсказать потребление электроэнергии предыдущего дня
    def shift_target(data):
        shifted_data = data.copy()
        shifted_data['baseline'] = shifted_data['target'].shift(-24)
        shifted_data['baseline'][-24:] = data['target'][-24:]
        return shifted_data

    @staticmethod
    def shift_predict(data):
        shifted_data = data.copy()
        shifted_data['predict_2'] = shifted_data['target'].shift(1)
        shifted_data['predict_2'].iloc[0] = data['target'].iloc[-1]
        return shifted_data

    @staticmethod
    # Функция получения метрик
    def get_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2_scr = r2_score(y_true, y_pred)
        print("MSE: %.2f" % mse)
        print("MAE: %.2f" % mae)
        print("MAPE: %.2f" % mape)
        print("R2_score: %.2f" % r2_scr)
        # return mse, mae, mape, r2_scr

    def gener_datagen(self):
        DataGen = TimeseriesGenerator(
            self.xTest, self.yTest,
            length=self.xLen,
            sampling_rate=1,
            batch_size=len(self.xTest))

        xVal, yVal = [], []
        for i in DataGen:
            xVal.append(i[0])
            yVal.append(i[1])
        xVal = np.array(xVal)
        yVal = np.array(yVal)
        print(DataGen[0][0].shape, DataGen[0][1].shape)
        print(self.df_private.shape[0])
        return xVal, yVal


    def predict(self):
        self.preprocessing_1()
        self.preprocessing_2()
        xVal, yVal = self.gener_datagen()

        # Предсказание модели
        yVal_reshaped = yVal[0][:, -1, :]
        (predVal_1, yValUnscaled) = self.getPred(self.loaded_model, xVal[0], yVal_reshaped, self.yScaler)
        self.df_private = self.shift_target(self.df_private)                    # baseline

        self.df_private['predict_today_tomorrow'] = predVal_1.tolist()

        self.df_private['predict'] = self.df_private['predict_today_tomorrow'].apply(lambda x: x[0])
        self.df_private['predict_2'] = self.df_private['predict_today_tomorrow'].apply(lambda x: x[1])
        self.df_private = self.df_private[['date', 'target', 'baseline', 'predict', 'predict_2']]
        df_ex = self.df_private.copy()
        print('\n\n\nМетрика target - baseline')
        self.get_metrics(self.df_private['target'], self.df_private['baseline'])
        print('\n\nМетрика target - predict на сегодня', '\n')
        self.get_metrics(self.df_private['target'], self.df_private['predict'])
        self.df_private = self.shift_predict(self.df_private) # Отпускаем предсказания на строку вниз, т.к. это прогноз на следующий час
        print('\n\nМетрика target - predict_2 на следующий час', '\n')
        self.get_metrics(self.df_private['target'], self.df_private['predict_2'])
        self.df_private = self.df_private[['date', 'predict']]
        self.df_private.to_csv(self.path_to_save, index=False)
        print('\n\nФайл с предсказаниями сохранен по пути:', self.path_to_save)
        return df_ex


if __name__ == '__main__':
    filepath = './test_dataset.csv'     # Указать путь к приватному датасету
    
    inf = 0
    if inf:
        path_to_test='./train_dataset.csv'
    else:
        path_to_test='./test_dataset.csv'
    path_to_model='models/model_lstm.h5'
    path_to_model_weights='models/model_lstm_weights.h5'
    path_to_save_csv='./predicted.csv'
    sol = PreprocessorPredict(filepath, path_to_test, path_to_model, path_to_model_weights, path_to_save_csv)
    df = sol.predict()