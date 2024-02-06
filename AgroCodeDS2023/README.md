# Архитектура решения
Итоговое решение содержит 8 моделей. Каждая модель состоит из ансамблевых каскадных моделей. Каждый каскад — это три популярных бустинга: lgbm, xgboost и catboost. Окончательным каскадным предсказаниям присваиваются равные веса.
![alt text](src/architecture.png?raw=true)  
## Генерация фич
Лучшими функциями были Milk_yield_2 и Farm, кроме них было несколько созданных вручную. Вы можете найти код генерации в функции `generate_features`.
## Нормализация таргета
Можно заметить, что данные имеют сильный временной тренд. Так что прогнозировать этот тип цели с помощью бустинга — не лучшая идея. Поэтому перед обучением нам необходимо нормализовать таргет. Здесь я использую два типа нормализации: разница с Milk_yield_2 и удаление тренда с помощью линейной регрессии. 
![alt text](src/linreg_detrend.png?raw=true)  
![alt text](src/diff_detrend.png?raw=true)  

# Запуск решения
 
```bash
pip install -r src/requirements.txt
python3 two_targets.py -s data/submission.csv -tr data/train.csv -te data/X_test_private.csv
python3 score.py -p submission.csv -t data/y_test_private.csv
```
