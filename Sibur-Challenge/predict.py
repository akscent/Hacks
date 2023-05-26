import pathlib
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np

DATA_DIR = pathlib.Path(".")
MODEL_FILE_0 = pathlib.Path(__file__).parent.joinpath("target0-cb-v1.cbm")
MODEL_FILE_1 = pathlib.Path(__file__).parent.joinpath("target1-cb-v1.cbm")


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление предсказаний.

    Параметры:
        df:
          датафрейм, содержащий строки из тестового множества.
          Типы и имена колонок совпадают с типами и именами в ноутбуке, не содержит `np.nan` или `np.inf`.

    Результат:
        Датафрейм предсказаний.
        Должен содержать то же количество строк и в том же порядке, а также колонки `target0` и `target1`.
    """    

    predictions = {}

    df["gas"] = 0
    df.loc[df.feature4=="gas2", "gas"] = 1
    FTS = df.columns.difference(["feature4"])

    for target, model_file in zip(["target0", "target1"], [MODEL_FILE_0, MODEL_FILE_1]):
        model = CatBoostRegressor()
        model.load_model(model_file)
        predictions[target] = model.predict(df[FTS])

    preds_df = pd.DataFrame(predictions, index=df.index)
    return preds_df
