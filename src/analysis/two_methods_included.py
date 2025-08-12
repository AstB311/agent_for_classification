import logging
import warnings
from collections import Counter
from typing import List, Dict, Any, Type, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

#Склеивает список словарей data и столбец labels
def concatenate_data_with_labels(data: np.ndarray, labels_data: np.ndarray, side: str) -> np.ndarray:
    """Склеивает массив данных с метками кластеров, добавляя столбец меток с указанной стороны.

    Args:
        data: np.ndarray. Массив данных, к которому нужно добавить метки.
        labels_data: np.ndarray. Массив меток кластеров.
        side: str. Указывает, с какой стороны добавить столбец меток: "front" (спереди) или "behind" (сзади).

    Returns:
        np.ndarray. Новый массив данных с добавленным столбцом меток.

    Raises:
        ValueError: Если длина массива `data` и массива `labels_data` не совпадает.
        ValueError: Если значение параметра `side` не является "front" или "behind".
    """
    if len(data) != len(labels_data):
        raise ValueError("Длина массива data и столбца должна совпадать.")
    labels_data = labels_data.reshape(-1, 1)
    if side == "front":
        result_data = np.concatenate((labels_data, data), axis=1)
    elif side == "behind":
        result_data = np.concatenate((data, labels_data), axis=1)
    else:
        raise ValueError("Значение параметра side должно быть 'front' или 'behind'.")
    return result_data

# Приведение данных в нужный формат
def data_formater(data: List[Dict], task_manager: str, method: str) -> Any:
    """Форматирует данные в зависимости от задачи (обучение или предсказание) и метода (кластеризация или классификация).

    Args:
        data: List[Dict[str, Any]]. Список словарей, представляющих данные.
        task_manager: str. Указывает задачу: "LEARN" (обучение) или "PREDICT" (предсказание).
        method: str. Указывает метод: "clasterization" (кластеризация) или "classification" (классификация).

    Returns:
        Any.  Возвращает данные в формате, зависящем от задачи и метода:
              - LEARN, clasterization: Tuple[np.ndarray, int, np.ndarray, pd.Series]: (массив признаков, количество кластеров, массив ID, столбец меток).
              - LEARN, classification: pd.DataFrame: DataFrame с переименованным столбцом меток ("claster_coloumn").
              - PREDICT, clasterization: Tuple[np.ndarray, int, np.ndarray, np.ndarray]: (массив признаков, количество кластеров, массив ID, массив временных данных).
              - PREDICT, classification: List[Dict]:  Исходный список словарей.
              - Другие случаи: np.array([]).  Пустой массив numpy.

    Raises:
        Exception: Если при обработке данных возникла ошибка.
    """
    try:
        df = pd.DataFrame(data)
        num_clusters = -1
        id_column = []
        if task_manager == "LEARN":
            if method == "clasterization":
                id_column = df.iloc[:, 0].to_numpy() # Извлекаем первый столбец (столбец с ID)
                columns_to_exclude = [col for col in df.columns if 'time' in col.lower()] # Исключаем колонки, содержащие "time" в их имени
                if len(df.columns) > 0:
                    columns_to_exclude.append(df.columns[0])  # Исключаем первую колонку (ID)
                label_column = df.iloc[:, -1] #Извлекаем последний столбец (предполагаем, что это столбец с метками)
                unique_labels = label_column.unique() #Определяем количество уникальных меток
                num_clusters = len(unique_labels)
                columns_to_use = [col for col in df.columns if col not in columns_to_exclude] # Исключаем колонки "time" из данных, используемых для обучения
                df_features = df[columns_to_use]
                data_for_learning = df_features.iloc[:, :-1].to_numpy() #Возвращаем все данные без столбца с метками
                return data_for_learning, num_clusters, id_column, label_column
            elif method == "classification":
                columns_to_rename = df.columns[len(df.columns) - 1]
                data_for_learning = df.rename(columns={columns_to_rename: "claster_coloumn"})
                return data_for_learning
        elif task_manager == "PREDICT":
            if method=="clasterization":
                id_column = df.iloc[:, 0].to_numpy()
                columns_time = [col for col in df.columns if 'time' in col.lower()]
                df_time = df[columns_time]
                df_time = df_time.to_numpy()
                columns_to_use = [col for col in df.columns if col not in columns_time and col != df.columns[0]]
                df_features = df[columns_to_use]
                data_for_prediction = df_features.to_numpy()
                return data_for_prediction, num_clusters, id_column, df_time
            elif method == "classification":
                return data
        else:
            print(f"Неизвестный task_manager: {task_manager}.  Возвращаем пустой массив.")
            return np.array([])
    except Exception as e:
        return np.array([])


# Общая функция для выбора метода из готового списка по анализу
def method_selector_by_analysis(analysis_results: Dict[str, str], rules: Dict) -> str:
    """Выбирает метод анализа на основе результатов анализа данных и заданных правил.

    Args:
        analysis_results: Dict[str, str].
            Словарь с результатами анализа данных, где ключ - критерий анализа, а значение - его результат.
        rules: Dict[str, Dict[str, list]].
            Словарь правил, определяющих, какие методы подходят для каких результатов анализа.
            Структура: {
                "criterion1": {"value1": ["methodA", "methodB"], "value2": ["methodC"]},
                "criterion2": {"value3": ["methodA"], "value4": ["methodD"]},
                "defaults": {"algorithm": "methodE"}  # Метод по умолчанию
            }

        Returns:
          str. Наиболее подходящий метод анализа на основе результатов анализа и правил.
                Если не найдено ни одного подходящего метода, возвращает метод по умолчанию,
                    указанный в rules['defaults']['algorithm'].
    """
    method_counts = Counter()
    for criterion, value in analysis_results.items():
        if criterion in rules:
            possible_methods = rules[criterion].get(value)
            if possible_methods is None:
                continue
            method_counts.update(possible_methods)
    if not method_counts:
        return rules['defaults']['algorithm']
    # Выбираем метод с наибольшим количеством голосов
    most_common_method, _ = method_counts.most_common(1)[0]
    return most_common_method

# Функция objective для подбора гиперпараметров
def objective_claster(
        trial: optuna.Trial,
        model_class: Type[Any],
        param_grid: Dict[str, Any],
        X: np.ndarray
) -> float:
    """Функция objective для подбора гиперпараметров модели кластеризации с использованием Optuna.

    Args:
      trial: optuna.Trial. Объект Trial Optuna, представляющий текущую попытку подбора гиперпараметров.
      model_class: Type[Any]. Класс модели кластеризации (например, KMeans, AgglomerativeClustering).
      param_grid: Dict[str, Any]. Словарь с сеткой гиперпараметров для подбора.  Значениями могут быть кортежи (для integer параметров) или списки (для categorical параметров).
             Пример: {"n_clusters": (2, 10), "init": ["k-means++", "random"]}
      X: np.ndarray. Массив данных для кластеризации.

    Returns:
      float. Значение силуэтного коэффициента для модели с текущими гиперпараметрами.  Возвращает -1, если силуэтный коэффициент равен NaN.
    """
    # Настройка логгера Optuna
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    # Подбор гиперпараметров
    params = {}
    for param_name, param_range in param_grid.items():
        if isinstance(param_range, tuple):
            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
        else:
            params[param_name] = trial.suggest_categorical(param_name, param_range)
    # Создание модели с подобранными гиперпараметрами
    model = model_class(**params)
    # Оценка модели с помощью силуэтного коэффициента
    model.fit(X)
    labels = model.labels_
    score = silhouette_score(X, labels)
    # Проверка на NaN значения
    if np.isnan(score):
        return -1  # Возвращаем отрицательное значение, если метрика возвращает NaN
    return score


# Функция для нахождения оптимальных гиперпараметров OPTUNA - кластеризация
def optimize_hyperparameters_claster(
        model_class: Type[Any],
        param_grid: Dict[str, Any],
        X: np.ndarray,
        n_trials: int = 50
) -> Tuple[Dict[str, Any], float]:
    """Находит оптимальные гиперпараметры для модели кластеризации с использованием Optuna.

    Args:
      model_class: Type[Any]. Класс модели кластеризации (например, KMeans, AgglomerativeClustering).
      param_grid: Dict[str, Any]. Словарь с сеткой гиперпараметров для подбора.
      Значениями могут быть кортежи (для integer параметров) или списки (для categorical параметров).
      X: np.ndarray. Массив данных для кластеризации.
      n_trials: int. Количество попыток подбора гиперпараметров (по умолчанию 100).

    Returns:
      Tuple[Dict[str, Any], float]. Кортеж, содержащий:
        - Словарь с наилучшими гиперпараметрами, найденными Optuna.
        - Значение метрики качества (силуэтного коэффициента) для наилучших гиперпараметров.
    """
    # Настройка логгера Optuna
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    # Создание исследования
    study = optuna.create_study(direction='maximize')
    # Подавление предупреждений о сходимости и запуск исследования
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        study.optimize(lambda trial: objective_claster(trial, model_class, param_grid, X), n_trials=n_trials)
    # Возвращение наилучших гиперпараметров и значения метрики качества
    return study.best_params, study.best_value


# Функция objective для подбора гиперпараметров
def objective_classif(
        trial: optuna.Trial,
        model_class: Type[Any],
        param_grid: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
) -> float:
    """Функция objective для подбора гиперпараметров модели классификации с использованием Optuna.

    Args:
      trial: optuna.Trial. Объект Trial Optuna, представляющий текущую попытку подбора гиперпараметров.
      model_class: Type[Any]. Класс модели классификации (например, LogisticRegression, SVC).
      param_grid: Dict[str, Any]. Словарь с сеткой гиперпараметров для подбора. Значениями могут быть кортежи (для integer параметров) или списки (для categorical параметров).
      X: np.ndarray. Массив признаков для обучения.
      y: np.ndarray. Массив меток классов для обучения.

    Returns:
      float. Значение средней точности (accuracy) на кросс-валидации для модели с текущими гиперпараметрами.  Возвращает -1, если точность равна NaN.

    Note:
      Использует StratifiedKFold для кросс-валидации с 5 разбиениями и перемешиванием данных.
    """
    # Настройка логгера Optuna
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    # Подбор гиперпараметров
    params = {}
    for param_name, param_range in param_grid.items():
        if isinstance(param_range, tuple):
            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
        else:
            params[param_name] = trial.suggest_categorical(param_name, param_range)
    # Создание модели с подобранными гиперпараметрами
    model = model_class(**params)
    # Оценка модели с помощью кросс-валидации
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, scoring='accuracy', cv=cv).mean()
    # Проверка на NaN значения
    if np.isnan(score):
        return -1  # Возвращаем отрицательное значение, если метрика возвращает NaN
    return score

# Функция для нахождения оптимальных гиперпараметров OPTUNA - классификация
def optimize_hyperparameters_classif(
        model_class: Type[Any],
        param_grid: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50
) -> Tuple[Dict[str, Any], float]:
    """Находит оптимальные гиперпараметры для модели классификации с использованием Optuna.

    Args:
      model_class: Type[Any]. Класс модели классификации (например, LogisticRegression, SVC).
      param_grid: Dict[str, Any]. Словарь с сеткой гиперпараметров для подбора.  Значениями могут быть кортежи (для integer параметров) или списки (для categorical параметров).
      X: np.ndarray. Массив признаков для обучения.
      y: np.ndarray. Массив меток классов для обучения.
      n_trials: int. Количество попыток подбора гиперпараметров (по умолчанию 50).

    Returns:
      Tuple[Dict[str, Any], float]. Кортеж, содержащий:
        - Словарь с наилучшими гиперпараметрами, найденными Optuna.
        - Значение метрики качества (средняя точность на кросс-валидации) для наилучших гиперпараметров.
    """
    # Настройка логгера Optuna
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    # Создание исследования
    study = optuna.create_study(direction='maximize')
    # Подавление предупреждений о сходимости и запуск ислледования
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        study.optimize(lambda trial: objective_classif(trial, model_class, param_grid, X, y), n_trials=n_trials)
    # Возвращение наилучших гиперпараметров и значения метрики качества
    return study.best_params, study.best_value

# Обработка ограничений строк
async def processing_limit_str(data: List[List[Any]], str_limit: str) -> List[List[Any]]:
    """Обрабатывает список строк, фильтруя его на основе заданного диапазона и условия наличия строки "Предел".

    Args:
      data: List[List[Any]]. Список строк, где каждая строка является списком элементов.  Предполагается, что первый элемент каждой строки (data[i][0]) является числом.
      str_limit: str. Строка, представляющая диапазон в формате "start:end", где start и end - целые числа.

    Returns:
      List[List[Any]]. Отфильтрованный список строк.  Строка включается в отфильтрованный список, если:
        - Первый элемент строки находится в заданном диапазоне (start <= data[i][0] <= end)
        - ИЛИ последний элемент строки равен "Предел".

    Prints:
      Сообщения в консоль, указывающие, был ли найден указанный диапазон данных.
    """
    start_index, end_index = map(int, str_limit.split(':'))
    filtered_data = []
    for row in data:
        if start_index <= row[0] <= end_index or row[-1] == "Предел":
            filtered_data.append(row)
    if len(filtered_data)>0:
        print("\tУказаный диапазон был найден. Будут возвращены необходимые данные")
        return filtered_data
    print("\tУказаный диапазон не был найден. Будут возвращены все данные")
    return data

# Обработка ограничений меток
async def processing_limit_label(data: List[List[Any]], label_limit: str) -> List[List[Any]]:
    """Обрабатывает список строк, фильтруя его на основе заданных меток.

    Args:
      data: List[List[Any]]. Список строк, где каждая строка является списком элементов.  Предполагается, что последний элемент каждой строки (data[i][-1]) является меткой.
      label_limit: str. Строка, представляющая метку для фильтрации.  Если значение равно "Все", возвращаются строки с метками "Норма", "Перегрузка" или "Предел".

    Returns:
      List[List[Any]]. Отфильтрованный список строк.  Строка включается в отфильтрованный список, если:
        - `label_limit` равно "Все" и последний элемент строки равен "Норма", "Перегрузка" или "Предел".
        - ИЛИ последний элемент строки равен `label_limit` или "Предел".

    Prints:
      Сообщения в консоль, указывающие, были ли найдены указанные метки.
    """
    if label_limit=="Все": filtered_data = [row for row in data if row[-1] == "Норма" or row[-1] == "Перегрузка" or row[-1] == "Предел"]
    else: filtered_data = [row for row in data if row[-1] == label_limit or row[-1] == "Предел"]
    if len(filtered_data)>0:
        print("\tУказаные метки были найдены. Будут возвращены необходимые данные")
        return filtered_data
    print("\tУказаные метки не были найдены. Будут возвращены все данные")
    return data
