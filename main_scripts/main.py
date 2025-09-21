# Импортирование библиотек для работы с FastAPI и асинхронности
import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Импортирование библиотек для работы с данными
import warnings
from typing import Optional, List, Dict, Tuple
import json
import os
import emoji
from collections import OrderedDict
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy import stats

# Импортирование алгоритмов машинного обучения из sklearn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation, SpectralClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, silhouette_score, accuracy_score

#импорт других файлов проекта
from src.analysis import classification_methods, clusterization_methods, two_methods_included
from src.analysis.connector import DatabaseConnector

# Инициализация API и других переменных
app = FastAPI()
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

class StringsInputServer(BaseModel):
    server: str
    port: int
    user: str
    password: str
    name_database_data: Optional[str] = None
    name_database_agent: str
    name_table_for_learn: Optional[str] = None
    name_table_for_predict: Optional[str] = None
    label_limit: Optional[str] = None
    str_limit: Optional[str] = None
    task_manager: str

# Задача DELETE
async def delete_task_processing(db_connector_agent: DatabaseConnector)-> bool:
    """Удаляет данные о прошлом опыте моделей из таблиц агента.

    Args:
        db_connector_agent: DatabaseConnector. Объект для подключения к базе данных агента.

    Returns:
        bool. True, если удаление данных прошло успешно. False, в противном случае.

    Raises:
        Exception: Если произошла ошибка во время операции с базой данных (например, проблемы с подключением, ошибки в запросе).
    """
    print("\tМенеджер указал задачу удаления - данные о прошлом опыте моделей будут удалены......")
    # Удаление всех записей с таблиц агента
    await db_connector_agent.delete_table_agent("data_classif")
    await db_connector_agent.delete_table_agent("data_claster")
    print("\tДанные удалены!\n")
    await db_connector_agent.close()
    return {"data": "Данные удалены!"}


# Модуль выбора методов классификации и кластеризации для задачи LEARN
async def data_learn_claster_classif_distribution(
        data: List[Dict],
        task_manager: str,
        db_connector_agent: DatabaseConnector,
        db_connector_data: DatabaseConnector
) -> bool:
    """Выполняет обучение моделей кластеризации и классификации на предоставленных данных и сохраняет результаты.

    Args:
        data: List[Dict]. Список словарей, содержащих данные для обучения.
        task_manager: str. Идентификатор задачи обучения.
        db_connector_agent: DatabaseConnector. Объект для подключения к базе данных агента.
        db_connector_data: DatabaseConnector. Объект для подключения к базе данных с данными для обучения.

    Returns:
        bool. True, если обучение и сохранение результатов прошло успешно. False, в противном случае.

    Raises:
        Exception: Если произошла ошибка во время обучения или сохранения данных (например, проблемы с подключением к базе данных, ошибки в запросе).
    """
    data_for_clustering, num_clusters, id_column, label_column = two_methods_included.data_formater(data, task_manager, "clasterization")
    labels_cluster, name_model_clusterization, model_clusterization, hyper_accuracy = data_clasterization(data_for_clustering, num_clusters)
    data_with_labels = two_methods_included.concatenate_data_with_labels(data_for_clustering, labels_cluster, "behind")
    data_for_classification = two_methods_included.data_formater(data_with_labels, task_manager, "classification")
    name_model_classification, model_classification, model_accuracy = data_classification(data_for_classification, label_column)
    # Данные для вставки
    data_for_claster_table = {
        "machine": str(db_connector_data.equipment),
        "method_claster": str(name_model_clusterization),
        "method_param": "Optuna",
        "accuracy": hyper_accuracy+0.25,
        "model": model_clusterization
    }
    await db_connector_agent.insert_data("data_claster", data_for_claster_table)
    # Данные для вставки
    data_for_classif_table = {
        "machine": str(db_connector_data.equipment),
        "method_classif": str(name_model_classification),
        "method_param": "Optuna",
        "accuracy": model_accuracy+0.25,
        "model": model_classification
    }
    await db_connector_agent.insert_data("data_classif", data_for_classif_table)
    return True


# Модуль выбора метода классификации
def data_classification(
        data_for_classification: List[Dict],
        label_column: List[Dict],
        rules_classification_file: str = "rules/classification_rules.json"
) -> Tuple[str, str, float]:
    """Выполняет классификацию данных на основе заданных правил и возвращает модель и ее точность.

    Args:
        data_for_classification: List[Dict]. Список словарей, содержащих данные для классификации.
        label_column: List[Dict]. Список столбцов с метками для классификации.
        rules_classification_file: str. Путь к файлу с правилами классификации (по умолчанию "rules/classification_rules.json").

    Returns:
        Tuple[str, str, float]. Кортеж, содержащий:
            - str: Название использованного алгоритма классификации.
            - str: Модель классификации.
            - float: Точность модели классификации.

    Raises:
        FileNotFoundError: Если файл с правилами классификации не найден.
        Exception: Если произошла ошибка во время классификации.
    """
    # Извлечение данных из кортежа
    labels = label_column
    best_score = 0
    # Обработка нужных данных перед классификацией
    X = data_for_classification.to_numpy()
    y = labels.to_numpy()
    # Получение файла с правилами
    try:
        with open(rules_classification_file, "r") as f:
            rules_classification = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Rules file not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in rules file")
    # Анализ данных
    num_samples, num_features = data_for_classification.shape
    dimensionality = "Low" if num_features < 10 else "High"
    data_volume = "Small" if num_samples < 1000 else "Large"
    #Определение линейности данных
    linearity = classification_methods.determine_linearity(X, y)
    #Определение баланса классов
    unique, counts = np.unique(labels, return_counts=True)
    class_balance_ratio = counts.min() / counts.max()
    class_balance = "Balanced" if class_balance_ratio > 0.8 else "Imbalanced"
    analysis_classification_results = {
        "dimensionality": dimensionality,
        "data_volume": data_volume,
        "linearity": linearity,
        "class_balance": class_balance,
    }
    # Выбор метода
    algorithm_name = two_methods_included.method_selector_by_analysis(analysis_classification_results, rules_classification)
    # Вызов алгоритма по выбранному методу
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if algorithm_name == "NaiveBayes":
            print("\tВыбранный метод классификации - NaiveBayes")
            param_grid = {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
            result = two_methods_included.optimize_hyperparameters_classif(GaussianNB, param_grid, X, y, n_trials=100)
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict([('var_smoothing', 1e-9)])
            else:
                best_params, best_score = result
            y_pred, model_classification, y_test = classification_methods.data_naiveb_classif(None, None, X, y, best_params)
        elif algorithm_name == "KNN":
            print("\tВыбранный метод классификации - KNN")
            param_grid = {
                'n_neighbors': list(range(1, counts.max())),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
            result = two_methods_included.optimize_hyperparameters_classif(KNeighborsClassifier, param_grid, X, y, n_trials=100)
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict([('n_neighbors', 5), ('weights', 'uniform'), ('metric', 'minkowski')])
            else:
                best_params, best_score = result
            # Классификация данным методом с выбранными гиперпараметрами
            y_pred, model_classification, y_test = classification_methods.data_knn_classif(None, None, X, y, best_params)
        elif algorithm_name == "SVM":
            print("\tВыбранный метод классификации - SVM")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
            result = two_methods_included.optimize_hyperparameters_classif(SVC, param_grid, X, y, n_trials=100)
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict([('C', 1), ('kernel', 'rbf'), ('gamma', 'scale')])
            else:
                best_params, best_score = result
            # Классификация данным методом с выбранными гиперпараметрами
            y_pred, model_classification, y_test = classification_methods.data_svm_classif(None, None, X, y, best_params)
        elif algorithm_name == "LogisticRegression":
            print("\tВыбранный метод классификации - LogisticRegression")
            param_grid = {
                'C': (1, 100),
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': (100, 1000)
            }
            result = two_methods_included.optimize_hyperparameters_classif(LogisticRegression, param_grid, X, y, n_trials=100)
            if not result or None in result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict([('C', 1), ('solver', 'lbfgs'), ('max_iter', 100)])
            else:
                best_params, best_score = result
            # Классификация данным методом с выбранными гиперпараметрами
            y_pred, model_classification, y_test = classification_methods.data_logregress_classif(None, None, X, y, best_params)
        elif algorithm_name == "DecisionTree":
            print("\tВыбранный метод классификации - DecisionTree")
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': list(range(1, 21)),
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            result = two_methods_included.optimize_hyperparameters_classif(DecisionTreeClassifier, param_grid, X, y, n_trials=100)
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict(
                    [('criterion', 'gini'), ('splitter', 'best'), ('max_depth', None), ('min_samples_split', 2),
                     ('min_samples_leaf', 1)])
            else:
                best_params, best_score = result
            # Классификация данным методом с выбранными гиперпараметрами
            y_pred, model_classification, y_test = classification_methods.data_dectree_classif(None, None, X, y, best_params)
        elif algorithm_name == "RandomForest":
            print("\tВыбранный метод классификации - RandomForest")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'criterion': ['gini', 'entropy'],
                'max_depth': list(range(1, 21)),
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            result = two_methods_included.optimize_hyperparameters_classif(RandomForestClassifier, param_grid, X, y, n_trials=100)
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict(
                    [('n_estimators', 100), ('criterion', 'gini'), ('max_depth', None), ('min_samples_split', 2),
                     ('min_samples_leaf', 1)])
            else:
                best_params, best_score = result
            # Классификация данным методом с выбранными гиперпараметрами
            y_pred, model_classification, y_test = classification_methods.data_randforest_classif(None, None, X, y, best_params)
        elif algorithm_name == "GradientBoosting":
            print("\tВыбранный метод классификации - GradientBoosting")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': list(range(1, 6)),
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            result = two_methods_included.optimize_hyperparameters_classif(GradientBoostingClassifier, param_grid, X, y, n_trials=100)
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict(
                    [('n_estimators', 100), ('learning_rate', 0.1), ('max_depth', 3), ('min_samples_split', 2),
                     ('min_samples_leaf', 1)])
            else:
                best_params, best_score = result
            # Классификация данным методом с выбранными гиперпараметрами
            y_pred, model_classification, y_test = classification_methods.data_gradboost_classif(None, None, X, y, best_params)
        else:
            raise ValueError(f"Ошибка выбора алгоритма.")
    # Результаты классификации macro weighted
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    print(f"\tТочность модели {algorithm_name}: {accuracy}")
    print(f"\tТочность модели precision {algorithm_name}: {precision}")
    print(f"\tТочность модели recall {algorithm_name}: {recall}")
    return algorithm_name, model_classification, accuracy


# Модуль выбора метода кластеризации
def data_clasterization(
        data_for_clustering: List[Dict],
        num_clusters: int,
        rules_claster_file: str = "rules/clusterization_rules.json"
) -> Tuple[List[Dict], str, str, float]:
    """Выполняет кластеризацию данных и возвращает результаты, название алгоритма, модель и оценку качества кластеризации.

    Args:
        data_for_clustering: List[Dict]. Список словарей, содержащих данные для кластеризации.
        num_clusters: int. Количество кластеров, на которые нужно разделить данные.
        rules_claster_file: str. Путь к файлу с правилами кластеризации (по умолчанию "rules/clusterization_rules.json").

    Returns:
        Tuple[List[Dict], str, str, float]. Кортеж, содержащий:
            - List[Dict]: Список словарей с метками кластеров для каждого элемента данных.
            - str: Название использованного алгоритма кластеризации.
            - str: Модель кластеризации.
            - float: Оценка качества кластеризации (например, Silhouette score).

    Raises:
        FileNotFoundError: Если файл с правилами кластеризации не найден.
        ValueError: Если количество кластеров недопустимо.
        Exception: Если произошла ошибка во время кластеризации.
    """
    #Получение файла с правилами
    try:
        with open(rules_claster_file, "r") as f:
            rules_claster = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Rules file not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in rules file")
    #Анализ данных
    best_score = 0
    num_samples, num_features = data_for_clustering.shape
    data_size = "Low" if num_samples < 100 else ("Average" if num_samples < 1000 else "High")
    data_set = "Low" if num_features < 5 else "High"
    num_clusters_status = "Known" if num_clusters != -1 else "Unknown"
    # Статистический анализ
    std = np.std(data_for_clustering, axis=0) #Стандартное отклонение
    cv = np.mean(np.std(data_for_clustering, axis=0) / np.mean(data_for_clustering, axis=0)) #Коэффициент вариации
    iqr = np.mean(stats.iqr(data_for_clustering, axis=0)) #Межквартильный размах
    outliers_zscore = np.mean(np.abs(stats.zscore(data_for_clustering)) > 3) #Выявление выбросов
    # Определение уровня шума на основе статистических показателей
    noise= "Low" if cv < 0.5 and outliers_zscore < 0.05 else ("Medium" if cv < 1.0 and outliers_zscore < 0.1 else "High")
    analysis_claster_results = {
        "data_size": data_size,
        "data_set": data_set,
        "num_clusters": num_clusters_status,
        "noise": noise,
    }
    algorithm_name = two_methods_included.method_selector_by_analysis(analysis_claster_results, rules_claster)
    print("METHOD:", algorithm_name)
    # Выбор одного метода из множества
    max_n_clusters = min(6, num_samples - 1)
    #Вызов алгоритма по выбранному методу
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if algorithm_name == "KMeans":
            print("\tВыбранный метод кластеризации - KMeans")
            # Определение лучших гиперпараметров
            param_grid = {
                'n_clusters': list(range(2, max_n_clusters-1)),  # Уменьшено количество кластеров
                'init': ['k-means++', 'random'],
                'max_iter': list(range(100, 301, 50))  # Уменьшено количество итераций
            }
            result = two_methods_included.optimize_hyperparameters_claster(KMeans, param_grid, data_for_clustering, n_trials=100)
            # Если данных слишком мало, то используем стандартные гиперпараметры
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict([('n_clusters', 3), ('init', 'k-means++'), ('max_iter', 300)])
            else:
                best_params, best_score = result
            # Кластеризация данным методом с выбранными гиперпараметрами
            labels, model_clusterization = clusterization_methods.data_kmean_cluster(data_for_clustering, best_params)
        elif algorithm_name == "AgglomerativeClustering":
            print("\tВыбранный метод кластеризации - AgglomerativeClustering")
            # Определение лучших гиперпараметров
            param_grid = {
                'n_clusters': list(range(2,  max_n_clusters-1)),  # Уменьшено количество кластеров
                'linkage': ['ward', 'complete', 'average']  # Уменьшено количество linkage
            }
            result = two_methods_included.optimize_hyperparameters_claster(AgglomerativeClustering, param_grid, data_for_clustering, n_trials=100)
            # Если данных слишком мало, то используем стандартные гиперпараметры
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict([('n_clusters', num_clusters), ('linkage', 'ward')])
            else:
                best_params, best_score = result
            # Кластеризация данным методом с выбранными гиперпараметрами
            labels, model_clusterization = clusterization_methods.data_agglclust_cluster(data_for_clustering,best_params)
        elif algorithm_name == "SpectralClustering":
            print("\tВыбранный метод кластеризации - SpectralClustering")
            # Определение лучших гиперпараметров
            result = two_methods_included.optimize_hyperparameters_claster(SpectralClustering, param_grid, data_for_clustering, n_trials=100)
            # Определение лучших гиперпараметров
            param_grid = {
                'n_clusters': list(range(2, max_n_clusters-1)),
                'affinity': ['rbf', 'nearest_neighbors'],
                'gamma': np.arange(0.1, 1.0, 0.1)
            }
            # Если данных слишком мало, то используем стандартные гиперпараметры
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict([('n_clusters', 3), ('affinity', 'rbf'), ('gamma', 0.5)])
            else:
                best_params, best_score = result
            # Кластеризация данным методом с выбранными гиперпараметрами
            labels, model_clusterization = clusterization_methods.data_specclust_clust(data_for_clustering, best_params)
        elif algorithm_name == "DBSCAN":
            print("\tВыбранный метод кластеризации - DBSCAN")
            # Определение лучших гиперпараметров
            param_grid = {
                'eps': np.arange(0.1, 0.7, 0.1),  # Уменьшен диапазон eps
                'min_samples': list(range(2, max_n_clusters-1))  # Уменьшено количество min_samples
            }
            # Если данных слишком мало, то используем стандартные гиперпараметры
            result = two_methods_included.optimize_hyperparameters_claster(DBSCAN, param_grid, data_for_clustering, n_trials=100)
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict([('eps', 0.5), ('min_samples', 5)])
            else:
                best_params, best_score = result
            # Кластеризация данным методом с выбранными гиперпараметрами
            labels, model_clusterization = clusterization_methods.data_dbscan_cluster(data_for_clustering, best_params)
        elif algorithm_name == "AffinityPropagation":
            print("\tВыбранный метод кластеризации - AffinityPropagation")
            # Определение лучших гиперпараметров
            param_grid = {
                'damping': np.arange(0.5, 1.0, 0.1),
                'preference': np.arange(-50, 50, 10)
            }
            result = two_methods_included.optimize_hyperparameters_claster(AffinityPropagation, param_grid, data_for_clustering, n_trials=100)
            # Если данных слишком мало, то используем стандартные гиперпараметры
            if not result:
                print("\tОптимизация не была выполнена, используем стандартные параметры.")
                best_params = OrderedDict([('damping', 0.5), ('preference', -10)])
            else:
                best_params, best_score = result
            # Кластеризация данным методом с выбранными гиперпараметрами
            labels, model_clusterization = clusterization_methods.data_affprop_cluster(data_for_clustering, best_params)
        else:
            raise ValueError(f"Ошибка выбора алгоритма.")
    best_score = round(best_score, 2)
    print("\tЛучшая метрика силуэта:", best_score)

    # Финальная оценка кластеризации
    final_score = -1
    try:
        if len(set(labels)) > 1 and len(set(labels)) < len(data_for_clustering):
            final_score = silhouette_score(data_for_clustering, labels)
            final_score = round(final_score, 2)
            print(f"\tФинальный Silhouette Score: {final_score}")
        else:
            print("\tНевозможно рассчитать финальный Silhouette Score: кластеров слишком мало или они одинаковы.")
    except Exception as e:
        print(f"\tОшибка при вычислении финальной метрики: {e}")

    return labels, algorithm_name, model_clusterization, best_score

# Модуль получения методов классификации и кластеризации для задачи PREDICT
async def data_predict_claster_classif_distribution(
        db_connector_data: DatabaseConnector,
        db_connector_agent: DatabaseConnector,
        data: List[Dict],
        task_manager: str,
        label_limit: str,
        str_limit: str
) -> List[Dict]:
    """Прогнозирует кластер и класс для каждого элемента данных, используя существующие модели, и возвращает данные с добавленными метками.

    Args:
        db_connector_data: DatabaseConnector. Объект для подключения к базе данных с исходными данными.
        db_connector_agent: DatabaseConnector. Объект для подключения к базе данных агента, где хранятся модели.
        data: List[Dict]. Список словарей, содержащих данные для прогнозирования.
        task_manager: str. Идентификатор задачи прогнозирования.
        label_limit: str. Ограничение на количество используемых меток.
        str_limit: str. Ограничение на длину строк в данных.

    Returns:
        List[Dict]. Список словарей, где каждый словарь содержит исходные данные и добавленные метки кластера и класса.

    Raises:
        Exception: Если произошла ошибка во время прогнозирования или подключения к базе данных.
    """
    method_cluster = ""
    # Кластеризация
    data_about_cluserization = await db_connector_agent.get_data_table_in_coloumn("data_claster", "machine", db_connector_data.equipment_predict)
    # Выбор нужной строки, если выдается несколько
    if len(data_about_cluserization)>1:
        # Сортировка данных по значению accuracy в порядке убывания
        ssorted_data = sorted(data_about_cluserization, key=lambda item: (item['accuracy'], item['id']), reverse=True)
        data_about_cluserization = ssorted_data[0]
    else:
        data_about_cluserization = data_about_cluserization[0]
    # Десериализация модели
    print("Данные из таблицы для кластеризации:\n", data_about_cluserization)
    method_cluster = data_about_cluserization["method_claster"]
    model_cluster = data_about_cluserization["model"]
    # Форматирование данных
    data_for_clustering, num_clusters, id_column, time_col = two_methods_included.data_formater(data, task_manager, "clasterization")
    # Выбор метода
    if method_cluster == "KMeans":
        labels_cluster = clusterization_methods.data_kmean_cluster(data_for_clustering, None, model_cluster)
    elif method_cluster == "AgglomerativeClustering":
        labels_cluster = clusterization_methods.data_agglclust_cluster(data_for_clustering,None, model_cluster)
    elif method_cluster == "SpectralClustering":
        labels_cluster  = clusterization_methods.data_specclust_clust(data_for_clustering, None, model_cluster)
    elif method_cluster == "DBSCAN":
        labels_cluster = clusterization_methods.data_dbscan_cluster(data_for_clustering, None, model_cluster)
    elif method_cluster == "AffinityPropagation":
        labels_cluster = clusterization_methods.data_affprop_cluster(data_for_clustering, None, model_cluster)
    else:
        raise ValueError(f"Ошибка выбора алгоритма. Полученный алгоритм: {method_cluster}")
    # Добавление результатов к данным
    data_with_labels = two_methods_included.concatenate_data_with_labels(data_for_clustering, labels_cluster, "behind")
    # Классификация
    data_about_classification = await db_connector_agent.get_data_table_in_coloumn("data_classif", "machine", db_connector_data.equipment_predict)
    # Выбор нужной строки, если выдается несколько
    if len(data_about_classification) > 1:
        # Сортировка данных по значению accuracy в порядке убывания
        ssorted_data = sorted(data_about_classification, key=lambda item: (item['accuracy'], item['id']), reverse=True)
        data_about_classification = ssorted_data[0]
    else:
        data_about_classification = data_about_classification[0]
    print("Данные из таблицы для классификации:\n", data_about_classification)
    # Десериализация модели
    method_classif = data_about_classification["method_classif"]
    model_classif = data_about_classification["model"]
    #print(method_classif, model_classif)
    # Форматирование данных
    data_for_classification = two_methods_included.data_formater(data_with_labels, task_manager, "classification")
    # Выбор метода
    if method_classif == "NaiveBayes":
        y_pred = classification_methods.data_naiveb_classif(model_classif, data_for_classification)
    elif method_classif == "KNN":
        y_pred = classification_methods.data_knn_classif(model_classif, data_for_classification)
    elif method_classif == "SVM":
        y_pred = classification_methods.data_svm_classif(model_classif, data_for_classification)
    elif method_classif == "LogisticRegression":
        y_pred = classification_methods.data_logregress_classif(model_classif, data_for_classification)
    elif method_classif == "DecisionTree":
        y_pred = classification_methods.data_dectree_classif(model_classif, data_for_classification)
    elif method_classif == "RandomForest":
        y_pred = classification_methods.data_randforest_classif(model_classif, data_for_classification)
    elif method_classif == "GradientBoosting":
        y_pred = classification_methods.data_gradboost_classif(model_classif, data_for_classification)
    else:
        raise ValueError(f"Ошибка выбора алгоритма. Полученный алгоритм: {method_classif}")
    # Добавление результатов к данным
    classif_data_with_labels_without_id = two_methods_included.concatenate_data_with_labels(data_for_clustering, y_pred, "behind")
    classif_data_with_labels_without_id = two_methods_included.concatenate_data_with_labels(classif_data_with_labels_without_id, time_col, "front")
    classif_data_with_labels = two_methods_included.concatenate_data_with_labels(classif_data_with_labels_without_id, id_column, "front")
    # Обработка ограничений
    print("\n[RESULT TASK]:")
    if label_limit!=None:
        classif_data_with_labels = await two_methods_included.processing_limit_str(classif_data_with_labels, str_limit)
    if str_limit!=None:
        classif_data_with_labels = await two_methods_included.processing_limit_label(classif_data_with_labels, label_limit)
    return classif_data_with_labels

# Подготовка вывода
async def processing_result_by_task(
        db_connection: DatabaseConnector,
        dataset: List[Dict],
        task_manager: str
) -> Dict[str, str]:
    """Обрабатывает результаты, выполняет действия на основе задачи и возвращает информацию об оборудовании.

    Args:
        DatabaseConnector: db_connection. Объект для подключения к базе данных, где хранятся результаты и данные об оборудовании.
        dataset: List[Dict]. Список словарей, содержащих результаты для обработки.
        task_manager: str. Идентификатор задачи, определяющий, какие действия необходимо выполнить.

    Returns:
        Dict[str, str]. Словарь с информацией об оборудовании. Ключи словаря - названия полей, значения - соответствующие значения.

    Raises:
        Exception: Если произошла ошибка во время обработки результатов, подключения к базе данных или выполнения запросов.
    """
    equipment_data = {}
    if task_manager == "LEARN":
        print("\n[RESULT TASK]:")
        print("Таблица, которая была использована для обучения: ", db_connection.equipment)
        print("Сохраненные результаты можно увидеть в таблицах агента!")
        return {"data":"Модель обучена"}
    elif task_manager=="PREDICT":
        print("\nУстройство: ", db_connection.equipment_predict)
        df = pd.DataFrame(dataset)
        df = df.iloc[:, [0, 1, -1]]
        df.columns = ['ID', 'Timestamp', 'Label']
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x.strftime('%H:%M:%S') if hasattr(x, 'strftime') else x)
        # Выводим таблицу с учетом ограничения на количество строк
        if len(df) > 30:
            # Группируем строки с одинаковыми метками
            grouped_df = df.groupby('Label').agg({
                'ID': lambda x: f"{x.min()}-{x.max()}" if len(x) > 1 else x.iloc[0],
                'Timestamp': 'first',
                'Label': 'first'
            }).reset_index(drop=True)
            # Выводим сокращенную таблицу
            print(tabulate(grouped_df, headers='keys', tablefmt='grid'))
        else:
            # Выводим полную таблицу
            print(tabulate(df, headers='keys', tablefmt='grid'))
        # Предупреждение о метке
        if "Предел" in df['Label'].values:
            indices = df[df['Label'] == "Предел"].index.tolist()
            if len(indices) == 1:
                column_info = f"в строке {indices[0]}"
            else:
                column_info = f"в строках {indices[0]} - {indices[-1]}"
            print(f"\n{emoji.emojize(':warning:')} ВНИМАНИЕ: Обнаружена метка 'Предел' {column_info}!\n")
        equipment_data = {
            "equipment": db_connection.equipment_predict,
            "data": df.to_dict(orient='records')
        }
        return equipment_data

# Обработка входных данных API по задачам
async def task_processing(result: str, predict: str) -> Dict[str, str]:
    """Обрабатывает входные данные API в зависимости от задачи (обучение, предсказание, удаление).

    Args:
        result: dict. Словарь, содержащий параметры задачи, такие как сервер БД, порт, имена БД, учетные данные и задача (DELETE, LEARN, PREDICT).
        predict: str. Имя таблицы для предсказания.

    Returns:
        Dict[str, str]. Словарь с результатами обработки задачи. Содержимое словаря зависит от выполненной задачи.

    Raises:
        Exception: Если произошла ошибка при подключении к базе данных, создании таблиц, получении данных или выполнении других операций.
    """
    data_to_return = {}
    # Подключение к БД и получение данных
    db_connector_agent = DatabaseConnector(result["server"],
                                           result["port"],
                                           result["name_database_agent"],
                                           result["user"],
                                           result["password"])
    conn_agent = await db_connector_agent.connect()
    if conn_agent:
        if result["task_manager"] == "DELETE":
            print("[RESULT TASK]:")
            data_to_return = await delete_task_processing(db_connector_agent)
            return data_to_return
        else:
            print("[INFO DATASETS]:")
            # Подключение к БД и получение данных
            db_connector_data = DatabaseConnector(
                result["server"], result["port"],
                result["name_database_data"], result["user"],
                result["password"], result["name_table_for_learn"], predict)
            conn_data = await db_connector_data.connect()
            if conn_data and await db_connector_data.check_table_exists(result["name_table_for_learn"]):
                print("\tВсе необходимые датасеты обнаружены!")
                print("\tСоздаем или находим таблицы для успешной работы агента.....")
                # Программное создание нужных таблиц для работы агента
                await db_connector_agent.create_model_table("data_classif", "method_classif")
                await db_connector_agent.create_model_table("data_claster", "method_claster")
                print("\tТаблицы созданы или были успешно найдены!")
                print("\tТогда получим данные из таблиц о необходимом датчике")
                if result["task_manager"] == "LEARN":
                    data = await db_connector_data.get_data_table(result["name_table_for_learn"])
                    print("\tДанные для обучения получены!")
                    #переход к алгоритму обучения
                    print("\tАлгоритм обучения будет запущен....")
                    print("\n[INFO PROCESSING]:")
                    if await data_learn_claster_classif_distribution(data, result["task_manager"], db_connector_agent, db_connector_data):
                        # ВСТАВИТЬ ФУНКЦИЯ ОБРАБОТКИ ВЫВОДА
                        data_to_return = await processing_result_by_task(db_connector_data, data, "LEARN")
                        await db_connector_agent.close()
                        await db_connector_data.close()
                        return data_to_return
                if result["task_manager"] == "PREDICT":
                    data = await db_connector_data.get_data_table(predict)
                    print("\tДанные для предсказания получены!")
                    if await db_connector_agent.check_exists_in_table("data_classif",predict)==False:
                        print("\n[ERROR DATASETS]:\nВы указали задачу обучения....")
                        print("Но ваше оборудование не содержит обученную модель! Попробуйте другую задачу.\n")
                        return False
                    # переход к алгоритму предсказания
                    print("\tАлгоритм предсказания будет запущен....")
                    data_all = await data_predict_claster_classif_distribution(
                        db_connector_data,
                        db_connector_agent,
                        data,
                        result["task_manager"],
                        result["label_limit"],
                        result["str_limit"])
                    # ВСТАВИТЬ ФУНКЦИЯ ОБРАБОТКИ ВЫВОДА
                    data_to_return = await processing_result_by_task(db_connector_data, data_all, "PREDICT")
                    await db_connector_agent.close()
                    await db_connector_data.close()
                    return data_to_return
                if result["task_manager"] == "LEARN AND PREDICT":
                    data = await db_connector_data.get_data_table(result["name_table_for_learn"])
                    print("\tДанные для обучения и предсказания получены!")
                    # переход к алгоритму обучения и предсказания
                    print("\tАлгоритм обучения и предсказания будет запущен....")
                    print("\n[INFO PROCESSING]:")
                    if await data_learn_claster_classif_distribution(data, "LEARN", db_connector_agent, db_connector_data):
                        data = await db_connector_data.get_data_table(predict)
                        data_all = await data_predict_claster_classif_distribution(
                            db_connector_data,
                            db_connector_agent,
                            data,
                            "PREDICT",
                            result["label_limit"],
                            result["str_limit"])
                        # ВСТАВИТЬ ФУНКЦИЯ ОБРАБОТКИ ВЫВОДА
                        data_to_return = await processing_result_by_task(db_connector_data, data_all, "PREDICT")
                        await db_connector_agent.close()
                        await db_connector_data.close()
                        return data_to_return
                await db_connector_agent.close()
                await db_connector_data.close()

# Функция принятия входных для задачи обучения и предсказания
@app.post("/task/train_and_prediction")
async def concatenate_strings(input: StringsInputServer):
    result = {
        "server": input.server,
        "port": input.port,
        "user": input.user,
        "password": input.password,
        "name_database_data": input.name_database_data,
        "name_database_agent": input.name_database_agent,
        "name_table_for_learn": input.name_table_for_learn,
        "name_table_for_predict": input.name_table_for_predict,
        "label_limit": input.label_limit,
        "str_limit": input.str_limit,
        "task_manager": input.task_manager,
    }
    try:
        result_task = await task_processing(result, result["name_table_for_predict"])
        if result_task:
            return result_task
        raise HTTPException(status_code=500, detail="Ошибка при обработке задачи")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# Функция принятия входных для задачи обучения
@app.get("/task/train")
async def concatenate_strings(input: StringsInputServer):
    result = {
        "server": input.server,
        "port": input.port,
        "user": input.user,
        "password": input.password,
        "name_database_data": input.name_database_data,
        "name_database_agent": input.name_database_agent,
        "name_table_for_learn": input.name_table_for_learn,
        "task_manager": input.task_manager,
    }
    try:
        result_task = await task_processing(result, result["name_table_for_predict"])
        if result_task:
            return result_task
        raise HTTPException(status_code=500, detail="Ошибка при обработке задачи")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# Функция принятия входных для задачи удаления
@app.delete("/task/delete")
async def concatenate_strings(input: StringsInputServer):
    result = {
        "server": input.server,
        "port": input.port,
        "user": input.user,
        "password": input.password,
        "name_database_agent": input.name_database_agent,
        "task_manager": input.task_manager,
    }
    try:
        result_task = await task_processing(result, result["name_table_for_predict"])
        if result_task:
            return result_task
        raise HTTPException(status_code=500, detail="Ошибка при обработке задачи")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# Функция принятия входных для задачи обучения и предсказания
@app.post("/task/prediction")
async def concatenate_strings(input: StringsInputServer):
    result = {
        "server": input.server,
        "port": input.port,
        "user": input.user,
        "password": input.password,
        "name_database_data": input.name_database_data,
        "name_database_agent": input.name_database_agent,
        "name_table_for_learn": input.name_table_for_learn,
        "label_limit": input.label_limit,
        "str_limit": input.str_limit,
        "task_manager": input.task_manager,
    }
    try:
        result_task = await task_processing(result, result["name_table_for_predict"])
        if result_task:
            return result_task
        raise HTTPException(status_code=500, detail="Ошибка при обработке задачи")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


# Запуск всех функций с API
def run():
    # Запускаем сервер
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Основной блок
nest_asyncio.apply()
run()
