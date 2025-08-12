import pickle
from typing import Any, Optional
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def determine_linearity(X: np.ndarray, y: np.ndarray, threshold: float = 0.85) -> str:
    """Определяет, являются ли данные линейно разделимыми, используя логистическую регрессию и SVM с линейным ядром.

    Args:
        X: numpy.ndarray. Массив признаков.
        y: numpy.ndarray. Массив меток классов.
        threshold: float. Пороговое значение точности, выше которого данные считаются линейно разделимыми (по умолчанию 0.85).

    Returns:
        str. "Linearly separable", если данные считаются линейно разделимыми, или "Linearly inseparable", если нет.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Обучение логистической регрессии
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    # Обучение SVM с линейным ядром
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    y_pred_svm = svm_linear.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    if accuracy_log_reg > threshold and accuracy_svm > threshold:
        return "Linearly separable"
    else:
        return "Linearly inseparable"


def data_naiveb_classif(
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        best_params: Optional[Any] = None
) -> Any:
    """Выполняет классификацию данных с использованием наивного байесовского классификатора (GaussianNB).

    Args:
        loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для предсказания. По умолчанию None.
        data: Optional[np.ndarray]. Данные для предсказания, если используется загруженная модель. По умолчанию None.
        X: Optional[np.ndarray]. Массив признаков для обучения, если модель не загружена. По умолчанию None.
        y: Optional[np.ndarray]. Массив меток классов для обучения, если модель не загружена. По умолчанию None.
        best_params: Optional[dict]. Словарь с лучшими параметрами для GaussianNB, если модель не загружена. По умолчанию None.

    Returns:
        Any. Предсказанные метки классов (numpy.ndarray), обученная модель GaussianNB (если модель не загружена), или кортеж с предсказанными метками, обученной моделью и тестовыми метками (если модель не загружена).
    """
    if loaded_model == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_classif_naiveb = GaussianNB(**best_params)
        model_classif_naiveb.fit(X_train, y_train)
        y_pred = model_classif_naiveb.predict(X_test)
        return y_pred, model_classif_naiveb, y_test
    loaded_model = pickle.loads(loaded_model)
    loaded_model_classif_naiveb = loaded_model.predict(data)
    return loaded_model_classif_naiveb

def data_knn_classif(
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        best_params: Optional[Any] = None
) -> Any:
    """Выполняет классификацию данных с использованием алгоритма k-ближайших соседей (KNeighborsClassifier).

    Args:
        loaded_model: Optional[bytes].
            Загруженная модель в виде байтовой строки (результат pickle.dumps).
                Если указана, используются загруженные параметры для предсказания. По умолчанию None.
        data: Optional[np.ndarray]. Данные для предсказания, если используется загруженная модель. По умолчанию None.
        X: Optional[np.ndarray]. Массив признаков для обучения, если модель не загружена. По умолчанию None.
        y: Optional[np.ndarray]. Массив меток классов для обучения, если модель не загружена. По умолчанию None.
        best_params: Optional[dict]. Словарь с лучшими параметрами для GaussianNB, если модель не загружена. По умолчанию None.

    Returns:
        Any. Предсказанные метки классов (numpy.ndarray),
            обученная модель (если модель не загружена), или кортеж с предсказанными метками,
                обученной моделью и тестовыми метками (если модель не загружена).
    """
    if loaded_model == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_classif_knn = KNeighborsClassifier(**best_params)
        model_classif_knn.fit(X_train, y_train)
        y_pred = model_classif_knn.predict(X_test)
        return y_pred, model_classif_knn, y_test
    loaded_model = pickle.loads(loaded_model)
    loaded_model_classif_knn = loaded_model.predict(data)
    return loaded_model_classif_knn

def data_svm_classif(
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        best_params: Optional[Any] = None
) -> Any:
    """Выполняет классификацию данных с использованием алгоритма Support Vector Machine (SVC).

    Args:
        loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для предсказания. По умолчанию None.
        data: Optional[np.ndarray]. Данные для предсказания, если используется загруженная модель. По умолчанию None.
        X: Optional[np.ndarray]. Массив признаков для обучения, если модель не загружена. По умолчанию None.
        y: Optional[np.ndarray]. Массив меток классов для обучения, если модель не загружена. По умолчанию None.
        best_params: Optional[dict]. Словарь с лучшими параметрами для GaussianNB, если модель не загружена. По умолчанию None.

    Returns:
        Any. Предсказанные метки классов (numpy.ndarray), обученная модель GaussianNB (если модель не загружена), или кортеж с предсказанными метками, обученной моделью и тестовыми метками (если модель не загружена).
    """
    if loaded_model == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_classif_svm = SVC(**best_params)
        model_classif_svm.fit(X_train, y_train)
        y_pred = model_classif_svm.predict(X_test)
        return y_pred, model_classif_svm, y_test
    loaded_model = pickle.loads(loaded_model)
    loaded_model_classif_svm = loaded_model.predict(data)
    return loaded_model_classif_svm

def data_logregress_classif(
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        best_params: Optional[Any] = None
) -> Any:
    """Выполняет классификацию данных с использованием логистической регрессии (LogisticRegression).

    Args:
        loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для предсказания. По умолчанию None.
        data: Optional[np.ndarray]. Данные для предсказания, если используется загруженная модель. По умолчанию None.
        X: Optional[np.ndarray]. Массив признаков для обучения, если модель не загружена. По умолчанию None.
        y: Optional[np.ndarray]. Массив меток классов для обучения, если модель не загружена. По умолчанию None.
        best_params: Optional[dict]. Словарь с лучшими параметрами для GaussianNB, если модель не загружена. По умолчанию None.

    Returns:
        Any. Предсказанные метки классов (numpy.ndarray), обученная модель GaussianNB (если модель не загружена), или кортеж с предсказанными метками, обученной моделью и тестовыми метками (если модель не загружена).
    """
    if loaded_model == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_classif_loggr = LogisticRegression(**best_params)
        model_classif_loggr.fit(X_train, y_train)
        y_pred = model_classif_loggr.predict(X_test)
        return y_pred, model_classif_loggr, y_test
    loaded_model = pickle.loads(loaded_model)
    loaded_model_classif_loggr = loaded_model.predict(data)
    return loaded_model_classif_loggr

def data_dectree_classif(
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        best_params: Optional[Any] = None
) -> Any:
    """Выполняет классификацию данных с использованием дерева решений (DecisionTreeClassifier).

    Args:
        loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для предсказания. По умолчанию None.
        data: Optional[np.ndarray]. Данные для предсказания, если используется загруженная модель. По умолчанию None.
        X: Optional[np.ndarray]. Массив признаков для обучения, если модель не загружена. По умолчанию None.
        y: Optional[np.ndarray]. Массив меток классов для обучения, если модель не загружена. По умолчанию None.
        best_params: Optional[dict]. Словарь с лучшими параметрами для GaussianNB, если модель не загружена. По умолчанию None.

    Returns:
        Any. Предсказанные метки классов (numpy.ndarray), обученная модель GaussianNB (если модель не загружена), или кортеж с предсказанными метками, обученной моделью и тестовыми метками (если модель не загружена).
    """
    if loaded_model == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_classif_tree = DecisionTreeClassifier(**best_params)
        model_classif_tree.fit(X_train, y_train)
        y_pred = model_classif_tree.predict(X_test)
        return y_pred, model_classif_tree, y_test
    loaded_model = pickle.loads(loaded_model)
    loaded_model_classif_tree = loaded_model.predict(data)
    return loaded_model_classif_tree


def data_randforest_classif(
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        best_params: Optional[Any] = None
) -> Any:
    """Выполняет классификацию данных с использованием случайного леса (RandomForestClassifier).

    Args:
        loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для предсказания. По умолчанию None.
        data: Optional[np.ndarray]. Данные для предсказания, если используется загруженная модель. По умолчанию None.
        X: Optional[np.ndarray]. Массив признаков для обучения, если модель не загружена. По умолчанию None.
        y: Optional[np.ndarray]. Массив меток классов для обучения, если модель не загружена. По умолчанию None.
        best_params: Optional[dict]. Словарь с лучшими параметрами для GaussianNB, если модель не загружена. По умолчанию None.

    Returns:
        Any. Предсказанные метки классов (numpy.ndarray), обученная модель GaussianNB (если модель не загружена), или кортеж с предсказанными метками, обученной моделью и тестовыми метками (если модель не загружена).
    """
    if loaded_model == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_classif_forest = RandomForestClassifier(**best_params)
        model_classif_forest.fit(X_train, y_train)
        y_pred = model_classif_forest.predict(X_test)
        return y_pred, model_classif_forest, y_test
    loaded_model = pickle.loads(loaded_model)
    loaded_model_classif_forest = loaded_model.predict(data)
    return loaded_model_classif_forest


def data_gradboost_classif(
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        best_params: Optional[Any] = None
) -> Any:
    """Выполняет классификацию данных с использованием градиентного бустинга (GradientBoostingClassifier).

    Args:
        loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для предсказания. По умолчанию None.
        data: Optional[np.ndarray]. Данные для предсказания, если используется загруженная модель. По умолчанию None.
        X: Optional[np.ndarray]. Массив признаков для обучения, если модель не загружена. По умолчанию None.
        y: Optional[np.ndarray]. Массив меток классов для обучения, если модель не загружена. По умолчанию None.
        best_params: Optional[dict]. Словарь с лучшими параметрами для GaussianNB, если модель не загружена. По умолчанию None.

    Returns:
        Any. Предсказанные метки классов (numpy.ndarray), обученная модель GaussianNB (если модель не загружена), или кортеж с предсказанными метками, обученной моделью и тестовыми метками (если модель не загружена).
    """
    if loaded_model == None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_classif_grand = GradientBoostingClassifier(**best_params)
        model_classif_grand.fit(X_train, y_train)
        y_pred = model_classif_grand.predict(X_test)
        return y_pred, model_classif_grand, y_test
    loaded_model = pickle.loads(loaded_model)
    loaded_model_classif_grand = loaded_model.predict(data)
    return loaded_model_classif_grand
