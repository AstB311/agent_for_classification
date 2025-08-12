import pickle
from typing import Any, Optional
import numpy as np

from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering

def data_kmean_cluster(
        data: np.ndarray,
        params: Optional[dict] = None,
        loaded_model: Optional[bytes] = None
) -> Any:
    """Выполняет кластеризацию данных с использованием алгоритма K-средних (KMeans).

    Args:
      data: np.ndarray. Массив данных для кластеризации.
      params: Optional[dict]. Словарь с параметрами для KMeans, такими как 'n_clusters', 'init', 'max_iter'.  Необходим, если модель не загружена.
      loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для кластеризации.

    Returns:
      Any. Метки кластеров для каждого элемента данных (numpy.ndarray), обученная модель (если модель не загружена), или предсказанные метки кластеров (numpy.ndarray) для входных данных (если модель загружена).
    """
    if loaded_model==None and params!=None:
        model_cluster_kmeans = KMeans(n_clusters=params['n_clusters'], init=params['init'], max_iter=params['max_iter'], random_state=42)
        labels = model_cluster_kmeans.fit_predict(data)
        return labels, model_cluster_kmeans
    loaded_model = pickle.loads(loaded_model)
    loaded_model_cluster_kmeans = loaded_model.fit_predict(data)
    return loaded_model_cluster_kmeans

def data_agglclust_cluster(
        data: np.ndarray,
        params: Optional[dict] = None,
        loaded_model: Optional[bytes] = None
) -> Any:
    """Выполняет кластеризацию данных с использованием агломеративной кластеризации (AgglomerativeClustering).

    Args:
      data: np.ndarray. Массив данных для кластеризации.
      params: Optional[dict].
        Словарь с параметрами для KMeans, такими как 'n_clusters', 'init', 'max_iter'.
            Необходим, если модель не загружена.
      loaded_model: Optional[bytes].
        Загруженная модель в виде байтовой строки (результат pickle.dumps).
            Если указана, используются загруженные параметры для кластеризации.

    Returns:
      Any. Метки кластеров для каждого элемента данных (numpy.ndarray),
        обученная модель KMeans (если модель не загружена),
            или предсказанные метки кластеров (numpy.ndarray) для входных данных (если модель загружена).
    """
    if loaded_model == None and params!=None:
        model_cluster_agg = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params['linkage'])
        labels = model_cluster_agg.fit_predict(data)
        return labels, model_cluster_agg
    loaded_model = pickle.loads(loaded_model)
    loaded_model_cluster_agg = loaded_model.fit_predict(data)
    return loaded_model_cluster_agg

def data_specclust_clust(
        data: np.ndarray,
        params: Optional[dict] = None,
        loaded_model: Optional[bytes] = None
) -> Any:
    """Выполняет кластеризацию данных с использованием спектральной кластеризации (SpectralClustering).

    Args:
      data: np.ndarray. Массив данных для кластеризации.
      params: Optional[dict]. Словарь с параметрами для KMeans, такими как 'n_clusters', 'init', 'max_iter'.  Необходим, если модель не загружена.
      loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для кластеризации.

    Returns:
      Any. Метки кластеров для каждого элемента данных (numpy.ndarray), обученная модель (если модель не загружена), или предсказанные метки кластеров (numpy.ndarray) для входных данных (если модель загружена).
    """
    if loaded_model == None and params!=None:
        model_cluster_spectral = SpectralClustering(n_clusters=params['n_clusters'], affinity=params['affinity'], gamma=params['gamma'],random_state=42)
        labels = model_cluster_spectral.fit_predict(data)
        return labels, model_cluster_spectral
    loaded_model = pickle.loads(loaded_model)
    loaded_model_cluster_spectral = loaded_model.fit_predict(data)
    return loaded_model_cluster_spectral

def data_dbscan_cluster(
        data: np.ndarray,
        params: Optional[dict] = None,
        loaded_model: Optional[bytes] = None
) -> Any:
    """Выполняет кластеризацию данных с использованием DBSCAN.

    Args:
      data: np.ndarray. Массив данных для кластеризации.
      params: Optional[dict]. Словарь с параметрами для KMeans, такими как 'n_clusters', 'init', 'max_iter'.  Необходим, если модель не загружена.
      loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для кластеризации.

    Returns:
      Any. Метки кластеров для каждого элемента данных (numpy.ndarray), обученная модель (если модель не загружена), или предсказанные метки кластеров (numpy.ndarray) для входных данных (если модель загружена).
    """
    if loaded_model == None and params!=None:
        model_cluster_dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels = model_cluster_dbscan.fit_predict(data)
        return labels, model_cluster_dbscan
    loaded_model = pickle.loads(loaded_model)
    loaded_model_cluster_dbscan = loaded_model.fit_predict(data)
    return loaded_model_cluster_dbscan

def data_affprop_cluster(
        data: np.ndarray,
        params: Optional[dict] = None,
        loaded_model: Optional[bytes] = None
) -> Any:
    """Выполняет кластеризацию данных с использованием Affinity Propagation.

    Args:
      data: np.ndarray. Массив данных для кластеризации.
      params: Optional[dict]. Словарь с параметрами для KMeans, такими как 'n_clusters', 'init', 'max_iter'.  Необходим, если модель не загружена.
      loaded_model: Optional[bytes]. Загруженная модель в виде байтовой строки (результат pickle.dumps).  Если указана, используются загруженные параметры для кластеризации.

    Returns:
      Any. Метки кластеров для каждого элемента данных (numpy.ndarray), обученная модель (если модель не загружена), или предсказанные метки кластеров (numpy.ndarray) для входных данных (если модель загружена).
    """
    if loaded_model == None and params!=None:
        model_cluster_affinity = AffinityPropagation(damping=params['damping'], preference=params['preference'], random_state=42)
        labels = model_cluster_affinity.fit_predict(data)
        return labels, model_cluster_affinity
    loaded_model = pickle.loads(loaded_model)
    loaded_model_cluster_affinity = loaded_model.fit_predict(data)
    return loaded_model_cluster_affinity
