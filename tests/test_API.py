import pytest
import requests

BASE_URL = "http://127.0.0.1:8000/task"

@pytest.fixture
def base_data():
    """
    Фикстура Pytest для предоставления базовых данных подключения к базе данных.
    """
    return {
        "server": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "5552225",
        "name_database_agent": "bake_agent",  # название БД для агента
    }

def test_train_and_prediction(base_data):
    """
    Тест для проверки эндпоинта train_and_prediction.
    Проверяет успешный статус код (200) и наличие ключа "data" в ответе.
    """
    data = base_data.copy()
    data.update({
        "name_database_data": "bake_data",  # название БД с которой берем данные
        "name_table_for_learn": "bake_cooling_system_learn",  # Название оборудования для которого берем данные
        "name_table_for_predict": "bake_cooling_system",  # Название оборудования для которого берем данные
        "label_limit": "Все",
        "str_limit": "1:21",
        "task_manager": "LEARN AND PREDICT"
    })
    response = requests.post(f"{BASE_URL}/train_and_prediction", json=data)
    assert response.status_code == 200
    assert "data" in response.json()


def test_train(base_data):
    """
    Тест для проверки эндпоинта train.
    Проверяет успешный статус код (200) и наличие ключа "data" в ответе.
    """
    data = base_data.copy()
    data.update({
        "name_database_data": "bake_data",  # название БД с которой берем данные
        "name_table_for_learn": "bake_cooling_system_learn",  # Название оборудования для которого берем данные
        "task_manager": "LEARN"
    })
    response = requests.get(f"{BASE_URL}/train", json=data)
    assert response.status_code == 200
    assert "data" in response.json()


def test_prediction(base_data):
    """
    Тест для проверки эндпоинта prediction.
    Проверяет успешный статус код (200) и наличие ключа "data" в ответе.
    """
    data = base_data.copy()
    data.update({
        "name_database_data": "bake_data",  # название БД с которой берем данные
        "name_table_for_learn": "bake_cooling_system",  # Название оборудования для которого берем данные
        "label_limit": "Норма",
        "str_limit": "1:20",
        "task_manager": "PREDICT"
    })
    response = requests.post(f"{BASE_URL}/prediction", json=data)
    assert response.status_code == 200
    assert "data" in response.json()


def test_delete(base_data):
    """
    Тест для проверки эндпоинта delete.
    Проверяет успешный статус код (200) и наличие ключа "data" в ответе.
    """
    data = base_data.copy()
    data.update({
        "task_manager": "DELETE"
    })
    response = requests.delete(f"{BASE_URL}/delete", json=data)
    assert response.status_code == 200
    assert "data" in response.json()