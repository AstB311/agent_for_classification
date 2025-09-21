import requests
from typing import Optional, Dict


class BakeSystemAPI:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url

    def _send_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None):
        url = f"{self.base_url}{endpoint}"
        print(f"Отправка {method} запроса на {url} с данными: {json_data}")
        try:
            response = requests.request(method=method, url=url, json=json_data, timeout=10)
            response.raise_for_status()
            result = response.json()
            print(f"Ответ сервера: {result}")
            return result
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса: {e}")
            return None

    def learn_and_predict(self, data: Dict):
        return self._send_request("POST", "/task/train_and_prediction", json_data=data)

    def learn(self, data: Dict):
        return self._send_request("GET", "/task/train", json_data=data)

    def delete(self, data: Dict):
        return self._send_request("DELETE", "/task/delete", json_data=data)

    def predict(self, data: Dict):
        return self._send_request("POST", "/task/prediction", json_data=data)


if __name__ == "__main__":
    api = BakeSystemAPI()

    data_learn_predict = {
        "server": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "5552225",
        "name_database_data": "bake_data",
        "name_database_agent": "bake_agent",
        "name_table_for_learn": "bake_cooling_system_learn",
        "name_table_for_predict": "bake_cooling_system",
        "label_limit": "Все",
        "str_limit": "1:21",
        "task_manager": "LEARN AND PREDICT"
    }

    result = api.learn_and_predict(data_learn_predict)
    if result:
        print("Результат LEARN AND PREDICT:", result)

    # Остальные запросы можно раскомментировать и использовать по аналогии
