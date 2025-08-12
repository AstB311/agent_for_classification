# Импортирование библиотеки для HTTP запросов
import requests

#Запрос LEARN AND PREDICT
data = {
    "server": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "5552225",
    "name_database_data": "bake_data", #название БД с которой берем данные (с метками или без, зависит от задачи)
    "name_database_agent": "bake_agent", #название БД для агента
    "name_table_for_learn": "conveyor_learn",   #Название оборудования для которого берем данные
    "name_table_for_predict": "conveyor",       #Название оборудования для которого берем данные
    "label_limit": "Все",
    "str_limit": "1:21",
    "task_manager": "LEARN AND PREDICT"
}
response = requests.post("http://127.0.0.1:8000/task/train_and_prediction", json=data)
print("Данные, полученные после запроса: ", response.json())
"""
#Запрос LEARN
data = {
    "server": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "5552225",
    "name_database_data": "bake_data", #название БД с которой берем данные (с метками или без, зависит от задачи)
    "name_database_agent": "bake_agent", #название БД для агента
    "name_table_for_learn": "bake_cooling_system_learn",   #Название оборудования для которого берем данные
    "task_manager": "LEARN"
}
response = requests.get("http://127.0.0.1:8000/task/train", json=data)
print("Данные, полученные после запроса: ", response.json())

#Запрос DELETE
data = {
    "server": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "5552225",
    "name_database_agent": "bake_agent", #название БД для агента
    "task_manager": "DELETE"
}
response = requests.delete("http://127.0.0.1:8000/task/delete", json=data)
print("Данные, полученные после запроса: ", response.json())

#Запрос PREDICTION
data = {
    "server": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "5552225",
    "name_database_data": "bake_data", #название БД с которой берем данные (с метками или без, зависит от задачи)
    "name_database_agent": "bake_agent", #название БД для агента
    "name_table_for_learn": "bake_cooling_system",   #Название оборудования для которого берем данные = название таблицы, с которой берем данные (с метками или нет)
    "label_limit": "Норма",
    "str_limit": "1:20",
    "task_manager": "PREDICT"
}
response = requests.post("http://127.0.0.1:8000/task/prediction", json=data)"""
#print("Данные, полученные после запроса: ", response.json())
