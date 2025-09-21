import pytest
import asyncpg
from unittest.mock import AsyncMock, patch

from src.analysis.connector import DatabaseConnector


@pytest.mark.asyncio
@patch("src.analysis.connector.asyncpg.connect", new_callable=AsyncMock)
async def test_connect_success(mock_connect):
    """
    Тест проверяет успешное подключение к базе данных.
    """
    mock_conn = AsyncMock()
    mock_connect.return_value = mock_conn
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    conn = await connector.connect()
    assert conn == mock_conn
    mock_connect.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.analysis.connector.asyncpg.connect", new_callable=AsyncMock)
async def test_connect_failure(mock_connect):
    """
    Тест проверяет обработку ошибки при подключении к базе данных.
    """
    mock_connect.side_effect = asyncpg.PostgresError("Connection failed")
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    with pytest.raises(Exception):
        await connector.connect()


@pytest.mark.asyncio
async def test_check_table_exists():
    """
    Тест проверяет, существует ли таблица в базе данных.
    """
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = True
    connector.conn = mock_conn
    result = await connector.check_table_exists("test_table")
    assert result is True
    mock_conn.fetchval.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_exists_in_table():
    """
    Тест проверяет, существует ли запись в таблице.
    """
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = True
    connector.conn = mock_conn
    result = await connector.check_exists_in_table("some_table", "machineX")
    assert result is True
    mock_conn.fetchval.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_model_table():
    """
    Тест проверяет создание таблицы для хранения моделей.
    """
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    connector.conn = mock_conn
    await connector.create_model_table("models", "model_name")
    assert mock_conn.execute.await_count == 1


@pytest.mark.asyncio
async def test_insert_data():
    """
    Тест проверяет вставку данных в таблицу.
    """
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    connector.conn = mock_conn
    test_model = object()
    test_data = {
        "machine": "machine1",
        "model": test_model,
        "method_param": "param",
        "accuracy": 0.95
    }
    await connector.insert_data("models", test_data)
    assert mock_conn.execute.await_count == 1


@pytest.mark.asyncio
async def test_get_data_table():
    """
    Тест проверяет получение данных из таблицы.
    """
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [
        {"id": 1, "machine": "A"},
        {"id": 2, "machine": "B"}
    ]
    connector.conn = mock_conn

    result = await connector.get_data_table("models")
    assert isinstance(result, list)
    assert result[0]["machine"] == "A"


@pytest.mark.asyncio
async def test_delete_table_agent():
    """
    Тест проверяет удаление таблицы агента.
    """
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    connector.conn = mock_conn
    await connector.delete_table_agent("models")
    mock_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_close():
    """
    Тест проверяет закрытие соединения с базой данных.
    """
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    connector.conn = mock_conn
    await connector.close()
    mock_conn.close.assert_awaited_once()
