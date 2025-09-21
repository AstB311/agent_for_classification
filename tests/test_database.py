import pytest
from unittest.mock import AsyncMock, patch
from src.analysis.connector import DatabaseConnector


@pytest.mark.asyncio
@patch("src.analysis.connector.asyncpg.connect", new_callable=AsyncMock)
async def test_connect_success(mock_connect):
    mock_conn = AsyncMock()
    mock_connect.return_value = mock_conn

    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    conn = await connector.connect()

    assert conn == mock_conn
    mock_connect.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.analysis.connector.asyncpg.connect", new_callable=AsyncMock)
async def test_connect_failure(mock_connect):
    mock_connect.side_effect = Exception("Connection failed")

    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    with pytest.raises(Exception):
        await connector.connect()


@pytest.mark.asyncio
async def test_check_table_exists():
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = True
    connector.conn = mock_conn

    result = await connector.check_table_exists("test_table")
    assert result is True
    mock_conn.fetchval.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_exists_in_table():
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = True
    connector.conn = mock_conn

    result = await connector.check_exists_in_table("some_table", "machineX")
    assert result is True
    mock_conn.fetchval.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_model_table():
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    connector.conn = mock_conn

    await connector.create_model_table("models", "model_name")
    mock_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_data():
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    connector.conn = mock_conn

    test_data = {
        "machine": "machine1",
        "model": object(),
        "method_param": "param",
        "accuracy": 0.95
    }

    await connector.insert_data("models", test_data)
    mock_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_data_table():
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
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    connector.conn = mock_conn

    await connector.delete_table_agent("models")
    mock_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_close():
    connector = DatabaseConnector("localhost", 5432, "db", "user", "pass")
    mock_conn = AsyncMock()
    connector.conn = mock_conn

    await connector.close()
    mock_conn.close.assert_awaited_once()
