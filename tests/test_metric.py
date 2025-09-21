import pytest
from unittest.mock import patch, Mock
from src.analysis.BakeSystemAPI import BakeSystemAPI


@patch("src.analysis.BakeSystemAPI.requests.request")
def test_learn_and_predict(mock_request):
    # Настраиваем мок-ответ
    mock_response = Mock()
    mock_response.json.return_value = {"status": "success"}
    mock_response.raise_for_status.return_value = None
    mock_request.return_value = mock_response

    api = BakeSystemAPI()
    data = {"some": "data"}
    result = api.learn_and_predict(data)

    # Проверяем, что запрос был сделан
    mock_request.assert_called_once_with(method="POST", url="http://127.0.0.1:8000/task/train_and_prediction", json=data, timeout=10)
    assert result == {"status": "success"}


@patch("src.analysis.BakeSystemAPI.requests.request")
def test_predict(mock_request):
    mock_response = Mock()
    mock_response.json.return_value = {"prediction": 42}
    mock_response.raise_for_status.return_value = None
    mock_request.return_value = mock_response

    api = BakeSystemAPI()
    data = {"some": "data"}
    result = api.predict(data)

    mock_request.assert_called_once_with(method="POST", url="http://127.0.0.1:8000/task/prediction", json=data, timeout=10)
    assert result == {"prediction": 42}


@patch("src.analysis.BakeSystemAPI.requests.request")
def test_delete(mock_request):
    mock_response = Mock()
    mock_response.json.return_value = {"deleted": True}
    mock_response.raise_for_status.return_value = None
    mock_request.return_value = mock_response

    api = BakeSystemAPI()
    data = {"some": "data"}
    result = api.delete(data)

    mock_request.assert_called_once_with(method="DELETE", url="http://127.0.0.1:8000/task/delete", json=data, timeout=10)
    assert result == {"deleted": True}


@patch("src.analysis.BakeSystemAPI.requests.request")
def test_learn(mock_request):
    mock_response = Mock()
    mock_response.json.return_value = {"trained": True}
    mock_response.raise_for_status.return_value = None
    mock_request.return_value = mock_response

    api = BakeSystemAPI()
    data = {"some": "data"}
    result = api.learn(data)

    mock_request.assert_called_once_with(method="GET", url="http://127.0.0.1:8000/task/train", json=data, timeout=10)
    assert result == {"trained": True}
