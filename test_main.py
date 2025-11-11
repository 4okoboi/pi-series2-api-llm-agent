from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_infer_positive_en():
    response = client.post("/infer",
                           json={"text": "Hi! This film was awesome!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_infer_negative_en():
    response = client.post("/infer",
                           json={"text": "U stinky asf!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'


def test_infer_positive_ru():
    response = client.post("/infer",
                           json={"text": "Привет! Этот фильм мне очень понравился, спасибо!", "language": "ru"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_infer_negative_ru():
    response = client.post("/infer",
                           json={"text": "Фу, капец ты воняешь!", "language": "ru"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'


def test_infer_unknown_language():
    response = client.post("/infer",
                           json={"text": "Cette langue n'est pas prise en charge", "language": "fr"})
    json_data = response.json()
    assert response.status_code == 400
    assert json_data['detail'] == 'Unsupported language'