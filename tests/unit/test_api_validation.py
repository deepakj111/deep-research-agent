from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Patch graph to avoid initializing real graph for these simple tests
    with patch("api.main.graph"):
        from api.main import app

        yield TestClient(app)


def test_query_max_length_validation(client):
    """Verify that queries exceeding the max length are rejected with 422."""
    long_query = "a" * 1501
    response = client.post("/research/stream", json={"query": long_query})
    assert response.status_code == 422
    assert "query" in response.text
    assert "at most 1500 characters" in response.text


def test_query_min_length_validation(client):
    """Verify that empty queries are rejected with 422."""
    response = client.post("/research/stream", json={"query": ""})
    assert response.status_code == 422


def test_rate_limiting(client):
    """
    Verify that rate limiting triggers after burst.
    Note: slowapi by default uses get_remote_address. In tests, this might be 127.0.0.1.
    """
    # 5 requests per minute is the limit
    for _ in range(5):
        response = client.post("/research/stream", json={"query": "test"})
        # We don't care about success here, just that it's NOT 429
        assert response.status_code != 429

    # 6th request should fail
    response = client.post("/research/stream", json={"query": "test"})
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.text
