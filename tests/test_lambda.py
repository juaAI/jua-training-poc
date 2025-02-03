"""Tests for the lambda code"""
from template.template import handler


def test_handler():
    """Test the lambda handler"""
    response = handler({"this is": "test"}, {})
    assert response["jua"] == "is awesome!"
