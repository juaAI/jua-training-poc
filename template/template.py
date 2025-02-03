"""Template module (for Lambda function)"""
import json


def handler(event: dict, _: dict) -> dict:
    """Lambda function handler

    Args:
        event (dict): A lambda event
        context (dict): A lambda context

    Returns:
        dict: Returns a response from the lambda
    """
    ### Simple pass through AWS Lambda function
    print(f"request: {json.dumps(event)}")

    return {
        "jua": "is awesome!",
        "statusCode": 200,
        "headers": {"Content-Type": "text/plain"},
        "body": "This is a pass through lambda for a stepfunction.",
    }
