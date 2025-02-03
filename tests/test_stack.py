"""Tests for the CDK stack"""
import aws_cdk as core
from aws_cdk import assertions

from stack.template_stack import TemplateStack


def test_lambda_handler_created():
    """Test that the Lambda handler was created"""
    app = core.App()
    stack = TemplateStack(app, "template")
    template = assertions.Template.from_stack(stack)

    template.has_resource_properties(
        "AWS::Lambda::Function", {"Handler": "template.handler"}
    )


def test_lambda_created():
    """Test that the Lambda was created"""
    app = core.App()
    stack = TemplateStack(app, "template")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::Lambda::Function", 1)
