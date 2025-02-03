"""A CDK template app"""
import os

import aws_cdk as cdk

from stack.template_stack import TemplateStack

STAGE = os.getenv("STAGE")

app = cdk.App()
stack = TemplateStack(app, f"template-{STAGE}")

cdk.Tags.of(stack).add("project", "template")
cdk.Tags.of(stack).add("stage", STAGE)

app.synth()
