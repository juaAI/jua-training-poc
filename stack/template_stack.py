"""Main CDK stack"""
from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from constructs import Construct


class TemplateStack(Stack):
    """The template stack"""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """init the template stack"""
        super().__init__(scope, construct_id, **kwargs)

        # Define the Lambda function
        a_lambda = lambda_.Function(
            self,
            f"{self.stack_name}-lambda",
            runtime=lambda_.Runtime.PYTHON_3_9,
            code=lambda_.Code.from_asset("template"),
            handler="template.handler",
        )

        # Add a URL, allow anonymous invocations
        a_lambda_url = a_lambda.add_function_url(
            auth_type=lambda_.FunctionUrlAuthType.NONE
        )

        # Allow Lambda to list buckets in S3
        s3_list_bucket_policy = iam.Policy(
            self,
            f"{self.stack_name}-lambda-policy",
            statements=[
                iam.PolicyStatement(
                    actions=["s3:ListAllMyBuckets"], resources=["arn:aws:s3:::*"]
                )
            ],
        )
        a_lambda.role.attach_inline_policy(s3_list_bucket_policy)

        # Allow Lambda to read from playground bucket
        playground_bucket = s3.Bucket.from_bucket_name(
            self,
            f"{self.stack_name}-playground-imported-bucket",
            "jua-playground",
        )
        playground_bucket.grant_read(a_lambda)

        CfnOutput(self, f"{self.stack_name}-lambda-url", value=a_lambda_url.url)
