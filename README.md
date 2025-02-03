# Template :construction:

This is a template for Python and AWS projects at Jua.

## Setup
Install python using `homebrew`:
```bash
brew install python3
```

Install [poetry](https://python-poetry.org/docs/):
```bash
curl -sSL https://install.python-poetry.org | python3 -
# Add poetry to your profile
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc 
```
If you encounter issues with SSL certificates, try to follow the steps described [here](https://stackoverflow.com/a/73270162).

Install initial dependencies:
```bash
poetry install
```

Add the `poetry-exec-plugin` plugin:
```bash
poetry self add poetry-exec-plugin
```

Setup `pre-commit`:
```bash
poetry run pre-commit install 
poetry run pre-commit install --hook-type commit-msg
```

## Manage Python dependencies

Add dependencies:
```bash
# For C based libraries you will often have to install
# from brew, for macOS M1 architectures
# brew install gdal
poetry add rasterio
```

Add development dependencies:
```bash
poetry add pytest-cov --group test
```

## Lint and format your code

Run the format and linter using the predefined [exec](https://pypi.org/project/poetry-exec-plugin/):
```bash
poetry exec format
```

## Run tests

Run tests:
```bash
poetry exec test
```

## Use AWS CDK to manage your AWS resources

### Setup

1. Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```
2. Install the [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html) (requires `node` to be installed).
```bash
brew install node@18.0.0
npm install -g aws-cdk
```
3. Authenticate into AWS (create your Access key ID and Secret access key):
```bash
aws configure
```

### Interact with the CDK

In case you need to deploy AWS resources you can make use of the AWS stack template included in `stack` folder. There, specifically in `template_stack.py` you can define any [AWS CDK](https://docs.aws.amazon.com/cdk/api/v2/) constructs, like [Lambda](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_lambda-readme.html), [S3](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_s3-readme.html), etc.

You can run multiple AWS CDK commands like (make sure to run `export STAGE=dev`, to set your deployment environment to `dev`):
 * `poetry run cdk ls`          list all stacks in the app
 * `poetry run run cdk synth`       emits the synthesized CloudFormation template
 * `poetry run run cdk deploy`      deploy this stack to your default AWS account/region
 * `poetry run run cdk diff`        compare deployed stack with current state
 * `poetry run run cdk docs`        open cdk documentation


## Deploy using GitHub actions

There is a workflow already defined in `.github/workflows/test_and_deploy.yaml` that is configured to automatically deploy your stack, to multiple stages. Specifically:
1. Using the [workflow dispatch](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow), you can deploy to a branch environment manually. This is a specific playground for you branche's code. 
2. Merging a PR to `main` triggers a deployment to a `stage` environment. This is an integration test environment.
3. Pushing a git tag along with your commit, using `git tag 0.1.0 && git push --tags`, will trigger a deployment to `prod`. The production environment is a stable, working environment, that could potentially interact with the outside world or other services.

Enjoy!
