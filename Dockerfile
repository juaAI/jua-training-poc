FROM --platform=linux/amd64 python:3.11-slim

# Python setup
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1 \
    # pip:
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # poetry:
    POETRY_VERSION=1.8.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PATH="$PATH:/root/.local/bin"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry==$POETRY_VERSION && \
    poetry --no-cache self add poetry-plugin-bundle poetry-plugin-export && \
    poetry cache clear --no-interaction --all .

COPY . .

# Prepare base venv
RUN ./scripts/prepare-base-venv.sh /venv

# Install dependencies
RUN poetry --no-cache bundle venv --with=profiling --with=validation /venv && poetry cache clear --no-interaction --all .

ENV PATH=/venv/bin:${PATH}

ENTRYPOINT []
CMD []