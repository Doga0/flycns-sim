FROM ghcr.io/astral-sh/uv:0.8.22-python3.12-bookworm AS builder
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project
COPY src ./src
RUN uv sync --locked --no-editable

FROM python:3.12.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libgl1 libglfw3 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:$PATH" \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tests ./tests
RUN mkdir -p data outputs
CMD ["python", "-m", "malecns_sim.simulation.smoke_test", "--headless"]
