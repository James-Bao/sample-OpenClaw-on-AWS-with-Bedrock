# Stage 1: Extract openclaw binary from official image
FROM ghcr.io/openclaw/openclaw:latest AS openclaw-base

# Stage 2: Python 3.12 slim runtime
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=openclaw-base /usr/local/bin/openclaw /usr/local/bin/openclaw

WORKDIR /app

# Install Python dependencies
COPY agent-container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Agent Container application files
COPY agent-container/server.py .
COPY agent-container/openclaw.json .
COPY agent-container/agent-card.json .
COPY agent-container/a2a.py .
COPY agent-container/permissions.py .
COPY agent-container/identity.py .
COPY agent-container/memory.py .
COPY agent-container/observability.py .
COPY agent-container/safety.py .

# Copy auth-agent module (needed by permissions.py for PermissionRequest)
RUN mkdir -p /app/auth-agent
COPY auth-agent/permission_request.py /app/auth-agent/permission_request.py
COPY auth-agent/__init__.py /app/auth-agent/__init__.py

# Copy systemd service unit (for non-Docker deployments)
COPY agent-container/openclaw-a2a.service /etc/systemd/system/openclaw-a2a.service

# Create session storage directory
RUN mkdir -p /tmp/openclaw/sessions

ENV OPENCLAW_SKIP_ONBOARDING=1
ENV OPENCLAW_CONFIG=/app/openclaw.json
ENV PORT=8080
ENV A2A_ENABLED=true

EXPOSE 8080

# Build from repo root: docker build -f agent-container/Dockerfile .
CMD ["python", "server.py"]
