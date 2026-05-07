#!/bin/bash
# Deploy alloul-agent to Hostinger server
# Usage: bash deploy.sh

SERVER="root@srv1431166.hstgr.cloud"
REMOTE_DIR="/root/alloul-agent"

echo "==> Deploying alloul-agent to $SERVER"

# 1. Copy files
rsync -avz --exclude='.env' --exclude='venv/' --exclude='__pycache__/' \
  ./ "$SERVER:$REMOTE_DIR/"

# 2. Setup on server
ssh "$SERVER" << 'EOF'
  cd /root/alloul-agent

  # Create venv if needed
  [ ! -d venv ] && python3 -m venv venv

  # Install deps
  venv/bin/pip install --quiet -r requirements.txt

  # Copy .env if not exists
  [ ! -f .env ] && cp .env.example .env && echo "⚠️  Edit /root/alloul-agent/.env and add GROQ_API_KEY"

  # Install systemd service
  cp alloul-agent.service /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable alloul-agent
  systemctl restart alloul-agent

  echo "==> Status:"
  systemctl status alloul-agent --no-pager
EOF

echo "==> Done! Agent running at http://srv1431166.hstgr.cloud:8002"
echo "==> Health check: curl http://srv1431166.hstgr.cloud:8002/health"
