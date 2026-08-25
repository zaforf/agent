#!/bin/bash
set -e

git config --global credential.helper '!f() { echo username=x-access-token; printf "password=%s\n" "$GITHUB_TOKEN"; }; f'

# Telegram long-polling waits up to 60s, and a reasoning/tool turn can also
# legitimately exceed Gunicorn's 30s default.  Keep the guard finite so a
# genuinely wedged worker can still be recycled, but leave enough headroom.
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-90}"
exec gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker \
  --timeout "$GUNICORN_TIMEOUT" --bind 0.0.0.0:8000
