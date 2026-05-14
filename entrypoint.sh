#!/bin/bash
set -e

gh auth login --with-token <<< "$GITHUB_TOKEN"
gh auth setup-git

exec gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
