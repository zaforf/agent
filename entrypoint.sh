#!/bin/bash
set -e

git config --global credential.helper '!f() { echo username=x-access-token; printf "password=%s\n" "$GITHUB_TOKEN"; }; f'

exec gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
