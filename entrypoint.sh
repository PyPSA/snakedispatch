#!/bin/bash
set -e

# Fix /data ownership for bind mounts created by Docker as root
mkdir -p /data
chown -R appuser:appuser /data

# Install git + pixi on demand for the local backend
if grep -q "^local:" /app/config.yaml 2>/dev/null; then
    if ! command -v git &>/dev/null || ! command -v pixi &>/dev/null; then
        echo "Local backend detected, installing dependencies..."
        apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates >/dev/null
        if ! command -v pixi &>/dev/null; then
            gosu appuser bash -c 'curl -fsSL https://pixi.sh/install.sh | bash'
            export PATH="/home/appuser/.pixi/bin:$PATH"
        fi
    fi
fi

# Use tini as PID 1 to clean up zombie processes (e.g. docker health checks)
exec tini -- gosu appuser "$@"
