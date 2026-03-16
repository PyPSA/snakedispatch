#!/bin/bash
set -e

# Install git + pixi on demand for the local backend
if grep -q "^local:" /app/config.yaml 2>/dev/null; then
    if ! command -v git &>/dev/null || ! command -v pixi &>/dev/null; then
        echo "Local backend detected, installing dependencies..."
        sudo apt-get update -qq && sudo apt-get install -y -qq --no-install-recommends git curl ca-certificates >/dev/null
        if ! command -v pixi &>/dev/null; then
            curl -fsSL https://pixi.sh/install.sh | bash
            export PATH="$HOME/.pixi/bin:$PATH"
        fi
    fi
fi

exec "$@"
