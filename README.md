# snakedispatch

Workflow execution backend for [pypsa-app](https://github.com/pypsa/pypsa-app), but can be used independently. Dispatches Snakemake workflows to compute targets via a FastAPI microservice.

Clones workflow repos onto compute targets, launches Snakemake inside [pixi](https://pixi.sh) environments, monitors execution by polling, and streams logs back via SSE.

Supported compute targets:
- SLURM over SSH: connects to an HPC head node via ssh, runs jobs as SLURM submissions
- Local: runs workflows on the same machine

## Quickstart

### Docker

```bash
cp compose.example.yml compose.yml
# Adjust settings and remove unneeded compute targets
docker compose up -d
```

The container image is published at `ghcr.io/pypsa/snakedispatch:latest`.

## Configuration

All configuration lives in a single YAML file (default path: `./config.yaml`).

The file must contain exactly one compute target key (`slurm_ssh` or `local`) with its settings. App-level settings can optionally be added at the top level. See `config/` for example configurations.

## API

With the dev server running, interactive API docs are available at `/docs` (Swagger UI) and `/redoc`.
