# Codebase Introduction

This repository is `snakedispatch`, a small FastAPI service that dispatches Snakemake workflows onto one compute backend and exposes job state over HTTP.

At a high level:

1. A client calls `POST /jobs` with a workflow source and optional execution settings.
2. The service creates a job record and starts an async background task.
3. The configured backend prepares the workflow, launches Snakemake, and monitors execution.
4. The service persists job metadata, streams logs, exposes outputs, and syncs Snakemake metadata from `snkmt.db`.

## What The Service Does

- Accepts workflow submissions from a Git URL or a local directory.
- Supports one compute backend per deployment:
  - `local`: runs on the same machine.
  - `slurm_ssh`: connects to a remote SLURM cluster over SSH.
- Stores job state on disk so jobs survive process restarts.
- Streams stdout logs over SSE.
- Exposes output files and Snakemake workflow metadata through REST endpoints.

## Main Runtime Flow

The app is wired in [app/main.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/main.py:26).

- `lifespan()` loads config from YAML plus environment variables.
- It creates:
  - a `JobStore`
  - one `ComputeBackend`
  - shared app state in `request.app.state.app`
- It restores persisted jobs from disk.
- It starts two background loops:
  - `gc_loop()` for cleaning old terminal jobs
  - `sync_job_data_loop()` for flushing logs and syncing `snkmt.db`

## The Files That Matter Most

Read these in order:

1. [README.md](/Users/mikoding/Documents/mikoding/snakedispatch/README.md:1)
2. [app/main.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/main.py:26)
3. [app/routes/jobs.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/jobs.py:98)
4. [app/tasks.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/tasks.py:180)
5. [app/store.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/store.py:31)
6. [app/backends/base.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/backends/base.py:23)
7. [app/backends/local.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/backends/local.py:24)
8. [app/backends/slurm_ssh.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/backends/slurm_ssh.py:97)
9. [app/routes/snkmt.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/snkmt.py:178)
10. [app/snkmt.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/snkmt.py:1)

## Mental Model By Layer

### API Layer

[app/routes/jobs.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/jobs.py:98) is the main API surface.

- `POST /jobs`: validate request, create job, spawn async execution
- `GET /jobs`: list jobs
- `GET /jobs/{job_id}`: get one job
- `GET /jobs/{job_id}/logs`: SSE log stream
- `GET /jobs/{job_id}/outputs`: list output files
- `GET /jobs/{job_id}/outputs/{path}`: download one output
- `POST /jobs/{job_id}/cancel`: cancel a running job
- `DELETE /jobs/{job_id}`: remove the job and trigger cleanup

[app/routes/snkmt.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/snkmt.py:178) exposes read-only workflow data from the Snakemake SQLite database.

### Execution Layer

[app/tasks.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/tasks.py:180) is the real execution pipeline.

The sequence is:

1. `backend.prepare()`
2. `store.mark_setup()`
3. `backend.setup()`
4. optional cache restore
5. `store.mark_running()`
6. `backend.launch()`
7. `backend.monitor()`
8. collect outputs and final status
9. flush logs and sync final Snakemake data

This file is the best place to understand job lifecycle transitions.

### Persistence Layer

[app/store.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/store.py:31) manages in-memory and on-disk job state.

Each job record tracks:

- status
- timestamps
- exit code
- workflow source
- Git ref and SHA
- cached workflow file listing
- log lines
- optional Snakemake progress counters

Persisted data lives under the configured `DATA_DIR`, typically in `jobs/<job_id>/`.

### Backend Layer

[app/backends/base.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/backends/base.py:82) defines the common backend protocol.

Important idea: the base class already handles Git-backed workflow preparation using a bare repo plus worktrees. Backend implementations mainly differ in how they:

- execute commands
- copy local workflows
- read files and logs
- clean up processes and directories
- sync `snkmt.db`

`LocalBackend` is the easiest backend to understand first. `SlurmSSHBackend` adds SSH persistence, SFTP, remote launch, remote status probing, and SLURM-oriented operation.

## Configuration Model

[app/config.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/config.py:17) splits configuration into:

- environment-driven app settings via `Settings`
- YAML-defined backend config via `load_config()`

The YAML file must define exactly one backend key:

- `local`
- `slurm_ssh`

Examples:

- [config/config.local.example.yaml](/Users/mikoding/Documents/mikoding/snakedispatch/config/config.local.example.yaml:1)
- [config/config.slurm_ssh.example.yaml](/Users/mikoding/Documents/mikoding/snakedispatch/config/config.slurm_ssh.example.yaml:1)

## Domain-Specific Piece: `snkmt.db`

The most project-specific part of the service is the Snakemake metadata database.

- The Snakemake wrapper script tries to enable the `snakemake_logger_plugin_snkmt` plugin.
- That plugin writes workflow metadata to `snkmt.db`.
- The service periodically syncs that DB locally and queries it through [app/snkmt.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/snkmt.py:1).

That is how the API can expose:

- workflow counts
- rules
- jobs
- files per job
- rulegraph
- errors

## Operational Behavior To Keep In Mind

- This service is restart-tolerant because job metadata and logs are persisted.
- Log streaming uses SSE while a job is running.
- Output downloads are plain streamed HTTP responses after or during execution.
- Old terminal jobs are garbage-collected by age.
- Environment variables passed into workflows are allowlisted by config.
- Most edge-case behavior is covered by tests rather than large amounts of inline documentation.

## Best Way To Learn The Code

1. Read [tests/conftest.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/conftest.py:1) to see how the app is assembled in tests.
2. Read [tests/test_api.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_api.py:1) to understand expected endpoint behavior.
3. Follow one job from `POST /jobs` through [app/routes/jobs.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/jobs.py:98), [app/tasks.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/tasks.py:180), and the chosen backend.
4. Read `LocalBackend` before `SlurmSSHBackend`.

## Short Summary

This is not a large framework-heavy codebase. It is a compact service with four key concepts:

- HTTP routes
- job execution orchestration
- backend abstraction
- persisted job state plus Snakemake metadata

If you understand those four pieces, you will be productive quickly.
