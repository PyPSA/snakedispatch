# How To Work In This Repo

This is a practical second-pass guide for making changes safely and quickly.

## What Kind Of Project This Is

This repo is small and test-heavy. There is not much indirection. Most changes fall into one of these buckets:

- API contract change
- job lifecycle change
- backend behavior change
- config/model validation change
- Snakemake metadata query change

Because the codebase is compact, the fastest path is usually:

1. find the route or behavior
2. trace into the store/task/backend layer
3. update tests close to that behavior

## Local Development Expectations

The repository uses Python packaging via `pyproject.toml` and locks dependencies with `uv.lock`.

Relevant tooling from [pyproject.toml](/Users/mikoding/Documents/mikoding/snakedispatch/pyproject.toml:1):

- Python `>=3.12`
- FastAPI
- Pydantic v2
- `pytest`
- `pytest-asyncio`
- `httpx`
- `ruff`

The container uses `uvicorn` to run the app, as shown in [Dockerfile](/Users/mikoding/Documents/mikoding/snakedispatch/Dockerfile:44).

## Recommended First Commands

If you are setting up locally outside Docker, the usual commands are:

```bash
uv sync
pytest
ruff check .
```

If you want to run the service directly:

```bash
uv run uvicorn app.main:app --reload
```

The docs UI will then be available at:

- `/docs`
- `/redoc`

If you want the containerized path, [README.md](/Users/mikoding/Documents/mikoding/snakedispatch/README.md:11) uses:

```bash
cp compose.example.yml compose.yml
docker compose up -d
```

## Configuration Workflow

The app expects one YAML config file at `./config.yaml` by default, controlled by `SNAKEDISPATCH_CONFIG`.

Use one of:

- [config/config.local.example.yaml](/Users/mikoding/Documents/mikoding/snakedispatch/config/config.local.example.yaml:1)
- [config/config.slurm_ssh.example.yaml](/Users/mikoding/Documents/mikoding/snakedispatch/config/config.slurm_ssh.example.yaml:1)

For local work, prefer the `local` backend unless you are specifically changing remote execution behavior.

## How To Trace A Change

### If you are changing an API endpoint

Start here:

- [app/routes/jobs.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/jobs.py:98)
- [app/routes/snkmt.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/snkmt.py:178)
- tests:
  - [tests/test_api.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_api.py:1)
  - [tests/test_snkmt.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_snkmt.py:1)

Typical pattern:

1. update request or response model in [app/models.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/models.py:1) if needed
2. update route logic
3. update tests first or immediately after

### If you are changing job lifecycle or orchestration

Start here:

- [app/tasks.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/tasks.py:180)
- [app/store.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/store.py:31)
- tests:
  - [tests/test_execute.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_execute.py:1)
  - [tests/test_store.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_store.py:1)

This is where status transitions, finalization, cleanup, cache restore/save, and sync behavior live.

### If you are changing execution behavior

Start here:

- [app/backends/base.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/backends/base.py:23)
- [app/backends/local.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/backends/local.py:24)
- [app/backends/slurm_ssh.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/backends/slurm_ssh.py:97)
- tests:
  - [tests/test_base.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_base.py:1)
  - [tests/test_local_backend.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_local_backend.py:1)
  - [tests/test_slurm_ssh.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_slurm_ssh.py:1)

Read `LocalBackend` first. It is simpler and reveals the shape of the interface.

### If you are changing validation or config behavior

Start here:

- [app/config.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/config.py:1)
- [app/models.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/models.py:1)
- tests:
  - [tests/test_models.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_models.py:1)
  - parts of [tests/test_local_backend.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_local_backend.py:1)

### If you are changing Snakemake metadata behavior

Start here:

- [app/snkmt.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/snkmt.py:1)
- [app/routes/snkmt.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/snkmt.py:178)
- tests:
  - [tests/test_snkmt.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_snkmt.py:1)

## Test Strategy

The tests are one of the strongest parts of this repo. Use them as executable documentation.

Suggested workflow:

1. run the narrowest test file first
2. fix failures locally
3. run the broader relevant subset
4. finish with the full suite if your change is cross-cutting

Examples:

```bash
pytest tests/test_api.py
pytest tests/test_execute.py
pytest tests/test_local_backend.py
pytest
```

## How The Tests Are Built

[tests/conftest.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/conftest.py:1) is worth reading early because it shows:

- how `AppState` is injected
- how the backend is mocked
- how the sample `snkmt.db` is created
- how the ASGI app is exercised with `httpx`

That fixture setup tells you what the team considers stable boundaries.

## Common Safe Assumptions

- Route handlers should stay thin.
- Shared runtime state should live in `AppState`.
- Job lifecycle transitions should go through `JobStore`.
- Backend-specific behavior should stay behind `ComputeBackend`.
- New functionality should come with tests near the touched layer.

## Common Footguns

- Do not bypass the store when mutating job state unless there is a strong reason.
- Do not mix backend-specific behavior into routes.
- Be careful with path validation for `configfile`, `extra_files`, `cache_dirs`, and output downloads.
- Remember that logs and `snkmt.db` are intentionally persisted for restart tolerance.
- The service is designed for exactly one configured backend, not many at once.
- `slurm_ssh` has an optional dependency on `asyncssh`; not every environment will have it installed.

## Good First Debugging Path

When something is wrong with a submitted job, inspect in this order:

1. route request validation in [app/routes/jobs.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/jobs.py:98)
2. lifecycle transitions in [app/tasks.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/tasks.py:180)
3. persisted state and logs in [app/store.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/store.py:31)
4. backend launch and monitor logic in the relevant backend file
5. Snakemake metadata sync and query behavior if the issue is in workflow detail endpoints

## If You Need To Add A New Backend

The intended extension path is explicitly documented in [app/backends/base.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/backends/base.py:23).

You would need to:

1. implement a new `ComputeBackend` subclass
2. add a config model in [app/config.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/config.py:1)
3. register the backend key in config loading
4. update [app/backends/__init__.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/backends/__init__.py:1)
5. add backend-specific tests

## A Good First Week Plan

1. Read [app/main.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/main.py:26), [app/routes/jobs.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/routes/jobs.py:98), and [app/tasks.py](/Users/mikoding/Documents/mikoding/snakedispatch/app/tasks.py:180).
2. Run `pytest` and skim failures only if they happen.
3. Read `LocalBackend` end to end.
4. Read [tests/conftest.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/conftest.py:1) and [tests/test_api.py](/Users/mikoding/Documents/mikoding/snakedispatch/tests/test_api.py:1).
5. Pick one small bugfix or test-only change before attempting a backend-level feature.

## Short Summary

The fastest way to work effectively in this repo is to treat it as a compact service with strong boundaries:

- routes define HTTP behavior
- tasks define execution flow
- store defines persisted state
- backends define environment-specific execution
- tests define expected behavior

If you preserve those boundaries, most changes stay simple.
