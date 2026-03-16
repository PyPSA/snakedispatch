# snakedispatch

Dispatches Snakemake workflows to compute backends.

## Architecture

- **Single-tenant service**: all API routes are trusted; auth is enforced at the network layer.
- **Compute backends** (local, slurm_ssh) are abstracted behind `ComputeBackend` (`app/backends/base.py`). The backend is selected at startup from `config.yaml` and injected via FastAPI DI.
- **Job lifecycle**: `PENDING` -> `SETUP` -> `RUNNING` -> `COMPLETED`/`FAILED`/`CANCELLED`. Orchestrated by `execute_job` in `app/tasks.py`; state persisted via `app/store.py`.
- **snkmt DB sync**: Snakemake job-graph metadata (rule counts, file listings) is synced from the compute backend to a local SQLite file during execution and exposed via `/snkmt/*` endpoints. See `app/snkmt.py` for the query layer.
