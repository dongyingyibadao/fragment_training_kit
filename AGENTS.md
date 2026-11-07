# Working in this project

Start with `AGENT_START_HERE.md`, then follow the linked architecture, model,
platform and validation documents. This is a standalone project; do not import
or copy files from its original experiment directory to make an example work.

- Use `vendor/` for the resident runtime. It is an intentional source snapshot
  with hashes in `runtime_manifest.json`, not a missing dependency to install.
- Add a model through an explicit `module:factory` adapter and `source_dirs`.
  Keep the fixed batch/optimizer and recovery contracts until separately tested.
- Keep authentication, real inventory, site settings, runs, data and large
  checkpoints outside the repository. Never print credential contents.
- `prepare`, `verify`, CPU oracle and tests allocate no platform resources.
  `run` creates jobs; honor the user's resource authorization and shared budget.
- Before experiments, inspect local device users and keepalive monitoring. Use
  the site's task-owned pause-file mechanism if necessary; preserve monitoring,
  remove only your own pause file in cleanup, and never kill unrelated jobs.
- Preserve failed-run evidence, original historical records and earliest errors.
  Stop only jobs owned by the current run and confirm official terminal states.
- Run CPU checks with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. Use an existing CPU
  environment or an isolated environment; do not upgrade shared NPU environments.
- Follow the verification and release-hash procedure in
  `docs/07_standalone_project.md` after changing this project.

Optional independent investigations may run in parallel with bounded ownership.
Do not have multiple agents edit the same file concurrently. The integrating
agent owns verification and the final result; subagents must not spawn agents.
