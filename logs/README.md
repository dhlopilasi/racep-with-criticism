# Logs

Store stdout/stderr from long-running experiments here.

Recommended command:

```bash
python experiments/run_racep_sine.py > logs/racep_sine_run001.log 2>&1
```

For background runs, use tmux or nohup. Do not use the agent session as a process supervisor.

Naming convention:

`<method>_<dataset>_<run_id>.log`
