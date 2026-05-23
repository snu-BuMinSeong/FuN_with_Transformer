# GCP Access Runbook

Use this file before connecting to the GCP VM. On this Windows machine, direct
SSH aliases can fail because the local Windows username contains non-ASCII
characters. Prefer `gcloud compute ssh` and `gcloud compute scp`.

## Current GCP Target

- Project: `project-ed2a3aec-0315-4f0f-95a`
- Instance: `instance-20260502-111554`
- Zone: `asia-east2-c`
- Remote user: `fumin0193`
- Remote repo: `/home/fumin0193/FuN_with_Transformer/fun-minigrid`
- Python: `/home/fumin0193/.venv/bin/python`

Cloud SDK path on this machine:

```powershell
$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd
```

For PowerShell commands, use:

```powershell
$GCLOUD = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
```

## Fast Connection Checklist

1. Check the active instance.

```powershell
& $GCLOUD compute instances list --project project-ed2a3aec-0315-4f0f-95a
```

2. If SSH aliases are missing or stale, refresh them.

```powershell
& $GCLOUD compute config-ssh --project project-ed2a3aec-0315-4f0f-95a
```

3. Prefer this SSH form.

```powershell
& $GCLOUD compute ssh fumin0193@instance-20260502-111554 --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a
```

4. Run a remote one-shot command.

```powershell
& $GCLOUD compute ssh fumin0193@instance-20260502-111554 --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a --command "cd /home/fumin0193/FuN_with_Transformer/fun-minigrid && pwd && tmux ls || true"
```

## Health Checks

```powershell
& $GCLOUD compute ssh fumin0193@instance-20260502-111554 --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a --command "hostname; df -h /; nvidia-smi || true"
```

```powershell
& $GCLOUD compute ssh fumin0193@instance-20260502-111554 --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a --command "cd /home/fumin0193/FuN_with_Transformer/fun-minigrid && /home/fumin0193/.venv/bin/python --version"
```

## Copy Results Back

From the local repo root, copy an experiment log directory:

```powershell
& $GCLOUD compute scp --recurse fumin0193@instance-20260502-111554:/home/fumin0193/FuN_with_Transformer/fun-minigrid/logs/<remote_log_dir> .\fun-minigrid\logs\ --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a
```

Copy checkpoint directory:

```powershell
& $GCLOUD compute scp --recurse fumin0193@instance-20260502-111554:/home/fumin0193/FuN_with_Transformer/fun-minigrid/checkpoints/<remote_checkpoint_dir> .\fun-minigrid\checkpoints\ --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a
```

Example: MultiRoom N4-S4 seed 1.

```powershell
& $GCLOUD compute scp --recurse fumin0193@instance-20260502-111554:/home/fumin0193/FuN_with_Transformer/fun-minigrid/logs/multiroom_n4s4_seed1 .\fun-minigrid\logs\ --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a
& $GCLOUD compute scp --recurse fumin0193@instance-20260502-111554:/home/fumin0193/FuN_with_Transformer/fun-minigrid/checkpoints/multiroom_n4s4_seed1 .\fun-minigrid\checkpoints\ --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a
```

## List Remote Experiment Files

```powershell
& $GCLOUD compute ssh fumin0193@instance-20260502-111554 --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a --command "cd /home/fumin0193/FuN_with_Transformer/fun-minigrid && find logs/<remote_log_dir> checkpoints/<remote_checkpoint_dir> -maxdepth 3 -type f 2>/dev/null | sort | sed -n '1,160p'"
```

Example:

```powershell
& $GCLOUD compute ssh fumin0193@instance-20260502-111554 --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a --command "cd /home/fumin0193/FuN_with_Transformer/fun-minigrid && find logs/multiroom_n4s4_seed1 checkpoints/multiroom_n4s4_seed1 -maxdepth 3 -type f 2>/dev/null | sort | sed -n '1,160p'"
```

## Common Pitfall

Avoid relying on a raw SSH alias like:

```powershell
ssh instance-20260502-111554.asia-east2-c.project-ed2a3aec-0315-4f0f-95a
```

On this machine it may try to connect as the local Windows username and fail
with `Permission denied (publickey)`. Use:

```powershell
& $GCLOUD compute ssh fumin0193@instance-20260502-111554 --zone asia-east2-c --project project-ed2a3aec-0315-4f0f-95a
```

## Useful Remote Commands

```bash
cd /home/fumin0193/FuN_with_Transformer/fun-minigrid
tmux ls
df -h /
nvidia-smi
tail -n 40 logs/<experiment>/<model>/run_stdout.log
tail -n 5 logs/<experiment>/<model>/eval.csv
```
