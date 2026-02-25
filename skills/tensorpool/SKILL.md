---
name: tensorpool
description: This skill helps users migrate their local machine learning scripts to run on TensorPool GPU clusters using the interactive cluster workflow (tp ssh). Use this when a user has a working local script and wants to scale it up to professional GPU hardware.
---

# SKILL: Convert Local ML Scripts to TensorPool GPU Production Runs

## Overview

This skill helps users migrate their local machine learning scripts to run on TensorPool GPU clusters using the interactive cluster workflow. Use this when a user has a working local script and wants to scale it up to professional GPU hardware (H100, H200, B200, B300, etc.).

**Trigger phrases:** "run this on H100", "scale to TensorPool", "convert to production", "run on cloud GPU", "migrate to H100", "make this faster on GPU", "run on B200"

**Workflow:** Analyze local script → Discover CLI commands → Prepare for cluster → Create GPU cluster → Transfer code → Run on GPU → Retrieve outputs → Clean up

---

## Error Handling & Iterative Fixing

When running the user's script on TensorPool and it throws errors, **Claude should proactively diagnose and fix the code** without asking for permission, as long as the fix is in service of the script's original training/inference objective.

**Guiding principles:**
- **Fix it, don't just report it.** If a script fails with a traceable error (import errors, CUDA issues, shape mismatches, OOM, path errors, dependency issues), read the traceback, identify the root cause, apply the fix, and re-run.
- **Stay in scope.** Fixes must serve the original purpose of the script. Fixing a broken dataloader or adjusting batch size for OOM is in scope. Rewriting the model architecture or changing the training objective is not — ask the user first.
- **Iterate until it runs.** It's normal for scripts to need 2-5 rounds of fixes when moving from local to cloud. Keep going — fix dependency issues, path problems, CUDA compatibility, dtype mismatches, etc. one by one.
- **Explain what you changed.** After each fix, briefly note what broke and what you did so the user can learn and replicate.

**Common auto-fixable errors:**
- `ModuleNotFoundError` → `pip install` the missing package, re-run
- `CUDA out of memory` → reduce batch size, enable gradient checkpointing, or switch to LoRA
- `FileNotFoundError` → fix hardcoded paths, create missing directories (`mkdir -p`)
- `RuntimeError: expected dtype` → add `.to(device)` or fix dtype mismatches (fp32 vs bf16)
- `NCCL timeout / distributed errors` → fix environment variables, world size, rank config
- `KeyError` in dataset/config → inspect the data format and adjust loading code
- `AttributeError` from API changes → update to current library API (e.g., deprecated HuggingFace args)

**Out of scope (ask the user):**
- Changing the model being trained
- Changing the training objective or loss function
- Significantly altering hyperparameters beyond what's needed for the hardware (e.g., changing learning rate schedule)
- Switching frameworks entirely (e.g., PyTorch → JAX)

---

## CLI Discovery (CRITICAL — Always Do This First)

The TensorPool CLI (`tp`) changes frequently. **Never assume command syntax from this document. Always discover current commands at runtime.**

### Step 0 — Discover Available Commands

Before running any `tp` commands, always run:

```bash
# 1. Check tp is installed
pip show tensorpool || pip install tensorpool

# 2. Discover top-level commands
tp --help

# 3. Drill into relevant subcommands
tp cluster --help
tp ssh --help
tp storage --help
tp me --help
```

**Use the output of `--help` to determine:**
- Exact command names and syntax
- Required vs optional arguments
- Available flags and their current names
- Available instance types

**If a command fails or the syntax has changed**, re-run `--help` on that subcommand to get the current usage.

**General expectations** (these may change — always verify with `--help`):
- There will be commands to: create clusters, list clusters, get cluster info, destroy clusters, SSH into instances, manage storage, and check account info
- Cluster creation will need at minimum an instance type (e.g., `1xH100`, `8xB200`)
- Multi-node clusters may only be supported for certain instance types
- SSH will take some form of instance or cluster identifier
- Cluster destruction will take a cluster identifier

---

## Prerequisites (Verify Before Starting)

1. **TensorPool CLI installed:**
   ```bash
   pip show tensorpool || pip install tensorpool
   ```

2. **Authenticated:**
   ```bash
   # Run the account info command (discover exact syntax via tp --help)
   # If not authenticated, configure your API key from:
   # https://tensorpool.dev/dashboard
   ```

3. **SSH key available** (may be needed for rsync/scp to cluster):
   ```bash
   ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
   ```

4. **User has a TensorPool account** — sign up at [tensorpool.dev](https://tensorpool.dev)

---

## Step-by-Step Migration Workflow

### Step 1 — Analyze the Local Script

Before migrating, understand:

1. **What does the script do?** (training, inference, data processing)
2. **What are the dependencies?** (check imports, requirements.txt)
3. **What data does it need?** (local files, datasets, pretrained models)
4. **What outputs does it produce?** (model checkpoints, logs, results)
5. **GPU requirements:** Single GPU? Multi-GPU? Memory needs?

**Key questions to ask:**
- Is there a `requirements.txt`? If not, create one.
- Are there hardcoded local paths that need adjustment?
- Does it use relative imports that might break?
- Are there large data files that need to be transferred?

### Step 2 — Prepare the Script for Cloud Execution

**2.1 Create/verify requirements.txt:**
```bash
# If it doesn't exist, create it
pip freeze > requirements.txt

# Or manually list dependencies
echo "torch>=2.0.0" > requirements.txt
echo "transformers>=4.30.0" >> requirements.txt
```

**2.2 Check for environment variables:**
```bash
# Create .env file if needed
cat > .env << EOF
HUGGINGFACE_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key_here
EOF
```

**2.3 Test locally first (if possible):**
```bash
# Quick sanity check with minimal data
python your_script.py --max_samples 10 --num_epochs 1
```

**2.4 Identify files to transfer:**
- Scripts (`.py` files)
- Configuration files (`.yaml`, `.json`, `.toml`)
- Requirements (`requirements.txt`)
- Small data files (< 1GB)
- Environment variables (`.env`)

**2.5 Identify files NOT to transfer:**
- Large datasets (download directly on cluster)
- Previous outputs/checkpoints
- Git history (`.git/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)

### Step 3 — Create GPU Cluster

**Choosing an instance type:**

| GPU | Memory | Best For |
|-----|--------|----------|
| H100 | 80GB | General training, fine-tuning 7B-70B models |
| H200 | 141GB | Large context, 70B+ models, memory-bound workloads |
| B200 | 192GB | Latest gen, native FP4/FP6, fastest training |
| B300 | Latest | Newest generation, highest performance |
| L40S | 48GB | Inference, smaller training workloads |

Instance types come in configurations like `1xH100`, `2xH100`, `4xH100`, `8xH100`, etc.

**To create a cluster:**
1. Run `tp cluster --help` to see the exact create syntax and required arguments
2. Choose your instance type
3. Create the cluster and note the cluster/instance identifiers from the output

**Wait for provisioning:** Clusters typically take 1-2 minutes to become ready.

### Step 4 — Get Cluster Information

Use the cluster list/info commands (discovered via `tp cluster --help`) to get:
- Cluster ID
- Instance ID(s) — needed for SSH
- IP address — needed for rsync/scp
- Status (should be RUNNING)

### Step 5 — Transfer Code to Cluster

**Recommended: Use rsync (fast, resumable, efficient)**
```bash
rsync -avz \
  --exclude=".git" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude="venv/" \
  --exclude="outputs/" \
  ./ ubuntu@<cluster-ip>:~/my-project/
```

**Alternative: Use scp for single files**
```bash
scp script.py ubuntu@<cluster-ip>:~/
scp -r ./data/ ubuntu@<cluster-ip>:~/data/
```

### Step 6 — SSH into Cluster

Use the SSH command discovered via `tp ssh --help`, passing the appropriate instance or cluster identifier.

**You're now on the GPU cluster!** All subsequent commands run on the cluster.

### Step 7 — Setup Environment on Cluster

**7.1 Verify GPU:**
```bash
nvidia-smi
```
Expected output: GPU(s) listed with available memory

**7.2 Navigate to your project:**
```bash
cd ~/my-project
ls -la  # Verify files transferred
```

**7.3 Install dependencies:**
```bash
# Using requirements.txt
pip install -r requirements.txt

# Or install packages individually
pip install torch transformers datasets accelerate
```

**7.4 Set environment variables (if needed):**
```bash
# Load from .env
export $(cat .env | xargs)

# Or set manually
export HUGGINGFACE_TOKEN=hf_your_token_here
export CUDA_VISIBLE_DEVICES=0
```

**7.5 Download large datasets (if needed):**
```bash
# Download directly on cluster (faster than transferring)
wget https://example.com/large-dataset.tar.gz
tar -xzf large-dataset.tar.gz

# Or use Hugging Face datasets (downloads automatically)
# No action needed - will download when script runs
```

### Step 8 — Run Your Script on GPU

**8.1 Quick test first:**
```bash
# Run with minimal data to verify everything works
python your_script.py --max_samples 100 --num_epochs 1
```

**8.2 Full production run:**
Often times the user will input information to the prompt of how to do production run
```bash
# Option 1: Direct execution
python train.py --num_epochs 5 --batch_size 32

# Option 2: Use screen/tmux (survives SSH disconnection)
screen -S training
python train.py --num_epochs 5
# Press Ctrl+A then D to detach
# Reconnect later with: screen -r training

# Option 3: Use nohup (runs in background)
nohup python train.py > training.log 2>&1 &
tail -f training.log  # Monitor progress
```

**8.3 Monitor progress:**
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f training.log

# Monitor disk space
df -h
```

### Step 9 — Retrieve Outputs

**From your LOCAL machine** (open new terminal):
```bash
CLUSTER_IP=<ip-from-cluster-info>

# Download outputs
rsync -avz ubuntu@$CLUSTER_IP:~/my-project/outputs/ ./outputs/

# Download specific files
rsync -avz ubuntu@$CLUSTER_IP:~/my-project/model.pt ./
rsync -avz ubuntu@$CLUSTER_IP:~/my-project/logs/ ./logs/
```

**Pro tip:** Use `--progress` flag to see transfer progress:
```bash
rsync -avz --progress ubuntu@$CLUSTER_IP:~/my-project/outputs/ ./outputs/
```

### Step 10 — Destroy the Cluster

** CRITICAL: Always delete the cluster to avoid charges**

Use the cluster destroy command (discovered via `tp cluster --help`), passing the cluster identifier.

Verify deletion by listing clusters again — it should no longer appear.

**Cost reminder:** Clusters bill continuously until destroyed!

---

## NFS Storage (Persistent Volumes)

TensorPool offers NFS storage for persistent data across cluster lifecycles and for sharing data across multi-node clusters. Discover the exact commands via:

```bash
tp storage --help
```

---

## Resources

- **TensorPool Docs:** https://docs.tensorpool.dev
- **Cluster Quickstart:** https://docs.tensorpool.dev/clusters-quickstart
- **Instance Types:** https://docs.tensorpool.dev/resources/instance-types
- **CLI Reference:** https://docs.tensorpool.dev/cli/overview
- **Community Slack:** https://tensorpool.dev/slack
- **Contact:** team@tensorpool.dev

---