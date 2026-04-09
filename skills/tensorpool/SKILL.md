---
name: tensorpool
description: This skill helps users run ML workloads on TensorPool GPU infrastructure. It supports two workflows — (1) Jobs, a fire-and-forget git-style interface for training experiments via tp job, and (2) Clusters, interactive SSH-based GPU development via tp ssh. Use this when a user wants to run scripts on cloud GPUs, submit training jobs, manage GPU clusters, or scale up to professional hardware (H100, H200, B200, B300, etc.).
---

# SKILL: Run ML Workloads on TensorPool GPUs

## Overview

This skill helps users run machine learning workloads on TensorPool GPU infrastructure. There are **two workflows**:

1. **Jobs (fire-and-forget):** Configure a `.tp.toml` file, push it to a cluster, and retrieve outputs when done. Best for experiments, batch training, and unattended runs.
2. **Clusters (interactive):** SSH into a GPU cluster, transfer code, run interactively. Best for debugging, development, and hands-on work.

**Trigger phrases:** "run this on H100", "submit a training job", "scale to TensorPool", "run on cloud GPU", "push a job", "run on B200", "create a cluster", "tp job", "tp ssh"

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
tp job --help
tp cluster --help
tp ssh --help
tp storage --help
tp object-storage --help
tp me --help
```

**Use the output of `--help` to determine:**
- Exact command names and syntax
- Required vs optional arguments
- Available flags and their current names
- Available instance types

**If a command fails or the syntax has changed**, re-run `--help` on that subcommand to get the current usage.

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

3. **SSH key available** (needed for rsync/scp and job pull):
   ```bash
   ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
   ```

4. **User has a TensorPool account** — sign up at [tensorpool.dev](https://tensorpool.dev)

---

## Choosing an Instance Type

| GPU | Memory | Best For |
|-----|--------|----------|
| H100 | 80GB | General training, fine-tuning 7B-70B models |
| H200 | 141GB | Large context, 70B+ models, memory-bound workloads |
| B200 | 192GB | Latest gen, native FP4/FP6, fastest training |
| B300 | Latest | Newest generation, highest performance |
| L40S | 48GB | Inference, smaller training workloads |

Instance types come in configurations like `1xH100`, `2xH100`, `4xH100`, `8xH100`, etc.

---

# WORKFLOW 1: Jobs (Fire-and-Forget)

Jobs are the recommended way to run training experiments. You configure a `.tp.toml` file defining your commands and outputs, push it to a cluster, and pull results when done.

**When to use Jobs:** batch training, hyperparameter sweeps, unattended experiments, anything that doesn't need interactive access.

## Job Workflow Overview

```
Analyze script → Create .tp.toml config → Create/select cluster → tp job push → tp job listen → tp job pull → (optional) teardown cluster
```

### Step J1 — Analyze the Script

Before configuring a job, understand:

1. **What does the script do?** (training, inference, data processing)
2. **What are the dependencies?** (check imports, requirements.txt)
3. **What data does it need?** (local files, datasets, pretrained models)
4. **What outputs does it produce?** (model checkpoints, logs, results)
5. **GPU requirements:** Single GPU? Multi-GPU? Memory needs?

### Step J2 — Create the Job Configuration

Initialize a config file. Figure out how by running `tp job --help`

This generates a `{name}.tp.toml` file. Edit it with three sections:

**commands** — Sequential shell commands executed in a fresh virtual environment:
```toml
commands = [
    "pip install -r requirements.txt",
    "python train.py --epochs 100 --batch_size 32",
]
```

**outputs** — Files and directories to preserve after the job completes. Supports glob patterns:
```toml
outputs = [
    "checkpoints/",
    "model.pth",
    "results.json",
    "logs/",
    "model_*.pth",
]
```

**ignore** — Files to exclude from upload to the cluster:
```toml
ignore = [
    ".venv",
    "venv/",
    "__pycache__/",
    ".git",
    "*.pyc",
    "data/",
    "outputs/",
]
```

### Step J3 — Create or Select a Cluster

Here, you select a cluster to submit a job to, or create a new one. 

Figure out how to list or create new clusters by reading through the clusters workflow below.

Note the cluster ID (e.g., `c-xxx`) from the output.

### Step J4 — Push the Job

You should then push the job to the cluster. Figure out how by running `tp job --help`

### Step J5 — Monitor the Job

You should then monitor the job. There are various tools to do this in `tp job --help` such as listening and checking for info.

### Step J6 — Pull Results

After the job is done, download all results. You probably will want to use the --force command to overwrite existing local files.

Figure out how to do this by accessing `tp job --help`.

### Managing Jobs

You can also cancel and delete jobs by fetching those commands from `tp job --help`

---

# WORKFLOW 2: Clusters (Interactive SSH)

Clusters give you full interactive SSH access to GPU machines. You manually transfer code, install dependencies, and run scripts.

**When to use Clusters:** debugging, interactive development, Jupyter notebooks, exploratory work, anything requiring a shell.

## Cluster Workflow Overview

```
Analyze script → Prepare code → Create cluster → Transfer code → SSH in → Setup env → Run script → Retrieve outputs → Destroy cluster
```

### Step C1 — Analyze the Local Script

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

### Step C2 — Prepare the Script for Cloud Execution

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

### Step C3 — Create GPU Cluster

**To create a cluster:**
1. Run `tp cluster --help` to see the exact create syntax and required arguments
2. Choose your instance type
3. Create the cluster and note the cluster/instance identifiers from the output

**Wait for provisioning:** Clusters typically take 1-2 minutes to become ready.

### Step C4 — Get Cluster Information

Use the cluster list/info commands (discovered via `tp cluster --help`) to get:
- Cluster ID
- Instance ID(s) — needed for SSH
- IP address — needed for rsync/scp
- Status (should be RUNNING)

### Step C5 — Transfer Code to Cluster

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

### Step C6 — SSH into Cluster

Use the SSH command discovered via `tp ssh --help`, passing the appropriate instance or cluster identifier.

**You're now on the GPU cluster!** All subsequent commands run on the cluster.

### Step C7 — Setup Environment on Cluster

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

### Step C8 — Run Your Script on GPU

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

### Step C9 — Retrieve Outputs

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

### Step C10 — Destroy the Cluster

** CRITICAL: Always delete the cluster to avoid charges**

Use the cluster destroy command (discovered via `tp cluster --help`), passing the cluster identifier.

Verify deletion by listing clusters again — it should no longer appear.

**Cost reminder:** Clusters bill continuously until destroyed!

---

## Object Storage (S3-Compatible)

TensorPool provides S3-compatible object storage for persistent data across jobs and clusters. This storage type is slower than NFS but globally replicated. No ingress/egress fees.

You can discover how to use this tool by running `tp object-storage --help`

---

## Shared Storage (Persistent Volumes)

TensorPool offers very fast shared storage for persistent data across cluster lifecycles and for sharing data across multi-node clusters. Discover the exact commands via:

```bash
tp storage --help
```

---

## Choosing Between Jobs and Clusters

| | Jobs | Clusters |
|---|---|---|
| **Best for** | Batch training, experiments | Interactive development, debugging |
| **Access** | No SSH needed | Full SSH access |
| **Workflow** | Configure .tp.toml → push → pull | SSH in → work manually |
| **Auto-cleanup** | `--teardown` flag | Must manually destroy |
| **Monitoring** | `tp job listen` | `nvidia-smi`, logs |
| **Output retrieval** | `tp job pull` | rsync/scp manually |
| **Cost safety** | `--teardown` prevents waste | Easy to forget to destroy |

**Rule of thumb:** If you can define your workflow as a sequence of commands → use Jobs. If you need to poke around interactively → use Clusters.

---

## Resources

- **TensorPool Docs:** https://docs.tensorpool.dev
- **Jobs Quickstart:** https://docs.tensorpool.dev/quickstart
- **Clusters Quickstart:** https://docs.tensorpool.dev/clusters-quickstart
- **Instance Types:** https://docs.tensorpool.dev/resources/instance-types
- **CLI Reference:** https://docs.tensorpool.dev/cli/overview
- **Community Slack:** https://tensorpool.dev/slack
- **Contact:** team@tensorpool.dev

---
