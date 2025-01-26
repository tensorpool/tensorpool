# TensorPool

TensorPool is the easiest way to execute ML jobs in the cloud. No infrastructure setup needed, just one command to use cloud GPUs.

## Prerequisites
1. Create an account at [tensorpool.dev](https://tensorpool.dev)
2. Get your API key from the [dashboard](https://tensorpool.dev/dashboard)
3. Install the CLI:
```bash
pip install tensorpool
```

## Quick Start
TensorPool uses natural language to understand and configure your ML jobs.

Simply describe what you want to do:

```bash
tensorpool train my model for 100 epochs on an L4
```

Behind the scenes TensorPool:
1. Generates a job configuration (`tp-config.toml`) based off of your project directory
2. Let's you review and modify the configuration
3. Upload and execute it in the cloud on GPUs for >50% cheaper than on-demand instances


While TensorPool lets you define your job in natural language, you can also manually define a `tp-config.toml` yourself.
See [Configuration](#configuration) for details on the tp-config.toml format.


## Key Features
- **Natural Language Job Configuration**: Describe your ML training or inference job in plain English
- **Zero Infrastructure Setup**: No GCP, No AWS, No Docker, no Kubernetes, no cloud configuration or cloud accounts required
- **>50% cheaper than on demand instances**: TensorPool gives you the prices savings of spot instances with the reliability of on demand instances through automatic recovery and checkpointing systems
- **Optimization Priorities**: optimize for price or time. We search for the cheapest instance types across several cloud providers (currently GCP and AWS, with more coming soon)

## Example Usage

```bash
# Run a simple training job
tensorpool train my model for 5 epochs"
```
```bash
# Specify specific cloud providers, regions, and GPUs
tensorpool train in AWS us-east-1 using an L4
```
```bash
# Define arguments that can be passed to your python script
tensorpool run with a learning rate of 0.9 and save my weights
```
See your job status and output on the [dashboard](https://tensorpool.dev/dashboard)

## Configuration

The heart of Tensorpool is the `tp-config.toml` which will automatically be created in your project directory. You can:
- Review and modify it before execution
- Reuse it for future runs
- Version control it with your code

Here’s what a `tp-config.toml` looks like:
```toml
commands = [
    # List of commands to run for the job
    "python main.py",
]
optimization_priority = "PRICE"  # Either "PRICE" or "TIME"

# Optional fields
gpu = "L4"          # Optional: Currently only "T4" or "L4" are supported (more GPUs coming soon)
cloud = "GCP"       # Optional: "GCP" or "AWS"
region = "us-east1" # Optional: datacenter region
```

## Getting Help
- [tensorpool.dev](https://tensorpool.dev)
- team@tensorpool.dev
- https://x.com/TensorPool

## Why TensorPool?
- **Simplicity**: Natural language job configuration
- **Zero Config**: No cloud setup, no unnecessary machine configuration, no Docker
- **Cost Effective**: Prices savings of spot instances with the reliability of on demand instances

Get started today at [tensorpool.dev](https://tensorpool.dev)!
