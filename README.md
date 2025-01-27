# TensorPool

TensorPool is the easiest way to execute ML jobs in the cloud for >50% cheaper. No infrastructure setup needed, just one command to use cloud GPUs.


- **>50% cheaper than traditional clouds**: Through automatic spot node recovery tech, TensorPool gives you the prices savings of spot nodes with the reliability of on demand instances by resuming jobs if they get interupted.
- **Natural Language Job Configuration**: Describe your ML training or inference job in plain English
- **Zero Infrastructure Setup**: No GCP, No AWS, No Docker, no Kubernetes, no cloud configuration or cloud accounts required
- **Optimization Priorities**: optimize for price or time. We search for the cheapest instance types across several cloud providers (currently GCP and AWS, more coming soon!)


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
3. Upload and execute it in the cloud on GPUs for >50% cheaper than on-demand instances on traditional cloud providers


While TensorPool lets you define your job in natural language, you can also manually define a `tp-config.toml` yourself.
See [Configuration](#configuration) for details on the tp-config.toml format.

## Example Usage

```bash
# Run a simple training job
tensorpool train my model for 5 epochs
```
```bash
# Specify specific cloud providers, regions, and GPUs
tensorpool train in AWS us-east-1 using an L4
```
```bash
# Define arguments that can be passed to your python script
tensorpool run with a learning rate of 0.9 and save my weights
```
```bash
# Making your own tp-config.toml? Just do:
tensorpool run
```
Check your jobs and their outputs on the [dashboard](https://tensorpool.dev/dashboard)

See more examples in the [examples](

## Configuration

The heart of Tensorpool is the `tp-config.toml` which can automatically be created for you. You can:
- Review and modify it before execution
- Reuse it for future runs
- Version control it with your code

Here’s what a `tp-config.toml` looks like:
```toml
commands = [
    # List of commands to run for the job
    "python main.py",
]
optimization_priority = "PRICE"  # Either "PRICE" or "TIME".

# Optional fields
gpu = "L4"          # Optional: Currently only "T4" or "L4" are supported (more GPUs coming soon!)
cloud = "GCP"       # Optional: "GCP" or "AWS"
region = "us-west1" # Optional: datacenter region
```

<details>
<summary>What does <code>optimization_priority</code> mean?</summary>
<br>

`optimization_priority = "PRICE"` means that TensorPool will search for the cheapest instance types across all cloud providers.
`optimization_priority = "TIME"` means that TensorPool will search for the fastest instance types (best GPU) across all cloud providers.
`cloud` or `region` can be specified to limit the search to a specific cloud provider or region.
</details>

## Best Practices
- **Save your outputs**: Always save your model weights and outputs to disk, you'll get them back at the end of the job!
  - Don't save files outside of your project directory, you won't be able to get them back
- **Download datasets and big files within your script**: All TensorPool machines are equipped 10+Gb/s networking and 100Gb of storage, so large files can be downloaded faster if done within your script

## Getting Help
- [tensorpool.dev](https://tensorpool.dev)
- team@tensorpool.dev
- https://x.com/TensorPool

## Why TensorPool?
- **Simplicity**: Natural language job configuration
- **Zero Config**: No cloud setup, no unnecessary machine configuration, no Docker
- **Cost Effective**: Prices savings of spot instances with the reliability of on demand instances

Get started today at [tensorpool.dev](https://tensorpool.dev)!
