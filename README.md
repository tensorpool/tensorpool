# TensorPool

TensorPool is the easiest way to execute ML jobs in the cloud for >50% cheaper. No infrastructure setup needed, just one command to use cloud GPUs.

## Features
- **>50% cheaper than traditional cloud providers**: TensorPool's automatic spot node recovery, you get the prices savings of spot nodes with the reliability of on demand instances.
  - Tensorpool will automatically resume your job if it gets interrupted. No code changes required.
- **Natural Language Job Configuration**: Describe your ML training or inference job in plain English
  - Or do it manually if you prefer!
- **Zero Infrastructure Setup**: No GCP, No AWS, No Docker, no Kubernetes, no cloud configuration or cloud accounts required
  - Use GCP or AWS without setting up an account
- **Optimization Priorities**: optimize for price or time.
  - We search for the cheapest instance types across several cloud providers (currently GCP and AWS, more coming soon!)

## Prerequisites
1. Create an account at [tensorpool.dev](https://tensorpool.dev)
2. Get your API key from the [dashboard](https://tensorpool.dev/dashboard)
3. Install the CLI:
```bash
pip install tensorpool
```

## Quick Start
TensorPool uses natural language to understand and orchestrate your ML jobs.

Simply describe what you want to do:

```bash
tensorpool train my model for 100 epochs on an L4
```

Behind the scenes TensorPool:
1. Generates a job configuration (`tp-config.toml`) based off of your project directory
2. Let's you review and modify the configuration
3. Uploads and executes it in the cloud on GPUs for >50% cheaper than on-demand instances on traditional cloud providers

Prefer to not use natural language? You can also manually define a `tp-config.toml` yourself!

See [Configuration](#configuration) for details the schema.

## Demo Video
[![TensorPool Llama 3 Inference](https://img.youtube.com/vi/QM6LHB-nLsE/0.jpg)](https://www.youtube.com/watch?v=QM6LHB-nLsE)

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
# Prefer to make your own tp-config.toml? Just do:
tensorpool run
```
Check your jobs and their outputs on the [dashboard](https://tensorpool.dev/dashboard)

More examples can be found [here](https://github.com/tensorpool/tensorpool/tree/main/examples/mnist)

## Configuration

The heart of Tensorpool is the `tp-config.toml` which can automatically be created for you, or you can create it yourself!

When you run `tensorpool <your command>`, TensorPool will automatically generate a `tp-config.toml` file in your project directory.

Here's a complete list of all fields supported in the `tp-config.toml`:
```toml
commands = [
    # List of commands to run for the job as if you were starting from a fresh virtual environment
    "pip install -r requirements.txt",
    "python main.py",
]
optimization_priority = "PRICE"  # Either "PRICE" or "TIME".

# Optional fields
gpu = "L4"          # Optional: Currently only "T4" or "L4" are supported (more GPUs coming soon!)
cloud = "GCP"       # Optional: "GCP" or "AWS"
region = "us-west1" # Optional: datacenter region
```

The beauty of the `tp-config.toml` is its simplicity and flexibility, this allows you to:
- Review and modify it before execution
- Reuse it for future runs
- Version control it with your code

<details>
<summary>What does <code>optimization_priority</code> mean?</summary>
<br>

`optimization_priority = "PRICE"` means that TensorPool will search for the cheapest instance types across all cloud providers.

`optimization_priority = "TIME"` means that TensorPool will search for the fastest instance types (best GPU) across all cloud providers.

`cloud` or `region` can be specified to limit the search to a specific cloud provider or region.
</details>

<details>
<summary>What GPUs are supported?</summary>
<br>
Currently T4s and L4s are supported. More GPUs are coming soon!
</details>

<details>
<summary>What cloud providers are supported?</summary>
<br>
Currently GCP and AWS are supported. More cloud providers are coming soon!
</details>

## Best Practices
- **Use commands like you're starting from scratch**
  - TensorPool executes your job in a fresh environment - include all setup commands like installing dependencies, downloading data, etc
- **Save your outputs**
  - Always save your model weights and outputs to disk, you'll get them back at the end of the job!
  - Don't save files outside of your project directory, you won't be able to get them back
- **Download datasets and big files within your script**
  - All TensorPool machines are equipped 10+Gb/s networking and 100Gb of storage, so large files can be downloaded faster if done within your script
- **Run from the root of your project**
  - TensorPool will send your project directory to the cloud, so make sure you're in the right directory
  - Don't run from your home directory or a subdirectory!

## Getting Help
- [tensorpool.dev](https://tensorpool.dev)
- team@tensorpool.dev
- https://x.com/TensorPool

## Why TensorPool?
- **Simplicity**: Natural language job configuration
- **Zero Config**: No cloud setup, no unnecessary machine configuration, no Docker
- **Cost Effective**: Prices savings of spot instances with the reliability of on demand instances

Get started today at [tensorpool.dev](https://tensorpool.dev)!
