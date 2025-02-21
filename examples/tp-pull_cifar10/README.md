
# TP Pull Example: Training on CIFAR-10

This example demonstrates how `tp pull` works within our GPU training platform. `tp pull` allows you to fetch your job's remote files and download them directly to your local machine. You can use it to retrieve intermediate checkpoints or final outputs from your training job without leaving your IDE.

In this demo, we'll train a simple Convolutional Neural Network (CNN) on the CIFAR-10 dataset—a widely used dataset in machine learning that consists of 60,000 32x32 color images across 10 classes. Let’s dive in!

## Environment Setup

```bash
python -m venv venv  # It's recommended to use a virtual environment
pip install -r requirements.txt
```

## Configuring Your Job

Our training script performs several key steps:
- Downloads the CIFAR-10 dataset.
- Trains a CNN model for a specified number of epochs.
- Generates helpful training log curves.
- Tests the model by generating and saving sample predictions.

Before deploying your job to the cloud, you need to create a configuration file (`tp.config.toml`) that defines your task and resource constraints. You have two options:

1. **Manual Configuration:**  
   Use `tp config new` to create a new configuration file manually. See [the README](https://github.com/tensorpool/tensorpool?tab=readme-ov-file#configuration) for more details.

2. **Automatic Configuration via Natural Language:**  
   For example, you can run:
   ```bash
   tp config run my script for 10 epochs on an L4 and optimize for price
   ```
   This command generates the following `tp.config.toml`:
   ```toml
   commands = [
     "pip install -r requirements.txt",
     "python cifar_10.py --epochs 10"
   ]
   optimization_priority = "PRICE"
   gpu = "L4"
   ```
   This configuration ensures your job runs on the cheapest L4 GPU instance available across all cloud providers and regions.

## Deploying and Running Your Job

Once your configuration is set up, deploy and run your code on the cloud with:
```bash
tp run
```
Your model is now deployed and training on cloud GPUs. (Note: When your job is initiated, a `<job_id>` is generated. This ID is used to reference your job in subsequent commands. Please refer to the platform documentation if you're unsure how to obtain your job ID.)

## Retrieving Files with `tp pull`

During training, you might want to retrieve files—like intermediate weights, checkpoints, or prediction outputs—without stopping your job. To pull files:

- **To retrieve all files:**
  ```bash
  tp pull <job_id>
  ```
- **To retrieve specific files:**
  ```bash
  tp pull <job_id> [files...]
  ```
  For example, to pull only the training curves:
  ```bash
  tp pull <job_id> /outputs/training_curves.png
  ```

### Excluding Unnecessary Files

If your script downloads many files you don't need, you can update your configuration to ignore specific directories. For example to exclude the `data/` folder, we modify the `tp.config.toml` as follows:

```toml
commands = [
  "pip install -r requirements.txt",
  "python cifar_10.py --epochs 10"
]
optimization_priority = "PRICE"
gpu = "L4"

ignore = ["data/"]
```

Now, when you run `tp pull <job_id>`, TensorPool will download all files except those in the `data/` folder.

## Additional Tips

- **Troubleshooting:**  
  If `tp pull` fails to fetch files, ensure that your network connection is stable and that you are using the correct job ID. Checking the logs for any error messages can also help diagnose issues.

- **Best Practices:**  
  Avoid pulling extremely large files frequently during training to minimize performance overhead.

And that's it! With `tp pull`, you simplify your development cycle by effortlessly accessing outputs and intermediate results from your cloud-based training jobs.

Happy training!