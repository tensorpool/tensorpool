# MNIST Example

This example implements a convolutional neural network (CNN) to classify handwritten digits.

## Setup:
```
python -m venv venv # Using a virtual environment is recommended
pip install -r requirements.txt
```

# Usage examples:
<details>
<summary><code>tensorpool train on 100 epochs on an T4, save the weights</code></summary>

Generated `tp-config.toml`:
```toml
commands = [
    "pip install -r requirements.txt",
    "python mnist.py --epochs 100 --save-model",
]
optimization_priority = "PRICE"
gpu = "T4"
```
This will run your job on the cheapest T4 instance across all cloud providers and regions
</details>

<details>
<summary><code>tensorpool train on AWS with the cheapest L4 you can find</code></summary>

Generated `tp-config.toml`:
```toml
commands = [
    "pip install -r requirements.txt",
    "python mnist.py",
]
optimization_priority = "PRICE"
gpu = "L4"
cloud = "AWS"
```
This will run your job on the cheapest L4 instance across all AWS regions

For this example you'll want to save your weights as well, otherwise you'll lose them! You should specify the `--save-model` flag like in the previous example.
</details>


<details>
<summary><code>tensorpool do a dry run without cuda</code></summary>

Generated `tp-config.toml`:
```toml
commands = [
    "pip install -r requirements.txt",
    "python mnist.py --dry-run --no-cuda",
]
optimization_priority = "PRICE"
```
Dry runs are useful for testing your setup and code without doing a full training run.

TensorPool will run your job on the cheapest instance across all cloud providers and regions
</details>

<details>
<summary><code>tensorpoool run training with a 128 batch size and 0.9 learning rate ASAP</code></summary>

Generated `tp-config.toml`:
```toml
commands = [
    "pip install -r requirements.txt",
    "python mnist.py --batch-size 128 --lr 0.9",
]
optimization_priority = "TIME"
```
This will run training on the fastest available instance across all cloud providers and regions.

Since no GPU is specified, TensorPool will choose the fastest available instance type in the nearest region / cloud provider.
</details>

Since this script exposes many command line arguments TensorPool is able to use them to help you create a job configurations, so you can develop and experiment faster.

Prefer not to use natural language? No problem, feel free define the `tp-config.toml` youself. See [the README](https://github.com/tensorpool/tensorpool?tab=readme-ov-file#configuration) on how to do this.


Common questions:
<details>
<summary>What is MNIST?</summary>
<br>
MNIST is a dataset of handwritten digits that is commonly used for training computer vision models. It's considered the "hello world" of machine learning.
</details>
<details>
<summary>How does TensorPool know what arguments my script accepts?</summary>
<br>
TensorPool detects the command line arguments accepted by your script. This means all command line arguments you define in your code (like batch size, learning rate, etc.) become available through TensorPool's natural language interface without any additional configuration.
</details>
