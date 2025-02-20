# DeepSeek R1 Fine-Tuning
This is an example of fine-tuning the distilled DeepSeek R1 model on the wikitext dataset. 

## Setup:
```
python -m venv venv # Using a virtual environment is recommended
pip install -r requirements.txt
```

# Usage:
First let's create a `tp.config.toml`, to define your task and your constraints

We can manually create a `tp.config.toml` with `tp config new`! See [the README](https://github.com/tensorpool/tensorpool?tab=readme-ov-file#configuration) for more info.

or you can use our natural language interface to have one auto-generated for you. Simply run `tp config <describe your task>`

For example, we can do things like:

<code>tp config run my train script on an A100 and optimize for price</code>

Generated `tp.config.toml`:
```toml
commands = [
  "pip install -r requirements.txt",
  "python train.py"
]
optimization_priority = "PRICE"
gpu = "A100"
```
This will run your job on the cheapest A100 instance across all cloud providers and regions

But there's more! The training script exposes many command line arguments that allows you to pass in various hyperparameters. 
TensorPool detects these command line arguments and is able to use them to help you create a job configurations, so you can develop and experiment faster.

So we can ask tensorpool to do things like: 

<code>tp config run my train script for 10 epochs with a lr of 5e-5 on an A100</code>

Generated `tp.config.toml`:
```toml
commands = [
  "pip install -r requirements.txt",
  "python train.py --num_train_epochs 10 --learning_rate 5e-5"
]
optimization_priority = "TIME"
gpu = "A100"
```
As seen TensorPool correctly resolved your description into the correct command line arguments

To now deploy and run your code on the cloud, you simply run: 

`tp run`

BOOM! That's it! You're fine-tuning DeepSeek on a cloud GPU! 

But what if you want to store intermediate weights/checkpoints and pull them to your local machine mid-job?
No worries! Let's first get TensorPool to generate a `tp.config.toml` that passes in the correct arguments for a saving strategy

<code>tp config run my train script for 10 epochs save every 2 epochs and run on an A100</code>

Generated `tp.config.toml`:
```toml
commands = [
  "pip install -r requirements.txt",
  "python train.py --num_train_epochs 10 --save_strategy epoch --save_steps 2"
]
optimization_priority = "TIME"
gpu = "A100"
```

As seen TensorPool correctly defines your saving strategy. Once they are ready, you can pull your checkpoints by running: 

`tp pull <job_id>`

and TensorPool will begin fetching your remote files and downloading them to your local machine!