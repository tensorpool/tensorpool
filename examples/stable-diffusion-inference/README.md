# Stable Diffusion Inference

This is an example of running inference with the Runway's Stable Diffusion v1.5 model using TensorPool to access cloud GPUs.

## Setup:
```bash
python -m venv venv # Using a virtual environment is recommended
source venv/bin/activate # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
pip install tensorpool
```

## Usage:

First, let's create a `tp.config.toml`, to define your task and your constraints.

We can manually create a `tp.config.toml` with `tp config new`! See [the README](https://github.com/tensorpool/tensorpool?tab=readme-ov-file#configuration) for more info.

Or you can use TensorPool's natural language interface to have one auto-generated for you. Simply run `tp config <describe your task>`

For example, we can do things like:

<code>tp config run main.py with the prompt "a photorealistic landscape with mountains and a lake at sunset" on an A100</code>

Generated `tp.config.toml`:
```toml
commands = [
  "pip install -r requirements.txt",
  "python stable_diffusion_inference.py --prompt \"a photorealistic landscape with mountains and a lake at sunset\""
]
optimization_priority = "TIME"
gpu = "A100"
```
This will run your job on an A100 instance for fastest generation.

But there's more! Our script exposes many command line arguments that allow you to customize your image generation. TensorPool detects these command line arguments and is able to use them to help you create job configurations.

So we can ask TensorPool to do things like: 

<code>tp config run main.py with the prompt "cyberpunk city at night with neon lights" with a negative prompt "blurry, low quality" generating 4 images at 768x512 resolution on an A100</code>

Generated `tp.config.toml`:
```toml
commands = [
  "pip install -r requirements.txt",
  "python stable_diffusion_inference.py --prompt \"cyberpunk city at night with neon lights\" --negative-prompt \"blurry, low quality\" --num-images 4 --width 768 --height 512"
]
optimization_priority = "TIME"
gpu = "A100"
```

As you can see, TensorPool correctly converted your description into the appropriate command line arguments.

To now deploy and run your code on the cloud, you simply run: 

```bash
tp run
```

BOOM! That's it! You're generating high-quality images with Stable Diffusion on a cloud GPU!

Once your images are generated, you can pull them to your local machine by running:

```bash
tp pull <job_id>
```

TensorPool will begin fetching your generated images from the "outputs" directory on the remote machine and downloading them to your local machine!

## Advanced Options

Our script supports several additional parameters you can include in your TensorPool configuration:

- `--steps` - Number of inference steps (default: 30)
- `--guidance-scale` - How strictly to follow the prompt (default: 7.5)
- `--seed` - Set a seed for reproducible results
- `--fp16` - Enable half-precision for faster generation

For example:

<code>tp config run main.py with the prompt "detailed portrait of a viking warrior" with 50 steps and a seed of 42 on an A100</code>

Generated `tp.config.toml`:
```toml
commands = [
  "pip install -r requirements.txt",
  "python stable_diffusion_inference.py --prompt \"detailed portrait of a viking warrior\" --steps 50 --seed 42"
]
optimization_priority = "TIME"
gpu = "A100"
```
