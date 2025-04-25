#!/usr/bin/env python
import os
import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Update Ultravox training configuration")

    # Model parameters
    parser.add_argument("--text_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Text model to use")
    parser.add_argument("--audio_model", type=str, default="openai/whisper-medium",
                        help="Audio model to use")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")

    # Output config name
    parser.add_argument("--output_config", type=str, default="custom_config.yaml",
                        help="Name of the output config file (saved in same dir as release_config.yaml)")

    args = parser.parse_args()

    # Find ultravox directory
    ultravox_dir = "ultravox"
    if not os.path.exists(ultravox_dir):
        print(f"Error: {ultravox_dir} directory not found. Make sure you're in the correct directory.")
        return

    # Find release_config.yaml
    release_config_path = None
    possible_paths = [
        os.path.join(ultravox_dir, "ultravox/training/configs/release_config.yaml"),
        os.path.join(ultravox_dir, "training/configs/release_config.yaml"),
        os.path.join(ultravox_dir, "configs/release_config.yaml")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            release_config_path = path
            break

    if not release_config_path:
        print("Error: Could not find release_config.yaml")
        return

    print(f"Using base config from: {release_config_path}")

    # Load the base config
    with open(release_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update model settings
    config['text_model'] = args.text_model
    config['audio_model'] = args.audio_model

    # Update training parameters
    config['batch_size'] = args.batch_size
    config['grad_accum_steps'] = args.grad_accum_steps
    config['max_steps'] = args.max_steps
    config['lr'] = args.lr

    # Disable wandb
    config['report_to'] = 'none'

    # Save the config in the same directory as the release config
    config_dir = os.path.dirname(release_config_path)
    output_path = os.path.join(config_dir, args.output_config)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Configuration saved to: {output_path}")
    print("\nTo run training with this config:")
    print(f"cd {ultravox_dir}")
    print(f"python -m ultravox.training.helpers.prefetch_weights --config_path {os.path.abspath(output_path)}")
    print(f"torchrun --nproc_per_node=1 -m ultravox.training.train --config_path {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()