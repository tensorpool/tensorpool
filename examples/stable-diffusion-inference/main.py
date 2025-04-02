#!/usr/bin/env python3
"""
Stable Diffusion Inference Script
---------------------------------
A command-line tool for generating images using Stable Diffusion.
Prioritizes GPU but falls back to CPU if needed.
"""

import argparse
import os
import time
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation"
    )

    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt to specify what you don't want in the image"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save generated images"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width (must be divisible by 8)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height (must be divisible by 8)"
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps"
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale (how strictly to follow the prompt)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use half-precision (faster but slightly lower quality)"
    )

    return parser.parse_args()


def main():
    """Run the Stable Diffusion inference pipeline."""
    start_time = time.time()
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Hard-coded model
    model_id = "runwayml/stable-diffusion-v1-5"

    # Try to use GPU, but fall back to CPU if unavailable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: GPU not available. Falling back to CPU, which will be much slower.")
        print("Generation may take several minutes per image.")

    print(f"Loading model: {model_id}")
    print(f"Using device: {device}")

    # Configure model precision
    torch_dtype = torch.float16 if args.fp16 and device == "cuda" else torch.float32
    if args.fp16 and device == "cpu":
        print("WARNING: Half-precision (fp16) requested but not available on CPU. Using full precision.")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None  # For faster inference
    )

    # Use DPM-Solver++ scheduler for faster inference with good quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Move model to device
    pipe = pipe.to(device)

    # Enable memory optimization if on GPU
    if device == "cuda":
        pipe.enable_attention_slicing()

    # Set seed for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"Using seed: {args.seed}")

    # Generate images
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nGenerating images with prompt:")
    print(f'"{args.prompt}"')
    if args.negative_prompt:
        print(f'Negative prompt: "{args.negative_prompt}"')

    for i in range(args.num_images):
        inference_start = time.time()

        # Run inference
        with torch.autocast(device, enabled=(device == "cuda" and args.fp16)):
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

        # Create filename
        prompt_slug = args.prompt.lower().replace(' ', '_')[:50]
        safe_slug = ''.join(c if c.isalnum() or c == '_' else '' for c in prompt_slug)
        filename = f"{timestamp}_{safe_slug}_{i + 1:03d}.png"
        filepath = os.path.join(args.output_dir, filename)

        # Save image
        image.save(filepath)

        inference_time = time.time() - inference_start
        print(f"Image {i + 1}/{args.num_images} saved to {filepath} (took {inference_time:.2f}s)")

    total_time = time.time() - start_time
    print(f"\nGeneration complete! Total time: {total_time:.2f}s")
    print(f"Generated {args.num_images} image(s) in directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()