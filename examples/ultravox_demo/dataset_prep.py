import os
from datasets import load_dataset, Audio
import torch
from tqdm import tqdm
import json
from transformers import pipeline

# Configuration
OUTPUT_DIR = "processed_dataset"
SAMPLE_RATE = 16000  # Ultravox expects 16kHz audio
NUM_SAMPLES = 1000  # Recommended minimum for fine-tuning


# Initialize an LLM to generate continuations (optional but improves results)
def init_llm():
    try:
        # Use a smaller model for generating continuations
        pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
        return pipe
    except Exception as e:
        print(f"Warning: Could not load LLM for continuations: {e}")
        print("Will use direct transcriptions instead")
        return None


def main():
    print("Loading LJ Speech dataset")

    # Load the LJ Speech dataset (about 2.6GB, but more likely to download successfully)
    # We can use a smaller subset with NUM_SAMPLES to make it even faster
    dataset = load_dataset(
        "lj_speech",
        split=f"train[:{NUM_SAMPLES}]",
        trust_remote_code=True
    )

    print(f"Successfully loaded dataset with {len(dataset)} samples")

    # Initialize the LLM (optional)
    llm = init_llm()

    # Convert to Ultravox format with 'audio' and 'continuation' fields
    def process_sample(example):
        # If we have an LLM, create a better continuation
        if llm and example["text"]:
            prompt = f"""Below is a transcript from an audio clip:
"{example["text"]}"

Please respond in a natural way to this message, as if you were in a conversation:"""

            try:
                response = llm(prompt, max_new_tokens=100)[0]["generated_text"]
                # Extract just the response part
                continuation = response.split(prompt)[-1].strip()
            except Exception as e:
                print(f"Error generating continuation: {e}")
                continuation = example["text"]
        else:
            continuation = example["text"]

        return {
            "audio": example["audio"],
            "continuation": continuation
        }

    print("Processing dataset samples...")
    # Process the dataset
    processed_dataset = dataset.map(
        process_sample,
        remove_columns=dataset.column_names,
        desc="Processing samples"
    )

    # Set the format for audio processing
    processed_dataset = processed_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # Save the processed dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed_dataset.save_to_disk(OUTPUT_DIR)

    # Save a metadata file with dataset information
    metadata = {
        "source": "lj_speech",
        "num_samples": len(processed_dataset),
        "sample_rate": SAMPLE_RATE,
        "ultravox_compatible": True
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset processed and saved to {OUTPUT_DIR}")
    print(f"Total samples: {len(processed_dataset)}")


if __name__ == "__main__":
    main()