import ollama
import csv
import pandas as pd

TOPICS_FILE = "topics.csv"         
OUTPUT_FILE = "argument_outputs.csv"

MODELS = ["llama3.1", "gemma"]


def load_topics(topics_file, limit=None):
    """Load unique topics from the topics CSV."""
    df = pd.read_csv(topics_file)
    topics = df["topic"].unique().tolist()
    if limit is not None:
        topics = topics[:limit]
    return topics


def get_argument(model, topic):
    """Ask a single model to argue in favor of a topic."""
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Write a short 2-3 sentence argument supporting: {topic}",
            }
        ],
    )
    return response["message"]["content"].strip()


def main():
    topics = load_topics(TOPICS_FILE)
    print(f"Loaded {len(topics)} topics. Generating arguments from {len(MODELS)} models...")

    # Long-format rows: one row per (topic, model) pair. This scales cleanly
    # to any number of models without hand-editing the header each time.
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "topic", "argument"])

        for i, topic in enumerate(topics, start=1):
            print(f"[{i}/{len(topics)}] {topic}")
            for model in MODELS:
                try:
                    argument = get_argument(model, topic)
                except Exception as e:
                    print(f"  ! {model} failed on '{topic}': {e}")
                    argument = f"ERROR: {e}"
                writer.writerow([model, topic, argument])

    print(f"\nDone. Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()