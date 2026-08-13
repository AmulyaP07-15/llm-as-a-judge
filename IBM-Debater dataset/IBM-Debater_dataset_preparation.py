from datasets import load_dataset
import csv

DATASET_NAME = "NLP-Debater-Project/IBM-Debater-ArgKP"
TOPICS_OUTPUT_FILE = "topics.csv"


def load_topics(dataset_name=DATASET_NAME):
    """
    downloads the dataset from huggingface and returns the unique topics.

    input: HF dataset name
    output: list of unique topics, first-seen order
    """
    ds = load_dataset(dataset_name, split="train")

    # Preserve first-seen order rather than using set() (which is unordered)
    seen = set()
    topics = []
    for topic in ds["topic"]:
        if topic not in seen:
            seen.add(topic)
            topics.append(topic)
    return topics


def save_topics(topics, output_file=TOPICS_OUTPUT_FILE):
    """
    writes the topic list out to csv.

    input: list of topics, output path
    output: nothing, writes topics.csv
    """
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic"])
        for topic in topics:
            writer.writerow([topic])


def main():
    """
    downloads topics and saves them to csv.

    input: nothing
    output: nothing
    """
    topics = load_topics()
    print(f"Found {len(topics)} unique topics (expected 41).")
    save_topics(topics)
    print(f"Saved topics to {TOPICS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()