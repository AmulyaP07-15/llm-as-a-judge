from datasets import load_dataset
import csv

DATASET_NAME = "NLP-Debater-Project/IBM-Debater-ArgKP"
TOPICS_OUTPUT_FILE = "topics.csv"


def load_topics(dataset_name=DATASET_NAME):
    """Download the dataset from HuggingFace and return the unique topics."""
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
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic"])
        for topic in topics:
            writer.writerow([topic])


def main():
    topics = load_topics()
    print(f"Found {len(topics)} unique topics (expected 41).")
    save_topics(topics)
    print(f"Saved topics to {TOPICS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()