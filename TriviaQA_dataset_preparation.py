from datasets import load_dataset
import pandas as pd

# load the validation split from HuggingFace
dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")
df = pd.DataFrame(dataset)

print(df.shape)
print(df.columns.tolist())


def first_alias(answer):
    """
    a single readable form of the answer, used for display only.
    not reliable for scoring since TriviaQA's first alias is often an odd
    variant, for example C-saw for teeter totter.
    """
    aliases = answer.get("aliases") or []
    return aliases[0] if aliases else None


def all_aliases(answer):
    """
    every acceptable form of the answer, joined with a pipe so it survives
    the CSV round trip. scoring matches against this rather than one alias,
    otherwise correct answers get marked wrong when the model happens to
    phrase it differently from whichever variant came first.

    normalized_aliases are lowercased and stripped of articles, plain
    aliases keep the original casing. both are kept since either can match.
    """
    normalized = answer.get("normalized_aliases") or []
    raw = answer.get("aliases") or []

    seen = []
    for a in list(normalized) + list(raw):
        a = str(a).strip()
        # pipe is the delimiter so an alias containing one would break parsing
        if a and "|" not in a and a.lower() not in [s.lower() for s in seen]:
            seen.append(a)

    return "|".join(seen) if seen else None


df["answer_text"] = df["answer"].apply(first_alias)
df["answer_aliases"] = df["answer"].apply(all_aliases)

# keep only what we need for the experiment
df = df[["question", "question_source", "answer_text", "answer_aliases"]]

# basic cleaning
df = df.dropna()
df = df[df["question"].str.strip() != ""]
df = df[df["answer_text"].str.strip() != ""]
df = df.drop_duplicates(subset="question")

print(df.shape)
print(df["question_source"].value_counts())

# 14 sources, 21 per source gets us to ~294 which is close enough to 300
n_per_source = 300 // df["question_source"].nunique()

df_sampled = (
    df.groupby("question_source", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), n_per_source), random_state=42))
    .reset_index(drop=True)
)

# shuffle so questions aren't grouped by source
df_sampled = (
    df_sampled
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
    .head(300)
)

print(f"Total samples: {len(df_sampled)}")
print(df_sampled.columns.tolist())
print(df_sampled.head())

# quick look at how many alternate forms we picked up, since the whole point
# of this column is that one alias was not enough
alias_counts = df_sampled["answer_aliases"].str.split("|").str.len()
print(f"\nmean aliases per answer: {alias_counts.mean():.1f}")
print(f"answers with only one alias: {(alias_counts == 1).sum()}")

df_sampled.to_csv("TriviaQA_sampled.csv", index=False)
print("\nsaved to TriviaQA_sampled.csv")