from collections import Counter

from datasets import load_dataset


def show_example(example: dict) -> None:
    print("ID:      ", example["id"])
    print("Question:", example["question"])
    print("Answer:  ", example["answer"])
    print("Type:    ", example["type"])

    print("\nSupporting facts (title, sentence index):")
    facts = example["supporting_facts"]
    for title, sent_id in zip(facts["title"], facts["sent_id"]):
        print(f"  - {title} [{sent_id}]")

    print("\nContext paragraphs:")
    context = example["context"]
    for title, sentences in zip(context["title"], context["sentences"]):
        marker = "*" if title in facts["title"] else " "
        print(f" {marker} {title}: {' '.join(sentences)[:100]}...")


def show_stats(dataset) -> None:
    total_paragraphs = 0
    unique_titles = set()
    for example in dataset:
        titles = example["context"]["title"]
        total_paragraphs += len(titles)
        unique_titles.update(titles)

    print("Questions:          ", len(dataset))
    print("Paragraphs (total): ", total_paragraphs)
    print("Paragraphs (unique):", len(unique_titles))
    print("Duplicates:         ", total_paragraphs - len(unique_titles))
    print("Question types:     ", dict(Counter(dataset["type"])))


if __name__ == "__main__":
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    print("=" * 60, "\nONE EXAMPLE\n" + "=" * 60)
    show_example(dataset[5])

    print("\n" + "=" * 60, "\nSTATS\n" + "=" * 60)
    show_stats(dataset)
