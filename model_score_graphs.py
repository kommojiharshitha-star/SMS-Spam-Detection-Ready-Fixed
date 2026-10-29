

import matplotlib.pyplot as plt

scores = {
    "Accuracy": 97.13,
    "Precision": 99.16,
    "Recall": 79.19,
    "F1-Score": 88.06
}

plt.figure(figsize=(9, 5))

bars = plt.bar(scores.keys(), scores.values())

plt.title("SMS Spam Detection - Model Performance Scores")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score (%)")
plt.ylim(0, 105)
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar, score in zip(bars, scores.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{score:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("model_score_graph.png", dpi=300)
plt.show()
