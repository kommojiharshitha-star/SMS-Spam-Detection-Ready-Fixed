

import os
import pandas as pd
import matplotlib.pyplot as plt

# Existing project folder
folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(folder, "spam.csv")

# Check dataset
if not os.path.exists(csv_path):
    print("ERROR: spam.csv not found in the project folder.")
    raise SystemExit

# Load dataset
df = pd.read_csv(csv_path, encoding="latin-1")

# Find the label column
if "label" in df.columns:
    labels = df["label"]
elif "v1" in df.columns:
    labels = df["v1"]
else:
    print("ERROR: Ham/Spam label column not found.")
    print("Available columns:", list(df.columns))
    raise SystemExit

# Count Ham and Spam messages
labels = labels.astype(str).str.strip().str.lower()
ham_count = (labels == "ham").sum()
spam_count = (labels == "spam").sum()

print("Dataset Message Counts")
print("Ham messages :", ham_count)
print("Spam messages:", spam_count)

# Create graph
plt.figure(figsize=(8, 5))

bars = plt.bar(
    ["Ham", "Spam"],
    [ham_count, spam_count]
)

plt.title("SMS Dataset - Ham vs Spam Message Count")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in bars:
    count = int(bar.get_height())
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Save graph separately
output_path = os.path.join(
    folder,
    "dataset_distribution_graph.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("Graph saved successfully:", output_path)

plt.show()
