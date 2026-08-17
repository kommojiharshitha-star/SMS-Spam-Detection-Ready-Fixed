import pandas as pd
import matplotlib.pyplot as plt


# Model performance results
results = {
    "Model": [
        "BiLSTM",
        "DistilBERT",
        "BERT"
    ],
    "Accuracy": [
        0.9865,
        0.9937,
        0.9919
    ],
    "Precision": [
        0.9855,
        0.9931,
        0.9730
    ],
    "Recall": [
        0.9128,
        0.9597,
        0.9664
    ],
    "F1-Score": [
        0.9477,
        0.9761,
        0.9697
    ]
}


# Create DataFrame
df = pd.DataFrame(results)

# Convert values to percentages
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

for metric in metrics:
    df[metric] = df[metric] * 100


# Display comparison
print("\n======================================")
print("MODEL PERFORMANCE COMPARISON")
print("======================================")

print(
    df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2f}%".format,
            "Precision": "{:.2f}%".format,
            "Recall": "{:.2f}%".format,
            "F1-Score": "{:.2f}%".format
        }
    )
)


# Find best model based on F1-score
best_model = df.loc[
    df["F1-Score"].idxmax(),
    "Model"
]

best_f1 = df["F1-Score"].max()


print("\n======================================")
print("BEST MODEL")
print("======================================")

print(f"Best Model : {best_model}")
print(f"F1-Score   : {best_f1:.2f}%")


# Create comparison chart
df_plot = df.set_index("Model")

df_plot[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("SMS Spam Detection Model Comparison")
plt.xlabel("Models")
plt.ylabel("Performance (%)")
plt.ylim(80, 100)
plt.legend()
plt.tight_layout()

plt.savefig("model_comparison.png")

plt.show()


# Save comparison results
df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nFiles created successfully:")
print("- model_comparison.csv")
print("- model_comparison.png")