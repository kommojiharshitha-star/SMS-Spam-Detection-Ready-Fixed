import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("spam.csv", encoding="latin-1")

if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

elif "label" in df.columns and "message" in df.columns:
    df = df[["label", "message"]]

else:
    raise ValueError("Dataset format not recognized.")

df.dropna(inplace=True)

print("Dataset shape:", df.shape)


# ==========================================
# 2. Convert Labels
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df.dropna(inplace=True)
df["label"] = df["label"].astype(int)

X = df["message"].astype(str)
y = df["label"]


# ==========================================
# 3. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. BERT Tokenizer
# ==========================================

MODEL_NAME = "bert-base-uncased"

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ==========================================
# 5. Dataset Class
# ==========================================

class SMSDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=128):

        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


train_dataset = SMSDataset(
    X_train,
    y_train,
    tokenizer
)

test_dataset = SMSDataset(
    X_test,
    y_test,
    tokenizer
)


# ==========================================
# 6. DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# ==========================================
# 7. Load BERT Model
# ==========================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# ==========================================
# 8. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model.to(device)


# ==========================================
# 9. Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)


# ==========================================
# 10. Training
# ==========================================

EPOCHS = 2

print("\nStarting BERT training...")

model.train()

for epoch in range(EPOCHS):

    total_loss = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for batch_index, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 50 == 0:

            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1} Average Loss: "
        f"{average_loss:.4f}"
    )


# ==========================================
# 11. Evaluation
# ==========================================

print("\nEvaluating BERT...")

model.eval()

predictions = []
actual_labels = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predicted = torch.argmax(
            outputs.logits,
            dim=1
        )

        predictions.extend(
            predicted.cpu().numpy()
        )

        actual_labels.extend(
            labels.cpu().numpy()
        )


# ==========================================
# 12. Metrics
# ==========================================

accuracy = accuracy_score(
    actual_labels,
    predictions
)

precision = precision_score(
    actual_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    predictions,
    zero_division=0
)


print("\n================================")
print("BERT MODEL RESULTS")
print("================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        actual_labels,
        predictions
    )
)

print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predictions,
        target_names=["ham", "spam"],
        zero_division=0
    )
)


# ==========================================
# 13. Save Model
# ==========================================

model.save_pretrained(
    "bert_model"
)

tokenizer.save_pretrained(
    "bert_model"
)

print("\nBERT model saved successfully!")

print("Folder created:")
print("- bert_model")