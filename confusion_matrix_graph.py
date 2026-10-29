
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import os

# Existing training results
cm = np.array([
    [965, 1],
    [31, 118]
])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

fig, ax = plt.subplots(figsize=(7, 5))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("SMS Spam Detection - Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

plt.tight_layout()

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "confusion_matrix_graph.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Graph saved successfully: {output_path}")

plt.show()
