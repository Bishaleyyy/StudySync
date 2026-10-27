"""
train.py
---------
Trains the StudySync DNN, evaluates it on the test set, and saves:
    - model/studysync_model.pth   (trained weights)
    - model/scaler_mean.npy, model/scaler_std.npy  (for scaling new inputs)
    - outputs/loss_curve.png      (training loss over epochs)
    - outputs/confusion_matrix.png (seaborn heatmap of predictions)

Run:
    python train.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import prepare_data, PRIORITY_MAP
from model import StudySyncDNN

# ---- 1. Load & preprocess data ------------------------------------------
X_train, X_test, y_train, y_test, mean, std = prepare_data("data/study_data.csv")

# ---- 2. Create model, loss function, optimizer ---------------------------
model = StudySyncDNN(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()          # combines softmax + NLL loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 150
loss_history = []

# ---- 3. Training loop ------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()               # clear old gradients
    outputs = model(X_train)            # forward pass
    loss = criterion(outputs, y_train)  # compute loss
    loss.backward()                     # backpropagation
    optimizer.step()                    # update weights

    loss_history.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {loss.item():.4f}")

# ---- 4. Evaluation on test set --------------------------------------------
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predicted_classes = torch.argmax(test_outputs, dim=1)
    accuracy = (predicted_classes == y_test).float().mean().item()

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("NOTE: This dataset is a small, synthetic prototype dataset created "
      "for academic demonstration purposes. Accuracy on real student data "
      "would need to be validated separately.")

# ---- 5. Save the trained model + scaler values ----------------------------
torch.save(model.state_dict(), "model/studysync_model.pth")
np.save("model/scaler_mean.npy", mean)
np.save("model/scaler_std.npy", std)
print("\nModel saved to model/studysync_model.pth")

# ---- 6. Plot training loss curve (Matplotlib) ------------------------------
plt.figure(figsize=(7, 4))
plt.plot(loss_history, color="steelblue")
plt.title("StudySync DNN - Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/loss_curve.png")
plt.close()
print("Saved outputs/loss_curve.png")

# ---- 7. Confusion matrix (Seaborn) -----------------------------------------
labels = ["Low", "Medium", "High"]
num_classes = len(labels)
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true_label, pred_label in zip(y_test.numpy(), predicted_classes.numpy()):
    conf_matrix[true_label][pred_label] += 1

plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Priority Level Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("Saved outputs/confusion_matrix.png")
