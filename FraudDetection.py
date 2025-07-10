import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load dataset
data = pd.read_csv('creditcard.csv')
print(data.info())
print(data.describe())
print(data['Class'].value_counts())

# Preprocessing
X = data.drop('Class', axis=1)
y = data['Class']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Define GNN Model
class FraudDetectionGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudDetectionGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Split Data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize Model
model = FraudDetectionGNN(in_channels=X_train.shape[1], hidden_channels=16, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Convert X_train to NumPy
X_train_np = np.array(X_train)

# Generate Edges using Nearest Neighbors
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')  
knn.fit(X_train_np)
edges = knn.kneighbors_graph(X_train_np, mode='connectivity')

# Convert edges to PyTorch tensor
edge_index_train = torch.tensor(np.array(edges.nonzero()), dtype=torch.long)

# Training Loop
epochs = 20  
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(torch.tensor(X_train, dtype=torch.float32), edge_index_train)  
    loss = criterion(out, torch.tensor(y_train, dtype=torch.long))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Testing Preparation
model.eval()
X_test_np = np.array(X_test)

# Generate Test Edges using Nearest Neighbors
knn_test = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_test.fit(X_test_np)
edges_test = knn_test.kneighbors_graph(X_test_np, mode='connectivity')

import torch_geometric.utils as pyg_utils

edge_index_test = torch.tensor(np.array(edges_test.nonzero()), dtype=torch.long)

if edge_index_test.shape[0] != 2:
    edge_index_test = edge_index_test.t()

edge_index_test, _ = pyg_utils.remove_self_loops(edge_index_test)

edge_index_test = edge_index_test.to(torch.long)

print(f"X_test_tensor shape: {X_test_np.shape}")
print(f"edge_index_test shape (before self-loops): {edge_index_test.shape}")

if edge_index_test.shape[1] == 0:
    raise ValueError("edge_index_test has no edges! Increase `n_neighbors` in NearestNeighbors.")

if edge_index_test.shape[0] == 2:
    edge_index_test, _ = pyg_utils.add_remaining_self_loops(edge_index_test)
else:
    print("Error: edge_index_test still has incorrect shape after transposing!")

print(f"edge_index_test shape (after self-loops): {edge_index_test.shape}")

# Convert X_test to Tensor
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

model.eval()
with torch.no_grad():
    out = model(X_test_tensor, edge_index_test)

# Normalize activations for visualization
with torch.no_grad():  
    activations = model.conv1(X_test_tensor, edge_index_test)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
activations_np = scaler.fit_transform(activations.cpu().numpy())

# Plot Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(activations_np, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title('Normalized Activation Heatmap')
plt.show()

# Compute ROC Curve
with torch.no_grad():
    probs = torch.nn.functional.softmax(out, dim=1)[:, 1].numpy()
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Get Predictions
y_pred = out.argmax(dim=1).numpy()

# Compute Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, probs)

# Print Results
print(f"Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Convert predictions and actual values to NumPy arrays
y_pred_np = np.array(y_pred)
y_test_np = np.array(y_test)

# Get indices where fraud (Class=1) was predicted
fraud_indices = np.where(y_pred_np == 1)[0]  # Indices where model predicted fraud

# Get corresponding transactions from X_test
fraud_transactions = pd.DataFrame(X_test_np[fraud_indices], columns=X.columns)

# Add actual and predicted labels to compare
fraud_transactions['Actual Class'] = y_test_np[fraud_indices]
fraud_transactions['Predicted Class'] = y_pred_np[fraud_indices]

# Display detected fraudulent transactions
print("\nFraudulent Transactions Detected by Model")
print(fraud_transactions.head(10))  # Display first 10 fraud cases

# Count correct fraud detections
true_frauds = (fraud_transactions["Actual Class"] == 1).sum()
false_frauds = (fraud_transactions["Actual Class"] == 0).sum()

print(f"\nTrue Fraud Detections: {true_frauds}")
print(f"False Fraud Detections (False Positives): {false_frauds}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_np)

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import precision_recall_curve

# Compute Precision-Recall values
precision, recall, _ = precision_recall_curve(y_test_np, probs)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', color='purple', label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Compute the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Count true frauds and non-frauds
true_fraud = (y_pred_np == 1).sum()  # Transactions predicted as fraud
true_non_fraud = (y_pred_np == 0).sum()  # Transactions predicted as non-fraud

# Create DataFrame for Seaborn

fraud_data = pd.DataFrame({
    "Transaction Type": ["Non-Fraud", "Fraud"],
    "Count": [true_non_fraud, true_fraud]
})

# Create bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=fraud_data, x="Transaction Type", y="Count", hue="Transaction Type", palette={"Non-Fraud": "blue", "Fraud": "red"}, legend=False)

# Add labels and title
plt.ylabel("Number of Transactions")
plt.xlabel("Transaction Type")
plt.title("Fraud vs Non-Fraud Transactions Detected by Model")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show values on bars
for i, val in enumerate([true_non_fraud, true_fraud]):
    plt.text(i, val + 200, str(val), ha='center', fontsize=12)

# Show plot
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=y_test_np, y=X_test_np[:, -1], hue=y_test_np, palette=["blue", "red"], legend=False)
plt.xlabel("Transaction Type (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.title("Transaction Amount Distribution by Fraud Type")
plt.grid(True)
plt.show()
