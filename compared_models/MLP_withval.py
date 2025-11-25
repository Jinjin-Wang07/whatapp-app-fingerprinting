import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128), ## 512 changed to 128
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output layer (number of classes)
        )

    def forward(self, x):
        return self.model(x)

def mlp_train_with_val(device, X_train, Y_train, X_val, Y_val, num_classes, epochs=300):   
    
    features = torch.tensor(X_train, dtype=torch.float32).to(device)
    lbl = torch.tensor(Y_train, dtype=torch.long).to(device)
    
    # Create TensorDataset and DataLoader
    dataset = TensorDataset(features, lbl)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Model parameters
    input_size = len(X_train[1])
    model = MLPModel(input_size, num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_model_state = None
    
    # Training loop
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        for inputs, labels in batch_pbar:
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        val_loss, val_accuracy = mlp_validate(device, model, X_val, Y_val)

        # Update epoch progress bar with validation metrics
        epoch_pbar.set_postfix({
            'train_loss': f'{epoch_loss/len(dataloader):.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_accuracy:.4f}'
        })        

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            epoch_pbar.set_description("Training (Early Stopped)")
            tqdm.write("No further improvement, early stopping triggered")
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def mlp_validate(device, model, X_val, Y_val):
    features = torch.tensor(X_val, dtype=torch.float32).to(device)
    lbl = torch.tensor(Y_val, dtype=torch.long).to(device)
    
    dataset = TensorDataset(features, lbl)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient for validation
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def mlp_evaluation(device, model, X_test, Y_test, label_names=None, rtn_mtx=False):
    model.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
    
    # Convert to NumPy array
    y_pred = np.array(y_pred)
    
    labels = None
    if label_names:
        labels=list(range(len(label_names)))
    # Compute metrics
    accuracy = accuracy_score(Y_test, y_pred)
    report_dict = classification_report(Y_test, y_pred, labels=labels, target_names=label_names, output_dict=True, digits=4)
    
    precision = report_dict["weighted avg"]["precision"]
    recall = report_dict["weighted avg"]["recall"]
    f1_score = report_dict["weighted avg"]["f1-score"]
   

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(Y_test, y_pred, labels=labels, target_names=label_names, digits=4))

    # Compute confusion matrix
    cm = confusion_matrix(Y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names) if label_names else pd.DataFrame(cm)

    if rtn_mtx:
        return report_dict, cm_df

    return report_dict
    # return cm_df
    # return accuracy, precision, recall, f1_score, cm_df

    
def mlp_clf(X_train, Y_train, X_val, Y_val, X_test, Y_test, num_classes, label_names=None, epochs=300, get_model=False):
    """
    Train MLP model and evaluate performance

    Args:
    - X_train: DataFrame of size N x M where N is number of samples and M is number of features
    - Y_train: DataSeries of size N
    
    """
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    Y_train = Y_train.values
    Y_val = Y_val.values
    Y_test = Y_test.values

    model = mlp_train_with_val(device, X_train_scaled, Y_train, X_val_scaled, Y_val, num_classes, epochs=epochs)
    if get_model:
        return scaler, model
    return mlp_evaluation(device, model, X_test_scaled, Y_test, label_names)

# def mlp_test_single_class(scaler, model, ):
