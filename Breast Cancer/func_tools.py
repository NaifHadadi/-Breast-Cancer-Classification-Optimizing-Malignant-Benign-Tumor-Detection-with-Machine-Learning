#Plot Random Samples
import matplotlib.pyplot as plt
import random
import torch

#loader function
from torch.utils.data import DataLoader
from torchvision import transforms

from typing import List, Optional
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


from tqdm.auto import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"


class TabularDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def create_dataloaders(
    X_train, y_train,
    X_test, y_test,
    batch_size=32,
    train_shuffle=True,
    test_shuffle=False,
    scale=True
):
    """
    Create train and test dataloaders from pre-split tabular data.
    
    Args:
        X_train, y_train: Training features and labels as NumPy arrays
        X_test, y_test: Testing features and labels as NumPy arrays
        batch_size: Batch size for DataLoaders (default=32)
        train_shuffle: Shuffle training data (default=True)
        test_shuffle: Shuffle test data (default=False)
        scale: Whether to standardize features (default=True)
    
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Convert to float32 for PyTorch
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    to_tensor = lambda x: torch.tensor(x)
    
    train_dataset = TabularDataset(X_train, y_train, transform=to_tensor)
    test_dataset = TabularDataset(X_test, y_test, transform=to_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)
    
    return train_loader, test_loader





# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100  # Percentage accuracy



def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time




#Training Step
device = "cuda" if torch.cuda.is_available() else "cpu"
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

#Testing Step
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    all_preds = []
    all_labels = []

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

            preds = test_pred.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
    return all_preds, all_labels

#Model Evaluation
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}
    


import json
import os

def handle_metrics(final_metrics=None, mode="save", filename="models.json"):
    if mode == "save":
        data = []
        if os.path.exists(filename):
            with open(filename) as f:
                data = json.load(f)
        data.append(final_metrics)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print("Saved metrics.")

    elif mode == "display":
        if not os.path.exists(filename):
            print("No metrics saved yet.")
            return
        with open(filename) as f:
            data = json.load(f)
        # Print table header
        print(f"{'Model Name':20} {'Loss':>10} {'Accuracy':>10}")
        print("-" * 45)
        for m in data:
            print(f"{m['model_name']:20} {m['model_loss']:10.4f} {m['model_acc']:10.2f}")
