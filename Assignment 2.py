# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:30:46 2024

@author: barab
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv(r"C:\Users\barab\OneDrive\Documents\McGill MMA\Courses\MGSC 673\churn.csv")

# Drop unnecessary columns
data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Geography', 'Gender'])

# Split features and target variable
X = data.drop(columns=['Exited'])
y = data['Exited']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Define custom dataset and sampler for oversampling
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define model architecture
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Define a function to train and evaluate the model
def train_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Define a function for hyperparameter tuning
def hyperparameter_tuning(train_data, test_data):
    param_grid = {
        'hidden_size1': [128, 256],
        'hidden_size2': [64, 128],
        'learning_rate': [0.001, 0.01],
        'num_epochs': [50, 100]
    }
    best_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    best_params = None
    for params in ParameterGrid(param_grid):
        model = FNN(input_size=X_train.shape[1], hidden_size1=params['hidden_size1'], hidden_size2=params['hidden_size2'], output_size=2)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64)
        train_evaluate_model(model, train_loader, test_loader, criterion, optimizer, params['num_epochs'])
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        if f1 > best_metrics['f1']:
            best_metrics['accuracy'] = accuracy
            best_metrics['precision'] = precision
            best_metrics['recall'] = recall
            best_metrics['f1'] = f1
            best_params = params
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Metrics: {best_metrics}")

# Main function
def main():
    train_data = CustomDataset(X_train, y_train)
    test_data = CustomDataset(X_test, y_test)

    # Oversample minority class in training data using WeightedRandomSampler
    class_counts = np.bincount(y_train)
    class_weights = 1 / torch.tensor(class_counts, dtype=torch.float32)
    samples_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=64)

    hyperparameter_tuning(train_data, test_data)

if __name__ == "__main__":
    main()
