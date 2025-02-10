import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # For progress bar
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device="cpu"):
        """
        Initializes the Trainer class.

        Args:
            model (nn.Module): The MM-STLF model.
            train_loader (DataLoader): DataLoader for training set.
            val_loader (DataLoader): DataLoader for validation set.
            test_loader (DataLoader): DataLoader for test set.
            criterion (function): Loss function.
            optimizer (torch.optim): Optimizer for training.
            device (str): "cpu" or "cuda".
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader  # New: Validation set
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs=50, save_path="model.pth"):
        """
        Trains the model.

        Args:
            num_epochs (int): Number of training epochs.
            save_path (str): Path to save the best model.

        Returns:
            None
        """
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            running_train_loss = 0.0

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
            for batch_X, batch_y in loop:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)

                # Compute loss
                loss = self.criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()
                loop.set_postfix(train_loss=loss.item())

            avg_train_loss = running_train_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

            # Validation step
            avg_val_loss = self.evaluate(self.val_loader)

            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(save_path)
                print(f"New best model saved at {save_path}")

    def evaluate(self, data_loader):
        """
        Evaluates the model on a given dataset (validation or test).

        Args:
            data_loader (DataLoader): DataLoader for the dataset.

        Returns:
            float: Average loss on the dataset.
        """
        self.model.eval()  # Set model to evaluation mode
        running_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        return avg_loss

    def test(self):
        """
        Evaluates the model on the test set.
        """
        print("\nEvaluating on Test Set:")
        test_loss = self.evaluate(self.test_loader)
        print(f"Final Test Loss: {test_loss:.7f}")

    def save_model(self, save_path):
        """Saves the model state dictionary."""
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        """Loads the model state dictionary."""
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            print(f"Model loaded from {load_path}")
        else:
            print(f"Model file not found: {load_path}")
