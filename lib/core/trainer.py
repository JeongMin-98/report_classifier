# -----------------------------------------------------------
#
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------

import os
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader, criterion, optimizer, cfg, device, log_dir="logs", save_dir="checkpoints"):
        """
        Trainer class to manage training, validation, logging, and checkpointing.

        Args:
            model (nn.Module): Pretrained model (e.g., BioBERT).
            tokenizer (AutoTokenizer): Pretrained tokenizer.
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Loss function for training.
            optimizer (Optimizer): Optimizer for training.
            cfg (CfgNode): Configuration object.
            device (torch.device): Device to use for training (e.g., 'cuda' or 'cpu').
            log_dir (str): Directory for TensorBoard logs.
            save_dir (str): Directory to save model checkpoints.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device
        self.writer = SummaryWriter(log_dir)  # TensorBoard writer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self, num_epochs):
        """
        Train the model for a given number of epochs.
        Args:
            num_epochs (int): Number of epochs to train.
        """
        logger.info("Starting training...")
        best_pert = -1
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")

            for batch in progress_bar:
                inputs, labels = batch
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
            self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)

            # Validate at the end of each epoch
            val_loss, val_accuracy = self.validate(epoch)
            if best_pert < val_accuracy:
                best_pert = val_accuracy 
                # Save checkpoint
                self.save_checkpoint(epoch, val_loss, val_accuracy)
            
            logger.info(f"Epoch {epoch}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        self.save_checkpoint(self.cfg.TRAIN.NUM_EPOCHS, val_loss, val_accuracy)
            

    def validate(self, epoch):
        """
        Evaluate the model using the validation dataloader.
        Args:
            epoch (int): Current epoch number for logging purposes.
        Returns:
            float: Validation loss.
            float: Validation accuracy.
        """
        logger.info("Starting validation...")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Validating")
            for batch in progress_bar:
                inputs, labels = batch
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()

                # Accuracy calculation
                preds = torch.sigmoid(outputs.logits) > 0.7
                correct += (preds == labels).sum().item()
                total += labels.numel()

                progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct / total
        self.writer.add_scalar("Loss/Validation", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/Validation", accuracy, epoch)

        return avg_loss, accuracy

    def save_checkpoint(self, epoch, val_loss, val_accuracy):
        """
        Save a checkpoint of the model.
        Args:
            epoch (int): Current epoch number.
            val_loss (float): Validation loss.
            val_accuracy (float): Validation accuracy.
        """
        
        if epoch == self.cfg.TRAIN.NUM_EPOCHS:
            checkpoint_path = os.path.join(self.save_dir, "final_model.pth.tar")
        checkpoint_path = os.path.join(self.save_dir, f"best_model.pth.tar")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")

