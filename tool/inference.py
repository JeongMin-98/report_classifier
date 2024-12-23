import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
# from config import cfg

def load_best_model_with_lora(model, checkpoint_path, device, lora_config):
    """
    Load the best LoRA model checkpoint.
    Args:
        model (nn.Module): The base model to load the checkpoint into.
        optimizer (Optimizer): The optimizer to load the checkpoint into.
        checkpoint_path (str): Path to the best model checkpoint.
        device (torch.device): Device to map the model and optimizer.
        lora_config (LoraConfig): LoRA configuration for the model.
    Returns:
        model (nn.Module): The model with loaded weights.
        optimizer (Optimizer): The optimizer with loaded state.
        int: The epoch number from the checkpoint.
        dict: Additional checkpoint data.
    """
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.Adam(model.parameters())

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model and optimizer states
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Retrieve epoch and additional metadata
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    val_accuracy = checkpoint["val_accuracy"]

    print(f"Loaded LoRA model from {checkpoint_path}")
    print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return model, optimizer, epoch, checkpoint

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and base model
model_name = "dmis-lab/biobert-base-cased-v1.1-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,  # Adjust based on your task
    ignore_mismatched_sizes=True

)

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,  # Scaling factor
    target_modules=["query", "value"],  # Modules to apply LoRA
    lora_dropout=0.1  # Dropout rate for LoRA
)

# Initialize optimizer
# Load best model with LoRA
checkpoint_path = "./checkpoints/best_model.pth.tar"
model, optimizer, epoch, checkpoint = load_best_model_with_lora(
    base_model, checkpoint_path, device, lora_config
)

# Set model to evaluation mode
model.to(device)
model.eval()

# Inference example
# OA
# sample_text = "degenerative change observed in the MTP joint."
# ref.Prev
# sample_text = "No significant interval change since last study"
sample_text = "joint space narrowing, erosions, partial fusion of radiocarpal, ulnocarpal, intercarpal, CMC joints, both \nerosion at Rt 2nd MP base \n-> RA involvement \n"
# Tokenize input
inputs = tokenizer(
    sample_text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Move inputs to device
inputs = {key: val.to(device) for key, val in inputs.items()}

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits)
preds = predictions > 0.7

print(f"Predictions: {predictions}")
print(f"pred : {preds}")