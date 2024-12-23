# -----------------------------------------------------------
#
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
# Add 'lib' directory to Python path
import os
import sys
import logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set the project root directory and import path initializer
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
import init_lib_paths

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.nn import BCEWithLogitsLoss

from core.trainer import Trainer
from dataset.diagnosis import DiagnosisTextDataset
from utils.tools import balance_dataset, split_dataset_parallel
from config.default import update_config
from config import cfg

# Set up logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train BioBERT+LoRA model for classification")
    parser.add_argument(
        "--cfg",
        default="./experiments/basic.yaml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./RA_hand_biobert.json",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the outputs"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    args = parser.parse_args()
    return args

def initialize_model_with_lora(cfg, tokenizer, use_lora):
    """Initialize the model with or without LoRA."""
    model_name = cfg.MODEL.PRETRAINED

    # Load the base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=cfg.MODEL.NUM_LABELS,
        ignore_mismatched_sizes=True
    )

    if use_lora:
        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=cfg.LORA.RANK,
            lora_alpha=cfg.LORA.ALPHA,
            target_modules=cfg.LORA.TARGET_MODULES,
            lora_dropout=cfg.LORA.DROPOUT
        )
        model = get_peft_model(model, lora_config)

        # Freeze all parameters except LoRA and classifier
        for name, param in model.named_parameters():
            if "lora_" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return model

def count_trainable_parameters(model):
    """Count the number of trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def main():

    args = parse_args()

    # Set up logging
    logger = setup_logging(args.log_dir)

    # Load configuration
    update_config(cfg, args)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.PRETRAINED)

    # Load dataset and dataloader
    json_path = cfg.DATASET.JSON
    dataset = DiagnosisTextDataset(json_path, tokenizer)

    # Balance the dataset (optional)
    balanced_dataset = balance_dataset(dataset)
    train_dataloader, val_dataloader, _ = split_dataset_parallel(
        balanced_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        num_workers=cfg.WORKERS
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train with LoRA
    print("Training with LoRA...")
    model_lora = initialize_model_with_lora(cfg, tokenizer, use_lora=True)
    model_lora.to(device)

    optimizer_lora = torch.optim.Adam(filter(lambda p: p.requires_grad, model_lora.parameters()), lr=cfg.TRAIN.LR)
    criterion = BCEWithLogitsLoss()

    trainer_lora = Trainer(
        model=model_lora,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer_lora,
        cfg=cfg,
        device=device
    )

    trainable_params_lora, total_params_lora = count_trainable_parameters(model_lora)
    logger.info(f"LoRA - Trainable Parameters: {trainable_params_lora}/{total_params_lora}")

    trainer_lora.train(cfg.TRAIN.NUM_EPOCHS)

    # Train without LoRA
    logger.info("\nTraining without LoRA...")
    model_no_lora = initialize_model_with_lora(cfg, tokenizer, use_lora=False)
    model_no_lora.to(device)

    optimizer_no_lora = torch.optim.Adam(model_no_lora.parameters(), lr=cfg.TRAIN.LR)

    trainer_no_lora = Trainer(
        model=model_no_lora,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer_no_lora,
        cfg=cfg,
        device=device
    )

    trainable_params_no_lora, total_params_no_lora = count_trainable_parameters(model_no_lora)
    logger.info(f"No LoRA - Trainable Parameters: {trainable_params_no_lora}/{total_params_no_lora}")

    trainer_no_lora.train(cfg.TRAIN.NUM_EPOCHS)

if __name__ == "__main__":
    main()