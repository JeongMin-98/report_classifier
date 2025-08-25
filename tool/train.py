# -----------------------------------------------------------
#
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
# Add 'lib' directory to Python path
import os
import sys
import logging
import time
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

# 상단의 import 문에 추가
from fvcore.nn import FlopCountAnalysis
from typing import Dict, Any

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


def log_gpu_memory_usage(stage, logger=None):
    """Log GPU memory usage at a specific stage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # Convert to GB
        msg = f"{stage} - GPU Memory Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    else:
        msg = f"{stage} - No GPU detected."
        if logger:
            logger.info(msg)
        else:
            print(msg)

def calculate_flops(model, tokenizer, logger):
    """Calculate FLOPs for the model."""
    # 샘플 입력 생성
    sample_text = "This is a sample text for FLOP calculation"
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 입력을 모델의 디바이스로 이동
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # FLOPs 계산
    flops = FlopCountAnalysis(
        model,
        (inputs["input_ids"], 
         inputs["attention_mask"])
    )
    total_flops = flops.total()
    
    # GFLOPs로 변환
    gflops = total_flops / (10**9)
    
    logger.info(f"Total FLOPs: {total_flops:,}")
    logger.info(f"GFLOPs: {gflops:.2f}")
    
    return total_flops, gflops

def calculate_detailed_params(model, logger):
    """Calculate detailed parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory_footprint = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2  # MB

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Memory footprint: {memory_footprint:.2f} MB")
    
    return total_params, trainable_params, memory_footprint


def calculate_performance_metrics(model, dataloader, criterion, device, logger):
    """Calculate various performance metrics."""
    model.eval()
    total_time = 0
    total_samples = 0
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        start_time = time.time()
        for inputs, labels in dataloader:
            # inputs는 dictionary이므로 각 키에 접근
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            # 추론 시간 측정
            batch_start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_time = time.time() - batch_start
            
            loss = criterion(outputs.logits, labels)
            
            # 정확도 계산
            predictions = (outputs.logits > 0).float()
            correct += (predictions == labels).float().sum().item()
            
            total_time += batch_time
            total_samples += input_ids.size(0)
            total_loss += loss.item()
    
    # 메트릭 계산
    accuracy = correct / (total_samples * labels.size(1))  # 멀티라벨의 경우
    avg_inference_time = total_time / total_samples
    throughput = total_samples / total_time
    avg_loss = total_loss / len(dataloader)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Average Inference Time per sample: {avg_inference_time*1000:.2f} ms")
    logger.info(f"Throughput: {throughput:.2f} samples/second")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    
    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time,
        'throughput': throughput,
        'avg_loss': avg_loss,
        'total_time': total_time  # Add this line
    }
    

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

    train_dataloader, val_dataloader, _ = split_dataset_parallel(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        num_workers=cfg.WORKERS
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train with LoRA
    logger.info("\nTraining with LoRA...")
    model_lora = initialize_model_with_lora(cfg, tokenizer, use_lora=True)
    model_lora.to(device)

    # LoRA 모델 분석
    logger.info("\nAnalyzing LoRA model...")
    logger.info("FLOPs calculation:")
    flops_lora, gflops_lora = calculate_flops(model_lora, tokenizer, logger)
    
    logger.info("\nParameter statistics for LoRA model:")
    total_params_lora, trainable_params_lora, memory_lora = calculate_detailed_params(model_lora, logger)

    optimizer_lora = torch.optim.Adam(filter(lambda p: p.requires_grad, model_lora.parameters()), lr=cfg.TRAIN.LR)
    criterion = BCEWithLogitsLoss()

    # LoRA 모델 초기 성능 측정
    logger.info("\nMeasuring initial LoRA model performance...")
    initial_lora_metrics = calculate_performance_metrics(
        model_lora, val_dataloader, criterion, device, logger
    )

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

    # Log initial GPU memory usage
    log_gpu_memory_usage("Before LoRA Training", logger)

    # LoRA 모델 훈련 및 시간 측정
    logger.info("\nStarting LoRA model training...")
    start_time = time.time()
    trainer_lora.train(cfg.TRAIN.NUM_EPOCHS)
    total_time_lora = time.time() - start_time
    avg_epoch_time_lora = total_time_lora / cfg.TRAIN.NUM_EPOCHS

    # Log final GPU memory usage
    log_gpu_memory_usage("After LoRA Training", logger)

    # LoRA 모델 최종 성능 측정
    logger.info("\nMeasuring final LoRA model performance...")
    final_lora_metrics = calculate_performance_metrics(
        model_lora, val_dataloader, criterion, device, logger
    )

    # Train without LoRA
    logger.info("\nTraining without LoRA...")
    model_no_lora = initialize_model_with_lora(cfg, tokenizer, use_lora=False)
    model_no_lora.to(device)

    # Non-LoRA 모델 분석
    logger.info("\nAnalyzing non-LoRA model...")
    logger.info("FLOPs calculation:")
    flops_no_lora, gflops_no_lora = calculate_flops(model_no_lora, tokenizer, logger)
    
    logger.info("\nParameter statistics for non-LoRA model:")
    total_params_no_lora, trainable_params_no_lora, memory_no_lora = calculate_detailed_params(model_no_lora, logger)

    # Non-LoRA 모델 초기 성능 측정
    logger.info("\nMeasuring initial non-LoRA model performance...")
    initial_no_lora_metrics = calculate_performance_metrics(
        model_no_lora, val_dataloader, criterion, device, logger
    )

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
    
    # Log initial GPU memory usage
    log_gpu_memory_usage("Before non-LoRA Training", logger)
    
    # Non-LoRA 모델 훈련 및 시간 측정
    logger.info("\nStarting non-LoRA model training...")
    start_time = time.time()
    trainer_no_lora.train(cfg.TRAIN.NUM_EPOCHS)
    total_time_no_lora = time.time() - start_time
    avg_epoch_time_no_lora = total_time_no_lora / cfg.TRAIN.NUM_EPOCHS
    
    # Log final GPU memory usage
    log_gpu_memory_usage("After non-LoRA Training", logger)

    # Non-LoRA 모델 최종 성능 측정
    logger.info("\nMeasuring final non-LoRA model performance...")
    final_no_lora_metrics = calculate_performance_metrics(
        model_no_lora, val_dataloader, criterion, device, logger
    )

    # 최종 비교 결과 출력
    logger.info("\n" + "="*50)
    logger.info("Final Comprehensive Comparison")
    logger.info("="*50)
    
    # 훈련 시간 비교
    logger.info("\nTraining Time Comparison:")
    logger.info("LoRA Model:")
    logger.info(f"- Total Training Time: {total_time_lora:.2f} seconds")
    logger.info(f"- Average Time per Epoch: {avg_epoch_time_lora:.2f} seconds")
    
    logger.info("\nNon-LoRA Model:")
    logger.info(f"- Total Training Time: {total_time_no_lora:.2f} seconds")
    logger.info(f"- Average Time per Epoch: {avg_epoch_time_no_lora:.2f} seconds")
    
    training_time_reduction = ((total_time_no_lora - total_time_lora) / total_time_no_lora) * 100
    logger.info(f"\nTraining Time Reduction with LoRA: {training_time_reduction:.2f}%")
    
    # 리소스 사용 비교
    logger.info("\nResource Usage:")
    logger.info("\nLoRA Model:")
    logger.info(f"- GFLOPs: {gflops_lora:.2f}")
    logger.info(f"- Total Parameters: {total_params_lora:,}")
    logger.info(f"- Trainable Parameters: {trainable_params_lora:,}")
    logger.info(f"- Memory Footprint: {memory_lora:.2f} MB")
    
    logger.info("\nNon-LoRA Model:")
    logger.info(f"- GFLOPs: {gflops_no_lora:.2f}")
    logger.info(f"- Total Parameters: {total_params_no_lora:,}")
    logger.info(f"- Trainable Parameters: {trainable_params_no_lora:,}")
    logger.info(f"- Memory Footprint: {memory_no_lora:.2f} MB")
    
    # 성능 메트릭 비교
    logger.info("\nPerformance Metrics:")
    logger.info("\nLoRA Model:")
    logger.info("Initial Performance:")
    logger.info(f"- Accuracy: {initial_lora_metrics['accuracy']:.4f}")
    logger.info(f"- Inference Time: {initial_lora_metrics['avg_inference_time']*1000:.2f} ms/sample")
    logger.info(f"- Throughput: {initial_lora_metrics['throughput']:.2f} samples/sec")
    logger.info("Final Performance:")
    logger.info(f"- Accuracy: {final_lora_metrics['accuracy']:.4f}")
    logger.info(f"- Inference Time: {final_lora_metrics['avg_inference_time']*1000:.2f} ms/sample")
    logger.info(f"- Throughput: {final_lora_metrics['throughput']:.2f} samples/sec")
    
    logger.info("\nNon-LoRA Model:")
    logger.info("Initial Performance:")
    logger.info(f"- Accuracy: {initial_no_lora_metrics['accuracy']:.4f}")
    logger.info(f"- Inference Time: {initial_no_lora_metrics['avg_inference_time']*1000:.2f} ms/sample")
    logger.info(f"- Throughput: {initial_no_lora_metrics['throughput']:.2f} samples/sec")
    logger.info("Final Performance:")
    logger.info(f"- Accuracy: {final_no_lora_metrics['accuracy']:.4f}")
    logger.info(f"- Inference Time: {final_no_lora_metrics['avg_inference_time']*1000:.2f} ms/sample")
    logger.info(f"- Throughput: {final_no_lora_metrics['throughput']:.2f} samples/sec")
    
    # 최종 성능 차이 계산
    logger.info("\nFinal Performance Differences (LoRA vs non-LoRA):")
    acc_diff = ((final_lora_metrics['accuracy'] - final_no_lora_metrics['accuracy']) / final_no_lora_metrics['accuracy']) * 100
    time_diff = ((final_no_lora_metrics['avg_inference_time'] - final_lora_metrics['avg_inference_time']) / final_no_lora_metrics['avg_inference_time']) * 100
    throughput_diff = ((final_lora_metrics['throughput'] - final_no_lora_metrics['throughput']) / final_no_lora_metrics['throughput']) * 100
    param_reduction = ((trainable_params_no_lora - trainable_params_lora) / trainable_params_no_lora * 100)
    memory_reduction = ((memory_no_lora - memory_lora) / memory_no_lora * 100)
    
    logger.info(f"- Accuracy change: {acc_diff:+.2f}%")
    logger.info(f"- Inference time reduction: {time_diff:+.2f}%")
    logger.info(f"- Throughput improvement: {throughput_diff:+.2f}%")
    logger.info(f"- Trainable parameter reduction: {param_reduction:.2f}%")
    logger.info(f"- Memory footprint reduction: {memory_reduction:.2f}%")
    logger.info("\n" + "="*50)

if __name__ == "__main__":
    main()