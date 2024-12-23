# Using BioBert and LoRA to classify report

### HuggingFace Login
```
huggingface-cli login
```
+ Please log in to Hugging Face and then follow the instructions below.
### Train
```shell
    python tool/train.py --cfg ./experiments/basic.yaml --data_dir ./RA_hand_biobert.json --output_dir ./results --log_dir ./logs --gpus 1 
```

