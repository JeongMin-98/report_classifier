# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from collections import Counter


# Define the Dataset class
class DiagnosisTextDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=128):
        """
        Args:
            json_file (str): Path to the JSON file.
            tokenizer (transformers.AutoTokenizer): Tokenizer for text preprocessing.
            max_length (int): Maximum sequence length for tokenization.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.data = data["data"]
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Add label mapping: map string labels to integers
        self.class_mapping = {
            "normal": 0,
            "ra": 1,
            "oa": 2,
            "gout": 3,
            "uncertain": 4,
            "ref.prev": 5
        }

        # Reverse mapping for decoding

        self.inverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        self.class_counts = Counter([cls for record in self.data for cls in map(lambda x: x.strip().lower(), record['class'].split(","))])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        diagnosis = record['diagnosis']

        # Multi-label encoding
        raw_classes = list(map(lambda x: x.strip().lower(), record['class'].split(","))) # Assumes list of classes, e.g., ["RA", "OA"]
        label = [0] * len(self.class_mapping)
        for cls in raw_classes:
            label[self.class_mapping[cls]] = 1

        tokens = self.tokenizer(
            diagnosis,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }

        return inputs, torch.tensor(label, dtype=torch.float)

    def decode_label(self, label):
        """
        Decode a numerical label back to its string representation.
        Args:
            label (int): Numerical label to decode.
        Returns:
            str: String representation of the label.
        """
        return self.inverse_class_mapping[label]