# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        diagnosis = record['diagnosis']
        label = record['class']
        file_path = record['file_path']

        # Tokenize the text
        tokens = self.tokenizer(
            diagnosis,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return tokens, label, file_path


# Initialize the tokenizer and model
def create_dataloader(json_path, batch_size=16):
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the dataset and dataloader
    dataset = DiagonisisTextDataset(json_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Example usage
if __name__ == "__main__":
    json_path = "./RA_hand_biobert.json"  # Replace with your JSON file path
    dataloader = create_dataloader(json_path)

    for batch in dataloader:
        inputs, labels, file_paths = batch
        print("Inputs:", inputs)
        print("Labels:", labels)
        print("File paths:", file_paths)
        break
