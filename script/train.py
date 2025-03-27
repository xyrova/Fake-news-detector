import os
import argparse
import tarfile
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ✅ Ensure script is in the right directory
if os.path.exists("/opt/ml/code/"):
    os.chdir("/opt/ml/code/")  
    print("✅ Changed working directory to:", os.getcwd())

# ✅ Debugging: Print available files
print("📂 Files in /opt/ml/code/:", os.listdir("/opt/ml/code/"))

# Custom Dataset class
class FakeNewsDataset(Dataset):
    def __init__(self, data, max_len=128):
        self.data = data
        self.max_len = max_len
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["text"]
        label = self.data.iloc[index]["label"]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0), torch.tensor(label)

# Model class
class FakeNewsClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(FakeNewsClassifier, self).__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ SageMaker auto-mounts S3 data in /opt/ml/input/data/
    train_data_path = os.path.join("/opt/ml/input/data/train", "train.csv")
    test_data_path = os.path.join("/opt/ml/input/data/test", "test.csv")

    print(f"🔍 Looking for training data at: {train_data_path}")
    print(f"🔍 Looking for testing data at: {test_data_path}")

    # ✅ Ensure the dataset exists before loading
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError("❌ Training/Test data not found in expected SageMaker input directories!")

    # ✅ Load data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    train_data.dropna(inplace=True)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_dataset = FakeNewsDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = FakeNewsDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = FakeNewsClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            batch_inputs, batch_masks, batch_labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(batch_inputs, attention_mask=batch_masks)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        print(f"✅ Epoch {epoch+1}/{args.epochs}, Training Loss: {total_train_loss/len(train_loader):.4f}")

    # ✅ Save the trained model
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved at {model_path}")

    # ✅ Compress model to .tar.gz for SageMaker deployment
    tar_path = os.path.join(args.model_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="model.pth")

    print(f"✅ Model compressed and saved at {tar_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)

    args = parser.parse_args()
    train(args)
