import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
import json
from sklearn.metrics import classification_report, f1_score
import argparse
import numpy as np
import os
from pathlib import Path
from datasets import load_dataset
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_data_subfolder(file_path):
    path = Path(file_path)
    # Replace the file extension with empty to effectively create a same-named folder path
    folder_path = path.with_suffix('')
    return str(folder_path)


def save_metrics(metrics, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {file_path}")


class RelationExtractionModel(nn.Module):
    def __init__(self, num_relations):
        super(RelationExtractionModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(args.model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_relations)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RelationExtractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if 'text_short' in item:
            text = item['text_short']
        else:
            text = item['text']
        label = item['_id']  # Use '_id' instead of 'label'

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def add_entity_tokens(tokenizer, dataset_name):
    if dataset_name == '2018' or dataset_name == 'conll04' or dataset_name == 'semeval2010':
        new_tokens = ['<entity>', '</entity>']
    elif dataset_name == 'scierc':
        new_tokens = ['<e1>', '</e1>', '<e2>', '</e2>']
    elif dataset_name == 'ddi':
        new_tokens = ['<drug>', '</drug>']
    elif dataset_name == 'chemprot':
        new_tokens = ['<chemical>', '</chemical>', '<protein>', '</protein>']
    elif dataset_name == '2010':
        new_tokens = ["<problem>", "</problem>", "<treatment>", "</treatment>"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    tokenizer.add_tokens(new_tokens)
    return tokenizer


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = sum([1 for p, t in zip(predictions, true_labels) if p == t]) / len(true_labels)

    # Compute micro F1 and macro F1
    micro_f1 = f1_score(true_labels, predictions, average='micro')
    macro_f1 = f1_score(true_labels, predictions, average='macro')

    return avg_loss, accuracy, micro_f1, macro_f1


def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, num_epochs, patience=3):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    training_metrics = {
        "train_loss": [], "val_loss": [], "val_accuracy": [],
        "val_micro_f1": [], "val_macro_f1": [],
        "test_loss": [], "test_accuracy": [],
        "test_micro_f1": [], "test_macro_f1": []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average train loss: {avg_train_loss:.4f}")

        training_metrics["train_loss"].append(avg_train_loss)

        if val_loader:
            val_loss, val_accuracy, val_micro_f1, val_macro_f1 = evaluate(model, val_loader, device)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            print(f"Validation Micro F1: {val_micro_f1:.4f}, Macro F1: {val_macro_f1:.4f}")

            training_metrics["val_loss"].append(val_loss)
            training_metrics["val_accuracy"].append(val_accuracy)
            training_metrics["val_micro_f1"].append(val_micro_f1)
            training_metrics["val_macro_f1"].append(val_macro_f1)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model = model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                model.load_state_dict(best_model)
                break

        # Evaluate on the test set at the end of each epoch
        test_loss, test_accuracy, test_micro_f1, test_macro_f1 = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        print(f"Test Micro F1: {test_micro_f1:.4f}, Macro F1: {test_macro_f1:.4f}")

        training_metrics["test_loss"].append(test_loss)
        training_metrics["test_accuracy"].append(test_accuracy)
        training_metrics["test_micro_f1"].append(test_micro_f1)
        training_metrics["test_macro_f1"].append(test_macro_f1)

    return model, training_metrics


def main(args):
    """
    extracted_path = extract_data_subfolder(args.train_data)
    output_dir = f"{extracted_path}/{args.model}/seed-{args.seed}"
    """
    set_seed(args.seed)
    output_dir = args.output_dir

    train_data = load_dataset('json', data_files=args.train_data)
    train_data = train_data['train']

    # Load test data
    test_data = load_dataset('json', data_files=args.test_data)
    test_data = test_data['train']

    # Initialize tokenizer and add entity tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = add_entity_tokens(tokenizer, args.dataset)

    # Create datasets and data loaders
    if args.val_data:
        val_data = load_dataset('json', data_files=args.val_data)
        val_data = val_data['train']
        train_dataset = RelationExtractionDataset(train_data, tokenizer, args.max_length)
        val_dataset = RelationExtractionDataset(val_data, tokenizer, args.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        # If no validation set is provided, use the entire train set for training
        train_dataset = RelationExtractionDataset(train_data, tokenizer, args.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_loader = None

    test_dataset = RelationExtractionDataset(test_data, tokenizer, args.max_length)

    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    num_relations = max(
        max(item['_id'] for item in train_data),
        max(item['_id'] for item in test_data)
    ) + 1  # Add 1 because relation IDs are 0-indexed

    model = RelationExtractionModel(num_relations)

    # Resize token embeddings after adding new tokens
    model.roberta.resize_token_embeddings(len(tokenizer))

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train and evaluate the model, and save training metrics
    model, training_metrics = train_and_evaluate(
        model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args.num_epochs, patience=3
    )
    save_metrics(training_metrics, output_dir, "training_metrics.json")

    # Final evaluation on the test set and save metrics
    test_loss, test_accuracy, test_micro_f1, test_macro_f1 = evaluate(model, test_loader, device)
    final_metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_micro_f1": test_micro_f1,
        "test_macro_f1": test_macro_f1
    }
    save_metrics(final_metrics, output_dir, "final_test_metrics.json")

    print("Final Test Results:")
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    print(f"Test Micro F1: {test_micro_f1:.4f}, Macro F1: {test_macro_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='chemprot')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default="roberta-large")
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)
