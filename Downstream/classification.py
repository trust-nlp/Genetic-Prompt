import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
import json
import argparse
import random
import os
import numpy as np
from datetime import datetime

class TextClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(TextClassificationModel, self).__init__()
        if 'roberta' in model_name.lower():
            self.encoder = RobertaModel.from_pretrained(model_name)
        elif 'bert' in model_name.lower():
            self.encoder = BertModel.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        try:
            label = int(item['_id']) 
        except ValueError:
            print(f"Warning: Unable to convert _id '{item['_id']}' to integer. Using 0 as default.")
            label = 0  

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

def load_json_data(file_path, max_samples=None):
    """
    Load data from a .json or .jsonl file. If the file has a .json extension but
    contains JSON Lines, it will be parsed line by line as a fallback.
    Optionally subsample to `max_samples`.
    """
    data = []
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_extension == '.json':
                try:
                    data = json.load(f)
                    print(f"Successfully loaded JSON file: {file_path}")
                except json.JSONDecodeError:
                    f.seek(0)
                    data = []
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                    print(f"Successfully loaded JSONL file: {file_path}")
            elif file_extension == '.jsonl':
                for line in f:
                    if line.strip(): 
                        data.append(json.loads(line))
                print(f"Successfully loaded JSONL file: {file_path}")
            else:
                # Fallback: try JSON first, then JSON Lines
                try:
                    data = json.load(f)
                    print(f"Successfully loaded JSON file: {file_path}")
                except json.JSONDecodeError:
                    f.seek(0)
                    data = []
                    for line in f:
                        if line.strip():  
                            data.append(json.loads(line))
                    print(f"Successfully loaded JSONL file: {file_path}")
                    
    except Exception as e:
        print(f"Error while loading data: {e}")
        raise
        
    # Ensure the loaded data is a list
    if not isinstance(data, list):
        print(f"Warning: Data loaded from {file_path} is not a list. Wrapping it into a list.")
        data = [data]
        
    print(f"Loaded {len(data)} samples from {file_path}")
    
    # Apply maximum sample limit if requested
    if max_samples is not None and max_samples < len(data):
        data = random.sample(data, max_samples)
        print(f"Randomly sampled {max_samples} samples")
        
    return data

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

    return avg_loss, accuracy, micro_f1, macro_f1, predictions, true_labels

def save_performance_metrics(metrics, base_path):
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    # Save performance metrics as JSON
    json_path = base_path + '.json'
    with open(json_path, 'w') as jsonfile:
        json.dump(metrics, jsonfile, indent=4)
    
    # Also save a detailed classification report for each epoch as text files
    for m in metrics:
        epoch_report_path = f"{base_path}_epoch_{m['epoch']}_report.txt"
        with open(epoch_report_path, 'w') as f:
            f.write(m['classification_report'])
    
    print(f"Saved performance metrics to {json_path}")
    print(f"Saved classification reports for each epoch")

def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, num_epochs, save_dir):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # List to collect performance metrics
    performance_metrics = []

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
        
        # Initialize validation metrics
        val_loss, val_accuracy, val_micro_f1, val_macro_f1 = 0, 0, 0, 0
        
        # Evaluate on the validation set (if provided)
        if val_loader:
            val_loss, val_accuracy, val_micro_f1, val_macro_f1, _, _ = evaluate(model, val_loader, device)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            print(f"Validation Micro F1: {val_micro_f1:.4f}, Macro F1: {val_macro_f1:.4f}")

        # Evaluate on the test set at the end of each epoch
        test_loss, test_accuracy, test_micro_f1, test_macro_f1, test_preds, test_labels = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        print(f"Test Micro F1: {test_micro_f1:.4f}, Macro F1: {test_macro_f1:.4f}")
        
        # Generate classification report
        class_report = classification_report(test_labels, test_preds)
        print("Classification Report:")
        print(class_report)
        
        # Record detailed prediction results
        predictions_data = []
        for pred, true_label in zip(test_preds, test_labels):
            predictions_data.append({
                'predicted': int(pred),
                'true': int(true_label),
                'correct': pred == true_label
            })
        
        # Save metrics
        metric_entry = {
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'val_micro_f1': float(val_micro_f1),
            'val_macro_f1': float(val_macro_f1),
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_micro_f1': float(test_micro_f1),
            'test_macro_f1': float(test_macro_f1),
            'classification_report': class_report,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'predictions': predictions_data
        }
        performance_metrics.append(metric_entry)
        
        # Save metrics at the end of each epoch
        # Save metrics for the current epoch
        epoch_metrics_file = os.path.join(save_dir, f'epoch_{epoch+1}_metrics')
        save_performance_metrics([metric_entry], epoch_metrics_file)
        
        # Save accumulated metrics for all epochs
        all_metrics_file = os.path.join(save_dir, 'all_performance_metrics')
        save_performance_metrics(performance_metrics, all_metrics_file)
    
    return performance_metrics

def get_num_classes(data):
    class_ids = set()
    for item in data:
        try:
            class_ids.add(int(item['_id']))
        except ValueError:
            print(f"Warning: Unable to convert _id '{item['_id']}' to integer. Skipping this item.")
    return max(class_ids) + 1 if class_ids else 0

def main(args):
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Derive dataset name and train split name from the train path
    dataset_name = os.path.basename(os.path.dirname(args.train_data))
    
    # Determine the training file base name based on extension
    if args.train_data.lower().endswith('.jsonl'):
        train_file_name = os.path.basename(args.train_data)[:-6]  # remove .jsonl
    else:
        train_file_name = os.path.basename(args.train_data).split('.')[0]  # works for .json and others
    
    # Shorten model name for directory path
    model_name_short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    
    # Build experiment directory that includes model, seed, and max_samples
    max_samples_str = str(args.max_samples) if args.max_samples else "all"
    experiment_dir = f"{model_name_short}-seed{args.seed}-{max_samples_str}"
    save_dir = os.path.join(current_dir, dataset_name, train_file_name, experiment_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Results will be saved to: {save_dir}")
    
    # Load data
    train_data = load_json_data(args.train_data, args.max_samples)
    test_data = load_json_data(args.test_data, args.max_samples)
    
    # Load validation data only if a valid path is provided
    val_data = None
    if args.val_data and args.val_data.lower() != 'none':
        val_data = load_json_data(args.val_data, args.max_samples)

    # Initialize tokenizer based on the model name
    if 'roberta' in args.model_name.lower():
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    elif 'bert' in args.model_name.lower():
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Create datasets and dataloaders
    train_dataset = TextClassificationDataset(train_data, tokenizer, args.max_length)
    test_dataset = TextClassificationDataset(test_data, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create validation dataloader only if validation data exists
    val_loader = None
    if val_data:
        val_dataset = TextClassificationDataset(val_data, tokenizer, args.max_length)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    num_classes = max(get_num_classes(train_data), get_num_classes(test_data))
    print(f"Number of classes: {num_classes}")
    model = TextClassificationModel(args.model_name, num_classes)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save training configuration info
    config_info = {
        'model_name': args.model_name,
        'seed': args.seed,
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'epochs': args.num_epochs,
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'val_samples': len(val_data) if val_data else 0,
        'device': str(device),
        'dataset_name': dataset_name,
        'train_file': train_file_name
    }
    
    # Persist configuration to disk
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config_info, f, indent=4)
    
    # Train and evaluate the model
    performance_metrics = train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, scheduler, device, args.num_epochs, save_dir)
    
    # Save final performance summary
    summary = {
        'model_name': args.model_name,
        'seed': args.seed,
        'max_samples': args.max_samples,
        'final_test_accuracy': performance_metrics[-1]['test_accuracy'],
        'final_test_micro_f1': performance_metrics[-1]['test_micro_f1'],
        'final_test_macro_f1': performance_metrics[-1]['test_macro_f1'],
        'best_epoch': max(range(len(performance_metrics)), 
                          key=lambda i: performance_metrics[i]['test_accuracy']) + 1,
        'best_test_accuracy': max(m['test_accuracy'] for m in performance_metrics),
        'best_test_micro_f1': max(m['test_micro_f1'] for m in performance_metrics),
        'best_test_macro_f1': max(m['test_macro_f1'] for m in performance_metrics),
        'total_epochs': len(performance_metrics),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = os.path.join(save_dir, 'performance_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Performance summary saved to {summary_path}")
    
    # Save model weights
    model_path = os.path.join(save_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use. Use None for no limit.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        help='Model name or path (e.g., roberta-base, bert-base-uncased)')
    args = parser.parse_args()

    main(args)
