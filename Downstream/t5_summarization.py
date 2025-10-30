import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from rouge import Rouge
import json
import argparse
import random
import os
import numpy as np
from datetime import datetime

class TextSummarizationModel(nn.Module):
    def __init__(self,model='t5-base', dropout_rate=0.1):
        super(TextSummarizationModel, self).__init__()
        # Set the dropout parameter when creating the T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(model, dropout_rate=dropout_rate)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

class TextSummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length, max_target_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = "summarize: "
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.prefix + item['text']
        summary = item['summary']
        
        # Tokenize inputs
        input_encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        target_encoding = self.tokenizer.encode_plus(
            summary,
            add_special_tokens=True,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids']
        # Replace padding token id's with -100 so they are not included in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

def load_json_data(file_path, max_samples=None):
    """
    Load data from JSON or JSONL format files.
    Automatically detects the file format and processes the data accordingly.
    
    Args:
        file_path: Path to the JSON or JSONL file
        max_samples: Maximum number of samples to load, None means no limit
        
    Returns:
        List of loaded data
    """
    data = []
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # First try to read as standard JSON file
            if file_extension == '.json':
                try:
                    # Try to read as a single JSON object/array
                    data = json.load(f)
                    print(f"Successfully loaded JSON file: {file_path}")
                except json.JSONDecodeError:
                    # If failed, reset file pointer and try as JSONL
                    f.seek(0)
                    data = []
                    for line in f:
                        if line.strip():  # Skip empty lines
                            data.append(json.loads(line))
                    print(f"Successfully loaded JSONL file: {file_path}")
            # Explicit JSONL file
            elif file_extension == '.jsonl':
                for line in f:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
                print(f"Successfully loaded JSONL file: {file_path}")
            else:
                # For unknown extensions, try both formats
                try:
                    # First try as JSON
                    data = json.load(f)
                    print(f"Successfully loaded JSON file: {file_path}")
                except json.JSONDecodeError:
                    # If failed, reset file pointer and try as JSONL
                    f.seek(0)
                    data = []
                    for line in f:
                        if line.strip():  # Skip empty lines
                            data.append(json.loads(line))
                    print(f"Successfully loaded JSONL file: {file_path}")
                    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
        
    # Ensure data is a list
    if not isinstance(data, list):
        print(f"Warning: Data loaded from {file_path} is not a list. Wrapping the data in a list.")
        data = [data]
        
    print(f"Loaded {len(data)} samples from {file_path}")
    
    # Apply max samples limit
    if max_samples is not None and max_samples < len(data):
        data = random.sample(data, max_samples)
        print(f"Randomly sampled {max_samples} samples")
        
    return data

def calculate_rouge_scores(generated_summaries, reference_summaries):
    """
    Calculate ROUGE scores between generated and reference summaries.
    
    Args:
        generated_summaries: List of generated summary texts
        reference_summaries: List of reference summary texts
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    # Ensure both lists have the same length and are not empty
    if not generated_summaries or not reference_summaries:
        print("Warning: Empty summaries provided to calculate_rouge_scores")
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
            'mean_rouge': 0.0
        }
    
    rouge = Rouge()
    scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)
    
    # Extract the F1 scores for each ROUGE metric
    rouge_1_f1 = scores['rouge-1']['f']
    rouge_2_f1 = scores['rouge-2']['f']
    rouge_l_f1 = scores['rouge-l']['f']
    
    return {
        'rouge-1': rouge_1_f1,
        'rouge-2': rouge_2_f1,
        'rouge-l': rouge_l_f1,
        'mean_rouge': (rouge_1_f1 + rouge_2_f1 + rouge_l_f1) / 3
    }

def evaluate(model, data_loader, tokenizer, device, max_target_length):
    model.eval()
    total_loss = 0
    generated_summaries = []
    reference_summaries = []
    
    # Check if the dataloader is empty
    if len(data_loader) == 0:
        print("Warning: data_loader is empty!")
        return 0.0, {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
            'mean_rouge': 0.0
        }, [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Generate summaries
            generated_ids = model.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length,
                num_beams=4,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            
            # Decode generated summaries
            for g_id, label in zip(generated_ids, labels):
                gen_summary = tokenizer.decode(g_id, skip_special_tokens=True)
                generated_summaries.append(gen_summary)
                
                # Create reference summary by filtering out -100 tokens
                label_ids = label.tolist()
                label_ids = [id_ for id_ in label_ids if id_ != -100]
                ref_summary = tokenizer.decode(label_ids, skip_special_tokens=True)
                reference_summaries.append(ref_summary)
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(generated_summaries, reference_summaries)
    
    return avg_loss, rouge_scores, generated_summaries, reference_summaries

def save_performance_metrics(metrics, base_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    # Save performance metrics in JSON format
    json_path = base_path + '.json'
    with open(json_path, 'w') as jsonfile:
        # Save only key metrics, not example summaries
        clean_metrics = []
        for m in metrics:
            clean_m = {k: v for k, v in m.items() if 'example' not in k and 'summaries' not in k}
            clean_metrics.append(clean_m)
        json.dump(clean_metrics, jsonfile, indent=4)
    
    print(f"Saved performance metrics to {json_path}")

def train_and_evaluate(model, train_loader, val_loader, test_loader, tokenizer, optimizer, scheduler, device, 
                      num_epochs, save_dir, max_target_length, max_grad_norm=1.0):
    model.to(device)
    
    # List to store performance metrics
    performance_metrics = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average train loss: {avg_train_loss:.4f}")
        
        # Initialize metric values
        val_loss, val_rouge_scores = 0, {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'mean_rouge': 0}
        val_gen_summaries, val_ref_summaries = [], []
        
        # Evaluate on validation set (if exists)
        if val_loader:
            val_loss, val_rouge_scores, val_gen_summaries, val_ref_summaries = evaluate(model, val_loader, tokenizer, device, max_target_length)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation ROUGE-1: {val_rouge_scores['rouge-1']:.4f}, ROUGE-2: {val_rouge_scores['rouge-2']:.4f}, ROUGE-L: {val_rouge_scores['rouge-l']:.4f}")

        # Evaluate on test set after each epoch
        test_loss, test_rouge_scores, test_gen_summaries, test_ref_summaries = evaluate(model, test_loader, tokenizer, device, max_target_length)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test ROUGE-1: {test_rouge_scores['rouge-1']:.4f}, ROUGE-2: {test_rouge_scores['rouge-2']:.4f}, ROUGE-L: {test_rouge_scores['rouge-l']:.4f}")
        
        # Save performance metrics (without examples)
        metric_entry = {
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(val_loss),
            'val_rouge_1': float(val_rouge_scores['rouge-1']),
            'val_rouge_2': float(val_rouge_scores['rouge-2']),
            'val_rouge_l': float(val_rouge_scores['rouge-l']),
            'val_mean_rouge': float(val_rouge_scores['mean_rouge']),
            'test_loss': float(test_loss),
            'test_rouge_1': float(test_rouge_scores['rouge-1']),
            'test_rouge_2': float(test_rouge_scores['rouge-2']),
            'test_rouge_l': float(test_rouge_scores['rouge-l']),
            'test_mean_rouge': float(test_rouge_scores['mean_rouge']),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        performance_metrics.append(metric_entry)
        
        # Save current-epoch metrics
        epoch_metrics_file = os.path.join(save_dir, f'epoch_{epoch+1}_metrics')
        save_performance_metrics([metric_entry], epoch_metrics_file)
        
        # Save all accumulated metrics
        all_metrics_file = os.path.join(save_dir, 'all_performance_metrics')
        save_performance_metrics(performance_metrics, all_metrics_file)
    
    return performance_metrics

def main(args):
    # Get current directory
    current_dir = os.getcwd()
    
    # Extract dataset name and train file name from train data path
    dataset_name = os.path.basename(os.path.dirname(args.train_data))
    
    # Determine train file name based on file extension
    if args.train_data.lower().endswith('.jsonl'):
        train_file_name = os.path.basename(args.train_data)[:-6]  # Remove .jsonl
    else:
        train_file_name = os.path.basename(args.train_data).split('.')[0]  # For .json and other extensions
    
    # Extract a short model name for directory naming
    model_short_name = os.path.basename(args.model)
    
    # Max samples info
    max_samples_str = str(args.max_samples) if args.max_samples else "all"

    # Create a directory structure including model name and max samples
    save_dir = os.path.join(current_dir, dataset_name, f"{train_file_name}_{model_short_name}_{max_samples_str}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Results will be saved to: {save_dir}")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    train_data = load_json_data(args.train_data, args.max_samples)
    test_data = load_json_data(args.test_data, args.max_samples)
    
    # Only load validation set if a valid path is provided
    val_data = None
    if args.val_data and args.val_data.lower() != 'none':
        val_data = load_json_data(args.val_data, args.max_samples)

    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    # Create datasets and dataloaders
    train_dataset = TextSummarizationDataset(train_data, tokenizer, args.max_input_length, args.max_target_length)
    test_dataset = TextSummarizationDataset(test_data, tokenizer, args.max_input_length, args.max_target_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Debug information
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test dataloader size: {len(test_loader)}")
    print(f"Test dataloader batch size: {test_loader.batch_size}")
    
    # Create validation dataloader only if validation data exists
    val_loader = None
    if val_data:
        val_dataset = TextSummarizationDataset(val_data, tokenizer, args.max_input_length, args.max_target_length)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        print(f"Val dataset size: {len(val_dataset)}")
        print(f"Val dataloader size: {len(val_loader)}")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Train dataloader size: {len(train_loader)}")
    
    # Initialize model with dropout
    model = TextSummarizationModel(args.model, dropout_rate=args.dropout_rate)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * args.warmup_ratio),  # Use a parameterized warmup ratio
        num_training_steps=total_steps
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save training configuration
    config_info = {
        'model_name': args.model,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout_rate': args.dropout_rate,
        'max_grad_norm': args.max_grad_norm,
        'warmup_ratio': args.warmup_ratio,
        'max_input_length': args.max_input_length,
        'max_target_length': args.max_target_length,
        'epochs': args.num_epochs,
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'val_samples': len(val_data) if val_data else 0,
        'device': str(device),
        'dataset_name': dataset_name,
        'train_file': train_file_name,
        'seed': args.seed,
        'max_samples': args.max_samples
    }
    
    # Save configuration to directory
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config_info, f, indent=4)
    
    # Train and evaluate model
    performance_metrics = train_and_evaluate(
        model, train_loader, val_loader, test_loader, 
        tokenizer, optimizer, scheduler, device, 
        args.num_epochs, save_dir, args.max_target_length,
        max_grad_norm=args.max_grad_norm
    )
    
    # Save final performance summary
    summary = {
        'final_test_loss': performance_metrics[-1]['test_loss'],
        'final_test_rouge_1': performance_metrics[-1]['test_rouge_1'],
        'final_test_rouge_2': performance_metrics[-1]['test_rouge_2'],
        'final_test_rouge_l': performance_metrics[-1]['test_rouge_l'],
        'final_test_mean_rouge': performance_metrics[-1]['test_mean_rouge'],
        'best_epoch': max(range(len(performance_metrics)), 
                         key=lambda i: performance_metrics[i]['test_mean_rouge']) + 1,
        'best_test_mean_rouge': max(m['test_mean_rouge'] for m in performance_metrics),
        'best_test_rouge_1': max(m['test_rouge_1'] for m in performance_metrics),
        'best_test_rouge_2': max(m['test_rouge_2'] for m in performance_metrics),
        'best_test_rouge_l': max(m['test_rouge_l'] for m in performance_metrics),
        'total_epochs': len(performance_metrics),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'max_samples': args.max_samples
    }
    
    # Log which epoch achieved the best performance
    best_epoch = summary['best_epoch']
    print(f"Best performance at epoch {best_epoch}")
    
    summary_path = os.path.join(save_dir, 'performance_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Performance summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune T5 for text summarization')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data file (JSON/JSONL)')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data file (JSON/JSONL)')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation data file (JSON/JSONL)')
    parser.add_argument('--max_input_length', type=int, default=512, help='Maximum input sequence length')
    parser.add_argument('--max_target_length', type=int, default=128, help='Maximum target sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use. Use None for no limit.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # Newly added hyperparameters
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for gradient clipping')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of total steps for warmup')
    parser.add_argument('--model', type=str, default='t5-base', help='Model name or path')
    args = parser.parse_args()

    main(args)
