import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from typing import Union, List
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from smipred.data import MyDataLoader

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(task_type: str,
          model: Union[BertForMaskedLM, BertModel],
          vocab_file: str,
          tokenizer: BertTokenizer,
          max_len: int,
          batch_size: int,
          epochs: int,
          mask_ratio: float,
          target_name: List[str],
          model_save_path: str,
          df_train: pd.DataFrame,
          df_val: pd.DataFrame,
          df_test: pd.DataFrame) -> None:
    """
    Executes the training loop for the BERT model (either MLM or Downstream task).

    Args:
        task_type (str): 'mlm' or 'downstream'.
        model (Union[BertForMaskedLM, BertModel]): The model to train.
        vocab_file (str): Path to vocabulary file.
        tokenizer (BertTokenizer): Tokenizer instance.
        max_len (int): Max sequence length.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
        mask_ratio (float): Masking ratio for MLM.
        target_name (List[str]): Target names for downstream regression.
        model_save_path (str): Directory to save model checkpoints.
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        df_test (pd.DataFrame): Test data.
    """
    # Move model to the specified device
    model.to(device)

    # Create data loaders based on the task type
    train_dataloader, val_dataloader, test_dataloader = MyDataLoader(task_type=task_type,
                                                                     vocab_file=vocab_file,
                                                                     tokenizer=tokenizer,
                                                                     max_len=max_len,
                                                                     batch_size=batch_size,
                                                                     mask_ratio=mask_ratio,
                                                                     target_name=target_name,
                                                                     df_train=df_train,
                                                                     df_val=df_val,
                                                                     df_test=df_test)

    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Loss function for regression task (Mean Squared Error)
    loss_fn = MSELoss() if task_type == 'downstream' else None

    # Training loop
    for epoch in range(epochs):
        model.train()

        total_loss = 0
        # Iterate over each batch of training data
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero the gradients to prevent accumulation
            optimizer.zero_grad()

            # Forward pass
            if task_type == 'mlm':
                # BERT for MLM returns loss directly
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
            elif task_type == 'downstream':
                # Custom downstream model returns predictions
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

            # Backward pass: compute gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
           
            # Accumulate training loss
            total_loss += loss.item()

        # Print average training loss for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.5f}")

        # Validation step
        avg_val_loss = evaluate(task_type, model, val_dataloader, device, loss_fn if task_type == 'downstream' else None)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.5f}")

    # Test step after all epochs
    avg_test_loss = evaluate(task_type, model, test_dataloader, device, loss_fn)
    print(f"Test Loss: {avg_test_loss:.5f}")

    # Save the trained model
    if task_type == 'mlm':
        savepath=f"{model_save_path}/{task_type}_batchsize{batch_size}_epochs{epochs}_{mask_ratio}masking/"
        # For MLM, save the HuggingFace model directly
        model.model.save_pretrained(savepath)
    else:
        savepath=f"{model_save_path}/{task_type}_batchsize{batch_size}_epochs{epochs}_{''.join(target_name)}/"
        # For downstream, save the underlying BERT model and the full state dict (including regression head)
        model.bert.save_pretrained(savepath)
        torch.save(model.state_dict(), f"{savepath}/DownstreamModel.pt")

def evaluate(task_type: str,
             model: Union[BertForMaskedLM, BertModel],
             dataloader: DataLoader,
             device: torch.device,
             loss_fn: nn.Module = None) -> float:
    """
    Evaluates the model on a given dataset (validation or test).

    Args:
        task_type (str): 'mlm' or 'downstream'.
        model (Union[BertForMaskedLM, BertModel]): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation set.
        device (torch.device): Computation device.
        loss_fn (nn.Module, optional): Loss function (required for downstream).

    Returns:
        float: Average loss over the dataset.
    """
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # Disable gradient calculation for evaluation to save memory
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move inputs and labels to the appropriate device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            total_samples += labels.shape[0]

            # Forward pass through the model
            if task_type == 'mlm':
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
            elif task_type == 'downstream':
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

            # Accumulate the loss (weighted by batch size for accurate average)
            total_loss += loss.item() * labels.shape[0]

    # Calculate average loss over all samples
    dataset_size = len(dataloader.dataset)
    average_loss = total_loss / dataset_size

    return average_loss
