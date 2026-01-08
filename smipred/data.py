import pandas as pd
import torch
from typing import List, Tuple
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BatchEncoding
from smipred.tokenizer import Tokenizer

class MyDataset(Dataset):
    """
    Custom Dataset class for SMILES data, compatible with PyTorch DataLoader.
    Handles both Masked Language Modeling (MLM) and Downstream tasks.
    """
    def __init__(self, 
                 task_type: str,
                 input_df: pd.DataFrame, 
                 vocab_file: str,
                 tokenizer: BertTokenizer,
                 max_len: int,
                 mask_ratio: float,
                 target_name: List[str]):
        """
        Initializes the MyDataset object.

        Args:
            task_type (str): Type of task, either 'mlm' or 'downstream'.
            input_df (pd.DataFrame): DataFrame containing the input data. Must have an 'input' column.
            vocab_file (str): Path to the vocabulary file.
            tokenizer (BertTokenizer): Tokenizer to be used for processing SMILES strings.
            max_len (int): Maximum length of token sequences.
            mask_ratio (float): Probability of masking a token for MLM task.
            target_name (List[str]): List of target column names for downstream tasks.
        """
        self.task_type = task_type
        self.input_df = input_df
        self.vocab_file = vocab_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.target_name = target_name

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.input_df)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'labels', and 'attention_mask'.
        """
        # Get the input string at the specified index
        text = self.input_df['input'].iloc[idx]

        # Tokenize and encode the input string using the tokenizer
        inputs: BatchEncoding = Tokenizer(smiles=text,
                                          vocab_file=self.vocab_file, 
                                          tokenizer=self.tokenizer,
                                          max_len=self.max_len)

        # Convert the encoded input_ids and attention mask to tensors
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).squeeze(0)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).squeeze(0)

        # For MLM Task
        if self.task_type == 'mlm':
            # Create labels: clone input_ids as the targets
            labels = input_ids.clone()
            
            # Generate random mask
            rand = torch.rand(input_ids.shape)
            
            # Create a boolean mask array where:
            # 1. Random value is less than mask_ratio
            # 2. Token is NOT [CLS], [SEP], or [PAD] (assumed special tokens are managed by tokenizer)
            mask_arr = (rand < self.mask_ratio) & (input_ids != self.tokenizer.cls_token_id) & \
                       (input_ids != self.tokenizer.sep_token_id) & (input_ids != self.tokenizer.pad_token_id)

            # Replace masked tokens in input_ids with the [MASK] token ID
            input_ids[mask_arr] = self.tokenizer.mask_token_id

        # For Downstream Task
        elif self.task_type == 'downstream':
            # Retrieve target values for the sample
            target_list = self.input_df[self.target_name].values
            labels = torch.tensor(target_list[idx], dtype=torch.float32)

        return {'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,}

def MyDataLoader(task_type: str,
                 vocab_file: str,
                 tokenizer: BertTokenizer,
                 max_len: int,
                 batch_size: int,
                 mask_ratio: float,
                 target_name: List[str],
                 df_train: pd.DataFrame,
                 df_val: pd.DataFrame,
                 df_test: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        task_type (str): Type of task ('mlm' or 'downstream').
        vocab_file (str): Path to the vocabulary file.
        tokenizer (BertTokenizer): Tokenizer instance.
        max_len (int): Maximum sequence length.
        batch_size (int): Batch size for the DataLoaders.
        mask_ratio (float): Masking ratio for MLM (ignored for downstream).
        target_name (List[str]): List of target names for downstream task.
        df_train (pd.DataFrame): Training DataFrame.
        df_val (pd.DataFrame): Validation DataFrame.
        df_test (pd.DataFrame): Testing DataFrame.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for train, val, and test sets.
    """
    # Create datasets for training, validation, and optionally testing
    train_dataloader = None

    if df_train is not None:
        train_dataset = MyDataset(task_type=task_type,
                                  input_df=df_train, 
                                  vocab_file=vocab_file,
                                  tokenizer=tokenizer,
                                  max_len=max_len,
                                  mask_ratio=mask_ratio, 
                                  target_name=target_name)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = None

    if df_val is not None:
        val_dataset = MyDataset(task_type=task_type,
                                input_df=df_val,
                                vocab_file=vocab_file,
                                tokenizer=tokenizer,
                                max_len=max_len,
                                mask_ratio=mask_ratio,
                                target_name=target_name)

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataloader = None

    if df_test is not None:
        test_dataset = MyDataset(task_type=task_type,
                                 input_df=df_test,
                                 vocab_file=vocab_file,
                                 tokenizer=tokenizer,
                                 max_len=max_len,
                                 mask_ratio=mask_ratio,
                                 target_name=target_name)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
