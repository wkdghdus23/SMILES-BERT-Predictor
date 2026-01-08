import os
import torch
import argparse
import pandas as pd
from torch.utils.data import random_split
from transformers import BertTokenizer, BertModel
from smipred.tokenizer import vocabulary
from smipred.model import BertForMLM, BertForDownstream
from smipred.trainer import train
from smipred.utils import set_seed

# Set the random seed for reproducibility
SEED = 42
set_seed(SEED)

def main():
    """
    Main function to orchestrate the training process for SMILES-BERT.
    Parses command-line arguments, sets up the dataset, initializes the model,
    and starts the training loop (which includes validation and testing).
    """
    # Parse command line arguments for model training configuration
    parser = argparse.ArgumentParser(description="Unified BERT Training for MLM and Downstream Tasks")
    parser.add_argument('--task', type=str, required=True, choices=['mlm', 'downstream'],
                        help="Task type: 'mlm' for Masked Language Modeling, 'downstream' for target prediction.")
    parser.add_argument('--pretrained', type=str, default=None, help='Path of pre-trained Model')
    parser.add_argument('--dataset', type=str, help='Path to the dataset with CSV format')
    parser.add_argument('--vocabfile', type=str, default='./vocab.txt', help='Vocabulary file for tokenizing')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum length of input sequence')
    parser.add_argument('--batchsize', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--modelsavepath', type=str, default='./results', help='Save path of finetuned model')
    parser.add_argument('--masking', type=float, default=0.0, help='Masking ratio for MLM task')
    parser.add_argument('--target', type=str, nargs='+', default=None,
                        help='List of target names for downstream prediction (e.g., --target_list HOMO LUMO)')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Check if the vocabulary file exists; if not, create it using vocabulary() function
    vocabulary(path = args.vocabfile)

    # Load dataset CSV file
    # The CSV must contain an 'input' column for SMILES strings
    df = pd.read_csv(args.dataset)

    # Initialize the tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocabfile, 
                              clean_up_tokenization_spaces=True,
                              do_lower_case=False,
                              do_basic_tokenize=False)

    # Load training, validation, and test datasets using pandas
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    test_size = len(df) - train_size - val_size

    # Split the dataset randomly
    train_subset, val_subset, test_subset = random_split(df, [train_size, val_size, test_size])

    # Convert Subsets back to DataFrame and reset index
    df_train = df.iloc[train_subset.indices].reset_index(drop=True)
    df_val = df.iloc[val_subset.indices].reset_index(drop=True)
    df_test = df.iloc[test_subset.indices].reset_index(drop=True)

    # Task type: Masked Language Modeling (MLM)
    # Used for pre-training the model on SMILES strings to learn chemical syntax features
    if args.task == 'mlm':
        # Load pre-trained BERT model or initialize a new one
        if args.pretrained is None:
            # Initialize from scratch
            model = BertForMLM(tokenizer=tokenizer)
        else:
            # Continue training from a checkpoint
            model = BertForMLM.from_pretrained_model(tokenizer=tokenizer,
                                                     pretrained_path=args.pretrained)

    # Task type: Downstream task for target prediction
    # Used for finetuning the model to predict specific chemical properties
    elif args.task == 'downstream':
        # Load pre-trained BERT model
        if args.pretrained is None:
            # Initialize from scratch (uncommon for downstream)
            model = BertForDownstream(tokenizer=tokenizer, target_name=args.target)
        else:
            # Initialize using pre-trained weights (e.g. from MLM task)
            model = BertForDownstream.from_pretrained_model(tokenizer=tokenizer,
                                                            target_name=args.target,
                                                            pretrained_path=args.pretrained)

    # Start training the model
    train(task_type=args.task,
            model=model,
            vocab_file=args.vocabfile,
            tokenizer=tokenizer,
            max_len=args.max_len,
            batch_size=args.batchsize,
            epochs=args.epochs,
            mask_ratio=args.masking,
            target_name=args.target,
            model_save_path=args.modelsavepath,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test)

if __name__ == '__main__':
    main()
