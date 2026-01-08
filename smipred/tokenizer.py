import os
import re
from typing import Union, List
from transformers import BertTokenizer, BatchEncoding

def vocabulary(path: str) -> None:
    """
    Generates a vocabulary file for SMILES tokenization.
    
    Args:
        path (str): The file path where the vocabulary will be saved.
    """
    # List of element symbols commonly used in SMILES strings
    element_symbols: List[str] = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                                  'K', 'Ca', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'In', 'Sn', 'Sb', 'Te', 'I']

    # List of special symbols used in SMILES notation
    special_symbols: List[str] = ['@', '@@', '/', '\\', '-', '=', '#', '(', ')', '[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '%', '+', '.']

    # List of aromatic symbols used in SMILES notation
    aromatic_symbols: List[str] = ['c', 'n', 'o', 'p', 's']

    # List of vocabulary tokens, including special BERT-like tokens
    # [PAD]: Padding
    # [CLS]: Classification start token
    # [SEP]: Separator token
    # [MASK]: Mask token for MLM
    # [UNK]: Unknown token
    vocab_list: List[str] = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'] + element_symbols + aromatic_symbols + special_symbols

    # Write each token from vocab_list to the vocabulary file
    with open(path, 'w') as f:
        for token in vocab_list:
            f.write(token + '\n')

    return

def TokenGenerator(smiles: str, vocab_file: str) -> List[str]:
    """
    Tokenizes a SMILES string based on the provided vocabulary.

    Args:
        smiles (str): The input SMILES string.
        vocab_file (str): Path to the vocabulary file.

    Returns:
        List[str]: A list of tokens extracted from the SMILES string.
    """
    # Load vocabulary symbols from file
    with open(vocab_file, 'r') as f:
        # Sort symbols by length in descending order to match longer tokens first (greedy matching)
        # e.g., match 'Br' before 'B', '@@' before '@'
        sorted_symbols = sorted([line.strip() for line in f.readlines()], key=len, reverse=True)

    # Tokenize the SMILES string using the generated pattern
    token_pattern = '(' + '|'.join(map(re.escape, sorted_symbols)) + '|.)'
    tokens = re.findall(token_pattern, smiles)

    return tokens

def Tokenizer(smiles: Union[str, List[str]], 
              vocab_file: str,
              tokenizer: BertTokenizer,
              max_len: int) -> BatchEncoding:
    """
    Tokenizes and encodes SMILES strings for BERT input.

    Args:
        smiles (Union[str, List[str]]): Input SMILES string or list of SMILES strings.
        vocab_file (str): Path to vocabulary file.
        tokenizer (BertTokenizer): Pre-initialized BertTokenizer.
        max_len (int): Maximum sequence length for padding/truncation.

    Returns:
        BatchEncoding: Object containing 'input_ids', 'token_type_ids', and 'attention_mask'.
    """
    # Ensure smiles is a list, even if a single SMILES string is provided
    smiles_list = smiles if isinstance(smiles, list) else [smiles]

    # Initialize lists to store tokenized and encoded information
    all_input_ids, all_token_type_ids, all_attention_mask = [], [], []

    # Iterate over each SMILES string in the list
    for smiles in smiles_list:
        tokens = TokenGenerator(smiles, vocab_file)

        # Encode the tokens using the provided tokenizer
        output = tokenizer.encode_plus(
                tokens,
                is_split_into_words=True,
                add_special_tokens=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                padding='max_length',
                max_length=max_len,
                truncation=True
                )

        # Append the encoded components to their respective lists
        all_input_ids.append(output['input_ids'])
        all_token_type_ids.append(output['token_type_ids'])
        all_attention_mask.append(output['attention_mask'])

    # Create a BatchEncoding object containing the encoded inputs
    return BatchEncoding(data={
        'input_ids': all_input_ids,
        'token_type_ids': all_token_type_ids,
        'attention_mask': all_attention_mask
        })
