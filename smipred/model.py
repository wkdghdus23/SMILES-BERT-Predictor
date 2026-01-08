import torch
import torch.nn as nn
from typing import List
from transformers import BertConfig, BertForMaskedLM, BertModel, BertTokenizer

class BertForMLM(nn.Module):
    """
    BERT model for Masked Language Modeling (MLM).
    Used for pre-training on SMILES data.
    """
    def __init__(self,
                 tokenizer: BertTokenizer,
                 hidden_size: int = 768, 
                 num_attention_heads: int = 12, 
                 num_hidden_layers: int = 12):
        """
        Initializes the BertForMLM model.

        Args:
            tokenizer (BertTokenizer): Tokenizer used to determine vocab size.
            hidden_size (int): Size of the hidden layers. Default is 768.
            num_attention_heads (int): Number of attention heads. Default is 12.
            num_hidden_layers (int): Number of hidden layers. Default is 12.
        """
        super(BertForMLM, self).__init__()

        # Configure BERT model for Masked Language Modeling
        self.tokenizer = tokenizer
        self.config = BertConfig.from_pretrained('bert-base-cased')
        self.config.is_decoder = False
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = num_attention_heads
        self.config.num_hidden_layers = num_hidden_layers
        self.config.vocab_size = self.tokenizer.vocab_size

        # Initialize the BERT model for Masked Language Modeling
        self.model = BertForMaskedLM(self.config)

    def forward(self, input_ids, labels, attention_mask):
        """
        Forward pass for the MLM task.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            labels (torch.Tensor): Labels for computing the masked language modeling loss.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.

        Returns:
            transformers.modeling_outputs.MaskedLMOutput: Output containing loss and logits.
        """
        return self.model(
                input_ids=input_ids, 
                labels=labels, 
                attention_mask=attention_mask
                )

    @classmethod
    def from_pretrained_model(cls, pretrained_path: str, tokenizer: BertTokenizer):
        """
        Loads a pre-trained MLM model.

        Args:
            pretrained_path (str): Path to the pre-trained model directory.
            tokenizer (BertTokenizer): Tokenizer instance.

        Returns:
            BertForMLM: An instance of the model populated with pre-trained weights.
        """
        model = cls(tokenizer=tokenizer)
        model.model = BertForMaskedLM.from_pretrained(pretrained_path)

        return model

class BertForDownstream(nn.Module):
    """
    BERT model for Downstream tasks (Property Prediction/Regression).
    """
    def __init__(self,
                 tokenizer: BertTokenizer,
                 target_name: List[str],
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 12):
        """
        Initializes the BertForDownstream model.

        Args:
            tokenizer (BertTokenizer): Tokenizer to determine vocab size.
            target_name (List[str]): List of target property names to predict.
            hidden_size (int): Size of the hidden layers.
            num_attention_heads (int): Number of attention heads.
            num_hidden_layers (int): Number of hidden layers.
        """
        super(BertForDownstream, self).__init__()

        # Configure BERT model from scratch (random initialization)
        self.tokenizer = tokenizer
        self.config = BertConfig.from_pretrained('bert-base-cased')
        self.config.is_decoder = False
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = num_attention_heads
        self.config.num_hidden_layers = num_hidden_layers
        self.config.vocab_size = self.tokenizer.vocab_size

        # Core BERT model without the classification head
        self.bert = BertModel(self.config)
        self.target_name = target_name
        
        # Regression head: Linear layer map hidden state to target properties
        self.fc = nn.Linear(self.bert.config.hidden_size, len(self.target_name))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for downstream regression task.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Predicted values for the target properties.
        """
        # Forward pass through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token output from the last hidden state
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout for regularization
        cls_output = self.dropout(cls_output)

        # Pass through the fully connected layer to predict target values
        prediction = self.fc(cls_output)

        return prediction

    @classmethod
    def from_pretrained_model(cls, pretrained_path: str, tokenizer: BertTokenizer, target_name: List[str]):
        """
        Loads a pre-trained BERT model for downstream fine-tuning.

        Args:
            pretrained_path (str): Path to the pre-trained BERT model.
            tokenizer (BertTokenizer): Tokenizer instance.
            target_name (List[str]): List of target names.

        Returns:
            BertForDownstream: Model initialized with pre-trained BERT weights.
        """
        model = cls(tokenizer=tokenizer, target_name=target_name)
        model.bert = BertModel.from_pretrained(pretrained_path)

        return model
