import torch
import torch.nn as nn

class PALMEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding layer for word tokens
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # Embedding layer for positional information (e.g., position of each token in the sequence)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Embedding layer for language information (0 for source, 1 for target)
        # 2 embeddings: one for source, one for target
        self.language_embeddings = nn.Embedding(2, config.hidden_size)  # 2 for source and target
        
        # Layer normalization to stabilize and accelerate training by normalizing the input of each layer
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Fixed length for source sequence; used to distinguish between source and target positions
        self.fixed_source_length = config.fixed_source_length

    def forward(self, input_ids):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension if input is a single sequence
        
        seq_length = input_ids.size(1) # Get sequence length of the input
        
        # Separate Positional Encoding (SPE)
        # Generate position IDs ranging from 0 to seq_length-1
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        
        # Adjust position_ids if sequence length exceeds the fixed source length
        if seq_length > self.fixed_source_length:
            position_ids[self.fixed_source_length:] = torch.arange(
                seq_length - self.fixed_source_length, 
                dtype=torch.long, 
                device=input_ids.device
            )
        
        # Language IDs: 0 for source, 1 for target
        # Initialize language IDs to 0 (source language)
        language_ids = torch.zeros_like(input_ids)

        # Set language IDs to 1 (target language) for positions beyond the fixed source length
        if seq_length > self.fixed_source_length:
            language_ids[:, self.fixed_source_length:] = 1

        # Get embeddings for the input word tokens
        word_embeddings = self.word_embeddings(input_ids)

        # Get embeddings for the positions
        position_embeddings = self.position_embeddings(position_ids)

        # Get embeddings for the language (source/target)
        language_embeddings = self.language_embeddings(language_ids)

        # Sum word, position, and language embeddings to form the final embedding representation
        embeddings = word_embeddings + position_embeddings + language_embeddings
        
        # Apply layer normalization to the combined embeddings
        embeddings = self.LayerNorm(embeddings)

        # Apply dropout for regularization
        embeddings = self.dropout(embeddings)

        return embeddings # Final embedding tensor
    