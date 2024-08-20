import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Logger object
logger = logging.getLogger()
# Set the level of the logger. Possible values: DEBUG, INFO, WARNING, ERROR, CRITICAL
logger.setLevel(logging.DEBUG)
# Handler that writes log messages to the notebook's output
handler = logging.StreamHandler()
# Set the format for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(handler)


class PALMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads # Number of attention heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # Size per attention head
        self.all_head_size = self.num_attention_heads * self.attention_head_size # Total size for all attention heads

        # Linear layers to project hidden states into query, key, and value representations
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Linear layer for output of the attention mechanism
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # Layer normalization for stability and improved training
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


    def transpose_for_scores(self, x):
        # Reshape input tensor for multi-head attention and permute dimensions
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # Permute dimensions to (batch, heads, seq_len, head_size)


    def forward(self, hidden_states, attention_mask=None):
        try:
            logger.debug(f"Hidden states shape: {hidden_states.shape}") # Log shape of hidden states
            query_layer = self.transpose_for_scores(self.query(hidden_states)) # Compute query matrix
            key_layer = self.transpose_for_scores(self.key(hidden_states)) # Compute key matrix
            value_layer = self.transpose_for_scores(self.value(hidden_states)) # Compute value matrix
    
            logger.debug(f"Query layer shape: {query_layer.shape}")
            logger.debug(f"Key layer shape: {key_layer.shape}")
            logger.debug(f"Value layer shape: {value_layer.shape}")
    
            # Calculate attention scores by performing matrix multiplication between query and key layers
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size) # Scale the scores
    
            logger.debug(f"Attention scores shape: {attention_scores.shape}")
    
            if attention_mask is not None:
                logger.debug(f"Original attention mask shape: {attention_mask.shape}")
                
                # Ensure attention_mask has the correct shape (4D tensor)
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                
                # Convert attention mask to float and scale it to large negative values where mask is 0
                attention_mask = attention_mask.to(dtype=torch.float32)
                attention_mask = (1.0 - attention_mask) * -10000.0
    
                logger.debug(f"Reshaped attention mask shape: {attention_mask.shape}")
    
            attention_scores = attention_scores + attention_mask # Apply attention mask
    
            # Compute attention probabilities using softmax
            attention_probs = F.softmax(attention_scores, dim=-1)
            logger.debug(f"Attention probs shape: {attention_probs.shape}")
    
            # Apply dropout to the attention probabilities
            attention_probs = self.dropout(attention_probs)

            # Compute context layer by applying attention to the value layer
            context_layer = torch.matmul(attention_probs, value_layer)
    
            logger.debug(f"Context layer shape before permute: {context_layer.shape}")
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # Permute dimensions back
            logger.debug(f"Context layer shape after permute: {context_layer.shape}")
    
            # Reshape context layer to combine attention heads
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
    
            logger.debug(f"Context layer shape after reshaping: {context_layer.shape}")
    
            # Pass context layer through a dense layer and apply layer normalization
            attention_output = self.dense(context_layer)
            attention_output = self.dropout(attention_output)
            attention_output = self.LayerNorm(attention_output + hidden_states)
    
            logger.debug(f"Attention output shape: {attention_output.shape}")
    
            return attention_output # Return the final attention output
    
        except Exception as e:
            logger.error(f"Error in PALMAttention forward pass: {str(e)}")
            logger.error(f"Input shapes - hidden_states: {hidden_states.shape} "
                         f"attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}")
            raise  # Re-raise the exception for further handling


class PALMPartialAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads # Number of attention heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # Size per attention head
        self.all_head_size = self.num_attention_heads * self.attention_head_size # Total size for all attention heads

        # Linear layers to project hidden states into query, key, and value representations
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Linear layer for output of the attention mechanism
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # Layer normalization for stability and improved training
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Fixed length for the source sequence
        self.fixed_source_length = config.fixed_source_length

        # Feed-forward network applied to the source states
        self.Fp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(), # Experiment with a reLu here
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def transpose_for_scores(self, x):
        # Reshape input tensor for multi-head attention and permute dimensions
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        # Permute dimensions to (batch, heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, source_states, attention_mask=None):
        # Apply feed-forward network to the source states
        P = self.Fp(source_states)
    
        # Compute query matrix
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # Compute key matrix from the transformed source states
        key_layer = self.transpose_for_scores(self.key(P))

        # Compute value matrix from the transformed source states
        value_layer = self.transpose_for_scores(self.value(P))

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # Scale the scores
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    
        if attention_mask is not None:
            # Ensure attention_mask has the correct shape (4D tensor)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Convert attention mask to float and scale it to large negative values where mask is 0
            attention_mask = (1.0 - attention_mask.float()) * -10000.0
            
            # Apply attention mask, but only for the source sequence part
            source_attention_mask = attention_mask[:, :, :, :self.fixed_source_length]
            attention_scores = attention_scores + source_attention_mask
    
        # Compute attention probabilities using softmax
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply dropout to the attention probabilities
        attention_probs = self.dropout(attention_probs)
    
        # Compute context layer by applying attention to the value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Ensure context_layer has the expected number of dimensions
        if context_layer.dim() != 4:
            print(f"Unexpected context_layer shape: {context_layer.shape}")
            context_layer = context_layer.view(hidden_states.size(0), -1, self.num_attention_heads, self.attention_head_size)
        
        # Permute dimensions back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        
        # Reshape the context layer
        context_layer = context_layer.view(*new_context_layer_shape)
    
        # Pass context layer through a dense layer and apply layer normalization
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)

        # Residual connection
        attention_output = self.LayerNorm(attention_output + hidden_states)
    
        # Return the final attention output
        return attention_output
    