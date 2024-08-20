from transformers import PretrainedConfig, AutoConfig


class PALMConfig(PretrainedConfig):
    '''Define configuration class for the partial 
        attention language model architecture'''
    # Specify the model type for identification in the broader framework
    model_type = 'PALM'

    def __init__(
        self,
        base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", # Default base model
        fixed_source_length=100, # Preset fixed length for source input
        vocab_size=128258, # Vocabulary size, defining number of tokens available
        hidden_size=4096, # Size of hidden layers in the model
        num_hidden_layers=32, # Number of hidden layers in the model
        num_attention_heads=32, # Number of attention heads for multi-head attention mechanism
        intermediate_size=11008, # Size of the intermediate feed-forward layer in transformer blocks
        hidden_act="silu", # Activation function used in hidden layers
        hidden_dropout_prob=0.1, # Dropout probability for hidden layers
        attention_probs_dropout_prob=0.1, # Dropout probability for attention probabilities
        max_position_embeddings=2048, # Maximum number of position embeddings (sequence length)
        initializer_range=0.02, # Range for weight initialization
        layer_norm_eps=1e-5, # Epsilon parameter for layer normalization to avoid division by zero
        pad_token_id=128257, # Token ID used for padding sequences
        bos_token_id=1, # Token ID for the beginning of a sequence
        eos_token_id=2, # Token ID for the end of a sequence
        **kwargs
    ):
        # Call the parent class (PretrainedConfig) constructor with specific token IDs
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        # Assign initialization parameters to instance variables
        self.base_model_name = base_model_name
        self.fixed_source_length = fixed_source_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # Load base model config
        base_config = AutoConfig.from_pretrained(base_model_name)

        # Handle rope_scaling attribute if it exists in the base model configuration
        if hasattr(base_config, 'rope_scaling'):
            if isinstance(base_config.rope_scaling, dict):
                # Initialize a new dictionary for rope_scaling, keeping only the 'type' and 'factor' keys
                new_rope_scaling = {}
                if 'type' in base_config.rope_scaling:
                    new_rope_scaling['type'] = base_config.rope_scaling['type']
                else:
                    new_rope_scaling['type'] = 'linear'

                if 'factor' in base_config.rope_scaling:
                    new_rope_scaling['factor'] = base_config.rope_scaling['factor']
                else:
                    new_rope_scaling['factor'] = base_config.rope_scaling.get('factor', 1.0)

                self.rope_scaling = new_rope_scaling
            else:
                # If rope_scaling is not a dict, set it to a default linear scaling value
                self.rope_scaling = {
                    'type': 'linear',
                    'factor': 1.0
                }

        # Copy attributes from base_config that are not already set
        for key, value in base_config.to_dict().items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    # Method to create an instance of PALMConfig from a pre-trained model's configuration
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        return config
    