import torch

device = torch.device("cpu") # Set to CPU, adjust as needed


def generate_text(model, tokenizer, prompt, max_length=500, temperature=0.7, top_p=0.9, model_type=None):
    try:
        # Encode the input prompt into token IDs using the tokenizer, converting it to a tensor
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"input_ids shape: {input_ids.shape}") # print the shape of the input tensor
        
        # Check if the model is of type "PALM" and if it has a method for creating a bidirectional attention mask
        if model_type == "PALM" and hasattr(model, 'create_bidirectional_attention_mask'):
            attention_mask = model.create_bidirectional_attention_mask(input_ids) # Create the attention mask
            print(f"attention_mask shape: {attention_mask.shape}") # print shape of the attention mask

        with torch.no_grad(): # Disable gradient computation for inference
            if model_type == "PALM":
                # Generate text using the PALM model with custom settings for attention mask and generation parameters
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True, # Enable sampling for more diverse text generation
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:  # Fallback for models like GPT-2 XL
                output = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id # Set padding token to the EOS token ID
                )
        
        # Decode generated tokens back into text, skipping special tokens like <PAD> or <EOS>
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text # Return the generated text
    
    except Exception as e: # Catch any exceptions that occur during the generation process
        print(f"Error in generate_text: {str(e)}") # Print error message
        print(f"Model device: {next(model.parameters()).device}") # Print device of the model
        print(f"Input device: {input_ids.device}") # Print device of the input tensor
        # return f"Error occurred: {str(e)}" # Return None in case of an error
        return None # Return None in case of an error
