import torch

def collate_fn(batch):
    # Initialize dictionaries to store the batched data
    batched_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "source_len": []
    }
    # Iterate through the batch and append each item to the corresponding list
    for item in batch:
        for key in batched_data:
            # Ensure each item is a tensor and append it to the list
            batched_data[key].append(torch.tensor(item[key]))
    # Stack tensors, making sure all elements are tensors and have the same shape
    for key in batched_data:
        batched_data[key] = torch.stack(batched_data[key])

    return batched_data
