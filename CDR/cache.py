"""
cache.py

This module provides functions to compute and store the key/value (kv) caches
of documents using a scoring model (e.g., ScoringWrapper) and later use these
cached representations to score new queries.
"""

import torch
from transformers import PreTrainedTokenizer
import os
from transformers.cache_utils import Cache, DynamicCache
from tqdm import tqdm

from transformers.cache_utils import DynamicCache  # needed for type checking

def get_documents_cache(model, documents: dict, tokenizer: PreTrainedTokenizer, device: torch.device, batch_size: int = 8):
    """
    Process multiple documents in batch to compute and aggregate their kv-caches.
    """
    doc_ids = list(documents.keys())
    all_docs = [documents[doc_id] for doc_id in doc_ids]
    
    # Tokenize all documents at once. Add special tokens, pad, and truncate as needed.
    input_ids, token_type_ids, attention_mask = model.prepare_documents_input(all_docs, tokenizer)
    num_docs = input_ids.size(0)
    
    all_pkv = []  # list to store past_key_values from each batch
    
    # Process documents in batches
    for i in tqdm(range(0, num_docs, batch_size), desc="Computing document batch caches", leave=False):
        batch_input_ids = input_ids[i:i+batch_size].to(device)
        batch_attention_mask = attention_mask[i:i+batch_size].to(device)
        batch_token_type_ids = token_type_ids[i:i+batch_size].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                token_type_ids=batch_token_type_ids,
                use_cache=True,
                return_dict=True,
            )
        batch_pkv = outputs.get("past_key_values", None)
        if batch_pkv is None:
            raise ValueError("The model did not return past_key_values. Ensure that use_cache=True is working.")
        all_pkv.append(batch_pkv)
    
    # Aggregate caches from all batches
    # If the cache is a DynamicCache (has key_cache and value_cache), use its helper method:
    if hasattr(all_pkv[0], "key_cache") and hasattr(all_pkv[0], "value_cache"):
        aggregated_cache = DynamicCache.from_batch_splits(all_pkv)
    else:
        # Legacy tuple format: tuple of length num_layers,
        # where each element is a tuple: (key_tensor, value_tensor)
        num_layers = len(all_pkv[0])
        agg_keys = []
        agg_values = []
        for layer_idx in range(num_layers):
            keys = [batch_cache[layer_idx][0] for batch_cache in all_pkv]
            values = [batch_cache[layer_idx][1] for batch_cache in all_pkv]
            agg_keys.append(torch.cat(keys, dim=0))
            agg_values.append(torch.cat(values, dim=0))
        aggregated_cache = (tuple(agg_keys), tuple(agg_values))
    
    return aggregated_cache

def score_with_cache(model, kv_caches, query, tokenizer: PreTrainedTokenizer, device: torch.device, batch_size: int = 8):
    # Determine maximum sequence length from model configuration (if needed)
    if hasattr(model.config, "max_position_embeddings"):
        max_seq_len = model.config.max_position_embeddings
    elif hasattr(model.config, "n_positions"):
        max_seq_len = model.config.n_positions
    elif hasattr(model.config, "model_max_length"):
        max_seq_len = model.config.model_max_length
    else:
        raise ValueError("Model configuration does not specify a maximum sequence length.")

    # Split the kv_caches into smaller chunks, regardless of whether full_batch equals batch_size.
    if isinstance(kv_caches, DynamicCache):
        full_batch_size = len(kv_caches.key_cache)
        split_caches = kv_caches.batch_split(full_batch_size, batch_size)
    else:
        # Assume kv_caches is a tuple of key and value caches, each being a tuple of tensors.
        full_batch_size = kv_caches[0][0].size(0)
        split_caches = []
        for i in range(0, full_batch_size, batch_size):
            split_key = tuple(tensor[i : i + batch_size] for tensor in kv_caches[0])
            split_value = tuple(tensor[i : i + batch_size] for tensor in kv_caches[1])
            split_caches.append((split_key, split_value))

    logits_list = []
    # Process each split
    for cache_chunk in split_caches:
        if isinstance(cache_chunk, DynamicCache):
            current_batch = len(cache_chunk.key_cache)
        else:
            current_batch = cache_chunk[0][0].size(0)

        # Create a list of queries matching the current split batch size
        queries = [query] * current_batch
        input_ids, token_type_ids, attention_mask = model.prepare_query_input(queries, tokenizer)
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                past_key_values=cache_chunk,
                use_cache=True,
                return_dict=True
            )
        logits_list.append(outputs["logits"])

    return torch.cat(logits_list, dim=0)

def save_kv_cache(kv_cache_dict: dict, filename: str):
    """
    Save the kv-cache dictionary to a file.
    """
    torch.save(kv_cache_dict, filename)
    print(f"kv-cache dictionary saved to {filename}")

def load_kv_cache_from_file(filename: str) -> dict:
    """
    Load the kv-cache dictionary from a file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cache file {filename} does not exist.")
    kv_cache_dict = torch.load(filename, map_location="cpu")
    return kv_cache_dict

def build_and_save_documents_cache(model, documents: dict, tokenizer: PreTrainedTokenizer, device: torch.device, filename: str, batch_size: int = 8):
    """
    Convert the kv-cache dict to doc_id: kv_cache dict and save it to a file.
    """

    # List of document IDs and corresponding texts.
    doc_ids = list(documents.keys())
    all_docs = [documents[doc_id] for doc_id in doc_ids]
    kv_cache_dict = {}

    # Tokenize all documents at once.
    # This function is assumed to exist on your model. If not, see below for a dummy implementation.
    input_ids, token_type_ids, attention_mask = model.prepare_documents_input(all_docs, tokenizer)

    num_docs = input_ids.size(0)

    # Process documents in batches.
    for i in range(0, num_docs, batch_size):
        # Get the current batch of document IDs.
        batch_doc_ids = doc_ids[i:i+batch_size]
        # Slice and move inputs to the target device.
        batch_input_ids = input_ids[i:i+batch_size].to(device)
        batch_attention_mask = attention_mask[i:i+batch_size].to(device)
        batch_token_type_ids = token_type_ids[i:i+batch_size].to(device)

        # Run the forward pass with caching enabled.
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            token_type_ids=batch_token_type_ids,
            use_cache=True,
            return_dict=True,
        )
        batch_past_key_values = outputs.get("past_key_values", None)

        if batch_past_key_values is None:
            raise ValueError("The model did not return past_key_values. Ensure that use_cache=True is working.")

        # For each document in the batch, extract its cache from the batch dimension.
        # If the KV cache is in tuple format:
        if isinstance(batch_past_key_values, (tuple, list)):
            # Each element in the tuple corresponds to a layer and is a tuple (key, value)
            for j, doc_id in enumerate(batch_doc_ids):
                doc_cache = []
                for layer_cache in batch_past_key_values:
                    # layer_cache[0] and layer_cache[1] are tensors of shape
                    # (batch_size, num_heads, seq_length, head_dim)
                    key = layer_cache[0][j:j+1].cpu()
                    value = layer_cache[1][j:j+1].cpu()
                    doc_cache.append((key, value))
                kv_cache_dict[doc_id] = tuple(doc_cache)
        # Else, if using a DynamicCache instance:
        elif hasattr(batch_past_key_values, "key_cache") and hasattr(batch_past_key_values, "value_cache"):
            for j, doc_id in enumerate(batch_doc_ids):
                # Extract j-th element from each layer's cache.
                doc_key_cache = [k[j:j+1].cpu() for k in batch_past_key_values.key_cache]
                doc_value_cache = [v[j:j+1].cpu() for v in batch_past_key_values.value_cache]
                kv_cache_dict[doc_id] = {"key_cache": doc_key_cache, "value_cache": doc_value_cache}
        else:
            raise TypeError("Unrecognized KV cache format.")

    return kv_cache_dict


def score_with_cache_from_file(model, cache_filename: str, query, tokenizer: PreTrainedTokenizer, device: torch.device, batch_size: int = 8):
    pass