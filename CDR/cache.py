"""
cache.py

This module provides functions to compute and store the key/value (kv) caches
of documents using a scoring model (e.g., ScoringWrapper) and later use these
cached representations to score new queries.

Functions:
    - get_document_kv_cache: Compute the kv-cache for a single document.
    - build_documents_kv_cache: Process multiple documents in batch and build a dict mapping doc_id to kv-cache.
    - save_kv_cache: Save the kv-cache dictionary to a file.
    - load_kv_cache: Load the kv-cache dictionary from a file.
    - score_with_kv_cache: Given a single document’s kv-cache and a new query, compute the relevance score.
    - score_batch_with_kv_cache: Batched version for scoring multiple documents (each with its own kv-cache) with queries.
"""

import torch
from transformers import PreTrainedTokenizer
import os
from transformers.cache_utils import Cache, DynamicCache

def get_documents_cache(model, documents: dict, tokenizer: PreTrainedTokenizer, device: torch.device, batch_size: int = 8):
    """
    Process multiple documents in batch to compute their kv-caches.
    """
    doc_ids = list(documents.keys())
    kv_cache_dict = {}
    all_docs = [documents[doc_id] for doc_id in doc_ids]
    
    # Tokenize all documents at once. We add special tokens, pad and truncate as needed.
    input_ids, token_type_ids, attention_mask = model.prepare_documents_input(all_docs, tokenizer)

    num_docs = input_ids.size(0)

    # Process in batches
    for i in range(0, num_docs, batch_size):
        batch_inputs_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]
        batch_token_type_ids = token_type_ids[i:i+batch_size]
        
        outputs = model(
            input_ids=batch_inputs_ids,
            attention_mask=batch_attention_mask,
            token_type_ids=batch_token_type_ids,
            use_cache=True,
            return_dict=True,
        )
        batch_past_key_values = outputs.get("past_key_values", None)

        if batch_past_key_values is None:
            raise ValueError("The model did not return past_key_values. Ensure that use_cache=True is working.")
        if not isinstance(batch_past_key_values, DynamicCache):
            raise ValueError("Expected past_key_values to be DynamicCache")

    return batch_past_key_values

def save_kv_cache(kv_cache_dict: dict, filename: str):
    """
    Save the kv-cache dictionary to a file.
    
    Args:
        kv_cache_dict (dict): Dictionary mapping doc_id to kv-cache.
        filename (str): The filename to save the kv-cache dictionary.
    """
    torch.save(kv_cache_dict, filename)
    print(f"kv-cache dictionary saved to {filename}")


def load_kv_cache_from_file(filename: str) -> dict:
    """
    Load the kv-cache dictionary from a file.
    
    Args:
        filename (str): The filename from which to load the kv-cache dictionary.
    
    Returns:
        kv_cache_dict (dict): Dictionary mapping doc_id to kv-cache.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cache file {filename} does not exist.")
    kv_cache_dict = torch.load(filename, map_location="cpu")
    return kv_cache_dict

def build_and_save_documents_cache(model, documents: dict, tokenizer: PreTrainedTokenizer, device: torch.device, filename: str, batch_size: int = 8):
    pass

def score_with_cache(model, kv_caches, query, tokenizer: PreTrainedTokenizer, device: torch.device, batch_size: int = 8):

    if hasattr(model.config, "max_position_embeddings"):
        max_seq_len = model.config.max_position_embeddings
    elif hasattr(model.config, "n_positions"):
        max_seq_len = model.config.n_positions
    elif hasattr(model.config, "model_max_length"):
        max_seq_len = model.config.model_max_length
    else:
        raise ValueError("Model configuration does not specify a maximum sequence length.")

    queries = [query for i in len(kv_caches)]

    input_ids, token_type_ids, attention_mask = model.prepare_query_input(queries, tokenizer)

    scores = torch.empty(0, device=input_ids.device)
    num_queries = input_ids.size(0) 
    for i in range(0, num_queries, batch_size):
        batch_inputs_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]
        batch_token_type_ids = token_type_ids[i:i+batch_size]
        batch_kv_caches = kv_caches[i:i+batch_size]
        
        outputs = model(
            input_ids=batch_inputs_ids,
            token_type_ids=batch_token_type_ids,
            attention_mask=batch_attention_mask,
            past_key_values=batch_kv_caches,
            use_cache=True,
            return_dict=True
        )

        logits = outputs["logits"] # (batch_size, )
        scores = torch.cat((scores, logits), dim=0)

    return scores