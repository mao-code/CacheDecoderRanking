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

def get_document_kv_cache(model, document: str, tokenizer: PreTrainedTokenizer, device: torch.device):
    """
    Compute the kv-cache for a single document using the provided model.
    
    Args:
        model: The scoring model (an instance of ScoringWrapper) that contains a decoder and token type embeddings.
        document (str): The document text.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        device (torch.device): The torch device to run computation on.
        
    Returns:
        past_key_values: The kv-cache (typically a tuple of tuples for each layer) returned by the model's decoder.
    """
    # Tokenize the document (adding special tokens if needed)
    input_ids = tokenizer.encode(document, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], device=device)  # shape: (1, seq_len)
    
    # For documents, token type is 0.
    token_type_ids = torch.zeros_like(input_ids, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    
    # Compute the input embeddings using the model's decoder and token type embeddings.
    token_embeds = model.decoder.get_input_embeddings()(input_ids)
    token_type_embeds = model.token_type_embeddings(token_type_ids)
    inputs_embeds = token_embeds + token_type_embeds
    
    # Pass through the decoder with use_cache=True to get the kv-cache.
    outputs = model.decoder(
        input_ids=None,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.get("past_key_values", None)
    return past_key_values


def build_documents_kv_cache(model, documents: dict, tokenizer: PreTrainedTokenizer, device: torch.device, batch_size: int = 8):
    """
    Process multiple documents in batch to compute their kv-caches.
    
    Args:
        model: The scoring model (instance of ScoringWrapper).
        documents (dict): A dictionary mapping doc_id (str) to document text (str).
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        device (torch.device): Device to run computation on.
        batch_size (int): Batch size for processing.
    
    Returns:
        kv_cache_dict (dict): Dictionary mapping each doc_id to its computed kv-cache.
    """
    doc_ids = list(documents.keys())
    kv_cache_dict = {}
    all_docs = [documents[doc_id] for doc_id in doc_ids]
    
    # Tokenize all documents at once. We add special tokens, pad and truncate as needed.
    encodings = tokenizer(all_docs, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    # For documents, token type IDs are all zeros.
    token_type_ids = torch.zeros_like(input_ids, device=device)
    
    # Compute input embeddings.
    token_embeds = model.decoder.get_input_embeddings()(input_ids)
    token_type_embeds = model.token_type_embeddings(token_type_ids)
    inputs_embeds = token_embeds + token_type_embeds
    
    num_docs = input_ids.size(0)
    # Process in batches
    for i in range(0, num_docs, batch_size):
        batch_inputs_embeds = inputs_embeds[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]
        batch_token_type_ids = token_type_ids[i:i+batch_size]
        
        outputs = model.decoder(
            input_ids=None,
            attention_mask=batch_attention_mask,
            token_type_ids=batch_token_type_ids,
            inputs_embeds=batch_inputs_embeds,
            use_cache=True,
            return_dict=True,
        )
        batch_past_key_values = outputs.get("past_key_values", None)
        if batch_past_key_values is None:
            raise ValueError("The model did not return past_key_values. Ensure that use_cache=True is working.")
        
        batch_size_actual = batch_attention_mask.size(0)
        # Split the batched kv-cache along the batch dimension.
        for j in range(batch_size_actual):
            # For each layer, extract the j-th element from both key and value.
            kv_cache_doc = tuple(
                (layer_cache[0][j:j+1].detach().cpu(), layer_cache[1][j:j+1].detach().cpu())
                for layer_cache in batch_past_key_values
            )
            doc_index = i + j
            doc_id = doc_ids[doc_index]
            kv_cache_dict[doc_id] = kv_cache_doc

    return kv_cache_dict


def save_kv_cache(kv_cache_dict: dict, filename: str):
    """
    Save the kv-cache dictionary to a file.
    
    Args:
        kv_cache_dict (dict): Dictionary mapping doc_id to kv-cache.
        filename (str): The filename to save the kv-cache dictionary.
    """
    torch.save(kv_cache_dict, filename)
    print(f"kv-cache dictionary saved to {filename}")


def load_kv_cache(filename: str) -> dict:
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


def score_with_kv_cache(model, kv_cache, query: str, tokenizer: PreTrainedTokenizer, device: torch.device):
    """
    Given a document's kv-cache and a new query, compute the relevance score.
    
    The function tokenizes the query, appends the special [SCORE] token,
    and calls the model with the provided kv-cache via the past_key_values argument.
    
    Args:
        model: The scoring model (instance of ScoringWrapper).
        kv_cache: The kv-cache for the document (as computed by get_document_kv_cache/build_documents_kv_cache).
        query (str): The query text.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        device (torch.device): Device to run computation on.
    
    Returns:
        score (float): The computed relevance score.
    """
    # Tokenize query without adding special tokens.
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    # Append the special [SCORE] token
    score_id = tokenizer.convert_tokens_to_ids("[SCORE]")
    input_ids = query_ids + [score_id]
    input_ids = torch.tensor([input_ids], device=device)
    # For query tokens, token type is 1.
    token_type_ids = torch.ones_like(input_ids, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    
    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        past_key_values=kv_cache,
        use_cache=True,
        return_dict=True,
    )
    # The scoring wrapper extracts the [SCORE] token's hidden state and passes it through a linear layer.
    score = outputs["logits"].item()
    return score


def score_batch_with_kv_cache(model, kv_caches: list, queries: list, tokenizer: PreTrainedTokenizer, device: torch.device):
    """
    Batch version: Given a list of document kv-caches and corresponding queries,
    compute the relevance scores for all documents.
    
    This function tokenizes queries in batch (appending the [SCORE] token to each),
    concatenates the individual kv-caches along the batch dimension, and performs a
    batched forward pass.
    
    Args:
        model: The scoring model (instance of ScoringWrapper).
        kv_caches (list): List of kv-cache objects (each from a document). Each should be a tuple
                          (with length equal to the number of layers) where each element is a tuple (key, value)
                          of shape (1, num_heads, seq_len, head_dim).
        queries (list): List of query strings (one per document).
        tokenizer (PreTrainedTokenizer): The tokenizer.
        device (torch.device): Device to run computation on.
    
    Returns:
        scores (list of float): List of computed relevance scores, one per document.
    """
    if len(kv_caches) != len(queries):
        raise ValueError("The length of kv_caches and queries must be equal.")

    # Append the special [SCORE] token to each query.
    score_token = "[SCORE]"
    queries_encoded = [tokenizer.encode(query, add_special_tokens=False) for query in queries]
    score_id = tokenizer.convert_tokens_to_ids(score_token)
    # Append score token id to each query's token list.
    queries_encoded = [q_ids + [score_id] for q_ids in queries_encoded]

    # Pad the queries so they all have the same length.
    encodings = tokenizer.pad({"input_ids": queries_encoded}, padding=True, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    # For queries, token type IDs are 1.
    token_type_ids = torch.ones_like(input_ids, device=device)

    # Combine (stack) the kv_caches along the batch dimension.
    # Each kv_cache is a tuple for each layer; we need to stack keys and values across documents.
    num_layers = len(kv_caches[0])
    batched_past = []
    for layer in range(num_layers):
        keys = []
        values = []
        for kv in kv_caches:
            key, value = kv[layer]
            keys.append(key)
            values.append(value)
        batched_key = torch.cat(keys, dim=0).to(device)
        batched_value = torch.cat(values, dim=0).to(device)
        batched_past.append((batched_key, batched_value))
    batched_past = tuple(batched_past)

    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        past_key_values=batched_past,
        use_cache=True,
        return_dict=True,
    )
    # The model returns logits for each token; the final token is the [SCORE] token.
    logits = outputs["logits"]  # shape: (batch_size, seq_length)
    
    scores = []
    for i in range(logits.size(0)):
        # Determine the actual length of the sequence (i.e. count of non-padded tokens).
        seq_len = int(attention_mask[i].sum().item())
        score = logits[i, seq_len - 1].item()
        scores.append(score)
    return scores