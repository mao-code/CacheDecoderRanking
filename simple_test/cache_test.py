import time
import torch
from transformers import LlamaTokenizer
from MLA_CDR import MLAConfig, MLAForSequenceScoring
from cache import move_cache_to_cpu, move_cache_to_gpu, get_documents_cache

def compute_cache_size(cache):
    """
    Compute the total size (in bytes) of a key/value cache.
    Supports both the legacy tuple format and DynamicCache (with key_cache/value_cache attributes).
    """
    total_bytes = 0
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        # DynamicCache format.
        for tensor in cache.key_cache:
            total_bytes += tensor.numel() * tensor.element_size()
        for tensor in cache.value_cache:
            total_bytes += tensor.numel() * tensor.element_size()
    elif isinstance(cache, tuple):
        # Legacy tuple format: tuple of layers, each layer is a tuple (key, value)
        for layer in cache:
            key, value = layer
            total_bytes += key.numel() * key.element_size()
            total_bytes += value.numel() * value.element_size()
    else:
        print("Warning: Unrecognized cache format.")
    return total_bytes

def measure_ttft_no_cache(model, full_input, device):
    """
    Measure time-to-first-token (TTFT) without caching (processing the full input at once).
    """
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(full_input, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
    return elapsed

def measure_ttft_cache(model, doc_tokens, query_tokens, device):
    """
    Measure TTFT with caching by first processing the document to obtain key/value cache,
    then processing the query using that cache.
    """
    with torch.no_grad():
        # Process document and get cached key/value states.
        doc_outputs = model(doc_tokens, use_cache=True)
        cache = doc_outputs.past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(query_tokens, past_key_values=cache, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
    return elapsed

def run_experiment(model, tokenizer, doc_text, query_text, batch_size, device):
    """
    For a given document and query, create batched inputs, warm up the model, and measure
    TTFT without and with cache.
    """
    # Tokenize a single sample.
    doc_tokens_single = tokenizer.encode(doc_text, return_tensors="pt").to(device)
    query_tokens_single = tokenizer.encode(query_text, return_tensors="pt").to(device)
    
    # Replicate the single sample along the batch dimension if needed.
    if batch_size > 1:
        doc_tokens = doc_tokens_single.repeat(batch_size, 1)
        query_tokens = query_tokens_single.repeat(batch_size, 1)
    else:
        doc_tokens = doc_tokens_single
        query_tokens = query_tokens_single

    # Warm-up runs to avoid startup overhead.
    with torch.no_grad():
        full_input = torch.cat((doc_tokens, query_tokens), dim=1)
        _ = model(full_input, use_cache=True)
        _ = model(doc_tokens, use_cache=True)
        doc_out = model(doc_tokens, use_cache=True)
        _ = model(query_tokens, past_key_values=doc_out.past_key_values, use_cache=True)

    # Measure TTFT without caching (full input).
    full_input = torch.cat((doc_tokens, query_tokens), dim=1)
    time_no_cache = measure_ttft_no_cache(model, full_input, device)
    # Measure TTFT with caching.
    time_cache = measure_ttft_cache(model, doc_tokens, query_tokens, device)
    
    return time_no_cache, time_cache

def test_inference_speed(model, tokenizer, device):
    """
    Test and print the inference performance (TTFT) with and without caching
    for various batch sizes.
    """
    print("\n=== Inference Speed Test ===")
    doc_text = "word " * 512  # ~512 tokens
    query_text = "query " * 15  # ~15 tokens
    batch_sizes = [1, 8, 16]
    
    header = "{:<12}{:<20}{:<20}{:<20}".format("BatchSize", "NoCache (ms)", "WithCache (ms)", "Improvement (%)")
    print(header)
    print("-" * len(header))
    
    for batch_size in batch_sizes:
        time_no_cache, time_cache = run_experiment(model, tokenizer, doc_text, query_text, batch_size, device)
        improvement = (time_no_cache - time_cache) / time_no_cache * 100
        print("{:<12}{:<20.3f}{:<20.3f}{:<20.2f}".format(
            batch_size, time_no_cache * 1000, time_cache * 1000, improvement
        ))

def test_cache_movement(model, tokenizer, device):
    """
    Compute the document cache using the helper function from cache.py,
    print the time taken to compute it, its size in MB,
    and measure the overhead for moving the cache between GPU and CPU.
    """
    print("\n=== Cache Movement and Size Test ===")
    # Create a dummy document.
    doc_text = "word " * 512
    documents = {"doc1": doc_text}
    
    # Compute the document cache.
    start = time.time()
    cache = get_documents_cache(model, documents, tokenizer, device, batch_size=1)
    elapsed_cache_comp = time.time() - start
    print("Time to compute document cache: {:.3f} seconds".format(elapsed_cache_comp))
    
    # Compute the size of the cache.
    cache_size_bytes = compute_cache_size(cache)
    cache_size_mb = cache_size_bytes / (1024 * 1024)
    print("Cache size: {:.3f} MB".format(cache_size_mb))
    
    # Measure time to move cache to CPU.
    start = time.time()
    cache_cpu = move_cache_to_cpu(cache)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_to_cpu = time.time() - start
    print("Time to move cache to CPU: {:.3f} seconds".format(elapsed_to_cpu))
    
    # Measure time to move cache back to GPU.
    start = time.time()
    cache_gpu = move_cache_to_gpu(cache_cpu, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_to_gpu = time.time() - start
    print("Time to move cache to GPU: {:.3f} seconds".format(elapsed_to_gpu))

def main():
    # Set device: GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the tokenizer.
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})
    
    # Initialize the model.
    print("Initializing MLA model for pretraining...")
    config = MLAConfig(vocab_size=len(tokenizer))
    model = MLAForSequenceScoring(config)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    model.eval()  # Set model to evaluation mode.
    
    # Run inference performance tests.
    test_inference_speed(model, tokenizer, device)
    
    # Test cache movement overhead and cache size.
    test_cache_movement(model, tokenizer, device)

if __name__ == "__main__":
    main()
