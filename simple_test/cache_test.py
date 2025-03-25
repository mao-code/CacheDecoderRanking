import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model and tokenizer
    model_name = "EleutherAI/pythia-410m"
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True).to(device)
    model.eval()  # set evaluation mode
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dummy inputs:
    # Document: roughly 512 tokens by repeating a word
    # Query: roughly 15 tokens by repeating another word
    doc_text = "word " * 512
    query_text = "query " * 15
    
    # Tokenize inputs (the tokenizer converts strings to token ids)
    doc_tokens = tokenizer.encode(doc_text, return_tensors="pt").to(device)
    query_tokens = tokenizer.encode(query_text, return_tensors="pt").to(device)
    
    # Warm-up runs (to avoid initial setup overhead affecting timings)
    with torch.no_grad():
        _ = model(torch.cat((doc_tokens, query_tokens), dim=1), use_cache=True)
        doc_outputs = model(doc_tokens, use_cache=True)
        _ = model(query_tokens, past_key_values=doc_outputs.past_key_values, use_cache=True)
    
    # -----------------------------
    # Measure TTFT without caching:
    # Process the full input (document + query) at once.
    full_input = torch.cat((doc_tokens, query_tokens), dim=1)
    with torch.no_grad():
        # If using GPU, synchronize before and after timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(full_input, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        full_time = time.time() - start_time

    # -----------------------------
    # Measure TTFT with caching:
    # Process the document first to obtain the cached key/value states,
    # then process the query using the cached states.
    with torch.no_grad():
        # Compute and store the cache from the document.
        doc_outputs = model(doc_tokens, use_cache=True)
        cache = doc_outputs.past_key_values

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(query_tokens, past_key_values=cache, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        cached_time = time.time() - start_time

    # Calculate the improvement rate
    improvement_rate = (full_time - cached_time) / full_time * 100

    # Print the results
    print("Time without cache: {:.3f} ms".format(full_time * 1000))
    print("Time with cache:    {:.3f} ms".format(cached_time * 1000))
    print("Time improvement rate: {:.2f}%".format(improvement_rate))

if __name__ == "__main__":
    main()