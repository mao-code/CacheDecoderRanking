import os
import logging
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
import torch

def load_dataset(logger, dataset: str, split: str):
    """Loads a BEIR dataset and prefixes ids with the dataset name."""
    out_dir = "datasets"
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        logger.info(f"Dataset '{dataset}' not found locally. Downloading...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, out_dir)
    else:
        logger.info(f"Dataset '{dataset}' found locally. Skipping download.")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    
    # Prefix each doc_id in corpus, and update queries and qrels accordingly.
    # new_corpus = {f"{dataset}_{doc_id}": content for doc_id, content in corpus.items()}
    # new_queries = {f"{dataset}_{qid}": query for qid, query in queries.items()}
    # new_qrels = {f"{dataset}_{qid}": {f"{dataset}_{doc_id}": score for doc_id, score in rels.items()}
    #              for qid, rels in qrels.items()}
    
    return corpus, queries, qrels

import os
import json

def load_json_file(file_path):
    """
    Load a JSON or JSONL file.
    """
    ext = os.path.splitext(file_path)[1]
    if ext == '.jsonl':
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    elif ext == '.json':
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            try:
                # Try to parse as a single JSON structure
                return json.loads(content)
            except json.decoder.JSONDecodeError:
                # Fallback: treat as JSON Lines if multiple JSON objects are found
                data = []
                for line in content.splitlines():
                    if line.strip():
                        data.append(json.loads(line))
                return data
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

class MainProcessFilter(logging.Filter):
    def filter(self, record):
        # Allow logging only if not in a distributed setup or if this is rank 0.
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            return True
        return False