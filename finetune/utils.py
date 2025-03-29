import random
from tqdm import tqdm

# Import Pyserini for retrieval.
from pyserini.search.lucene import LuceneSearcher, FaissSearcher

def prepare_training_samples_infonce(
    corpus: dict,
    queries: dict,
    qrels: dict,
    n_per_query: int = 7,
    hard_negative: bool = False,
    index_name: str = "msmarco-v1-passage",
    index_type: str = "dense"
):
    training_samples = []
    all_doc_ids = list(corpus.keys())

    if index_type == "dense":
        searcher = FaissSearcher.from_prebuilt_index(index_name)
    elif index_type == "sparse":
        searcher = LuceneSearcher.from_prebuilt_index(index_name)
    else:  
        raise ValueError(f"Unsupported index type: {index_type}. Use 'dense' or 'sparse'.")

    hard_negatives = {}
    for qid in tqdm(qrels, desc=f"Precomputing hard negatives using index {index_name}"):
        if qid not in queries:
            continue
        query = queries[qid]

        if isinstance(query, dict):
            query = query['text']

        hits = searcher.search(query, k=100)
        doc_ids = [hit.docid for hit in hits]
        candidate_negatives = [doc_id for doc_id in doc_ids if doc_id not in qrels[qid]]
        if candidate_negatives:
            hard_negatives[qid] = candidate_negatives
        else:
            neg_doc_id = random.choice(all_doc_ids)
            while neg_doc_id in qrels[qid]:
                neg_doc_id = random.choice(all_doc_ids)
            hard_negatives[qid] = [neg_doc_id]

    for qid, rel_docs in tqdm(qrels.items(), total=len(qrels), desc="Processing queries"):
        if qid not in queries:
            continue
        query_text = queries[qid]

        if isinstance(query_text, dict):
            query_text = query_text['text']

        pos_doc_ids = [doc_id for doc_id, score in rel_docs.items() if score > 0]
        if not pos_doc_ids:
            continue
        for pos_doc_id in pos_doc_ids:
            # Select n_per_query negatives
            if hard_negative and qid in hard_negatives:
                candidate_negatives = hard_negatives[qid]
                if len(candidate_negatives) >= n_per_query:
                    neg_doc_ids = random.sample(candidate_negatives, n_per_query)
                else:
                    neg_doc_ids = candidate_negatives.copy()
                    while len(neg_doc_ids) < n_per_query:
                        neg_doc_ids.append(random.choice(candidate_negatives))
            else:
                neg_doc_ids = []
                while len(neg_doc_ids) < n_per_query:
                    neg_doc_id = random.choice(all_doc_ids)
                    if neg_doc_id not in rel_docs and neg_doc_id not in neg_doc_ids:
                        neg_doc_ids.append(neg_doc_id)
            # Add positive sample
            training_samples.append({
                'query_text': query_text,
                'doc_text': corpus[pos_doc_id]['text'],
                'label': 1.0
            })
            # Add negative samples
            for neg_doc_id in neg_doc_ids:
                training_samples.append({
                    'query_text': query_text,
                    'doc_text': corpus[neg_doc_id]['text'],
                    'label': 0.0
                })
    return training_samples

def prepare_training_samples_bce(
    corpus: dict,
    queries: dict,
    qrels: dict,
    n_per_query: int = 5,
    hard_negative: bool = False,
    index_name: str = "msmarco-v1-passage"
):
    """
    Creates training sample tuples: (query_text, doc_text, label)
    where label is 1.0 for relevant docs and 0.0 for negatives.

    For each query, this function:
      - Selects up to n_per_query positive documents (shuffled if more than n available).
      - Selects up to n_per_query negative documents. If hard_negative is enabled, it uses the provided index
        (via LuceneSearcher) to retrieve candidate negatives; otherwise, negatives are sampled randomly.
    """
    training_samples = []
    all_doc_ids = list(corpus.keys())
    searcher = LuceneSearcher.from_prebuilt_index(index_name)

    # Precompute hard negatives for each query.
    hard_negatives = {}
    for qid in tqdm(qrels, desc=f"Precomputing hard negatives using index {index_name}"):
        if qid not in queries:
            continue
        
        query = queries[qid]
        # Handle different query formats
        if isinstance(query, str):
            query_text = query
        elif isinstance(query, dict) and 'text' in query:
            query_text = query['text']  # Extract the query text
        else:
            print(f"Skipping qid {qid}: Invalid query format {query}")
            continue

        hits = searcher.search(query_text, k=100)
        doc_ids = [hit.docid for hit in hits]
        candidate_negatives = [doc_id for doc_id in doc_ids if doc_id not in qrels[qid]]
        if candidate_negatives:
            hard_negatives[qid] = candidate_negatives
        else:
            # Fallback: sample one negative document randomly if no candidate is found.
            neg_doc_id = random.choice(all_doc_ids)
            while neg_doc_id in qrels[qid]:
                neg_doc_id = random.choice(all_doc_ids)
            hard_negatives[qid] = [neg_doc_id]

    # Process each query.
    for qid, rel_docs in tqdm(qrels.items(), total=len(qrels), desc="Processing queries"):
        if qid not in queries:
            continue
        query_text = queries[qid]
        # Get all positive doc IDs for this query.
        pos_doc_ids = [doc_id for doc_id, score in rel_docs.items() if score > 0]
        if not pos_doc_ids:
            continue
        # random.shuffle(pos_doc_ids)

        pos_n_per_query = min(n_per_query, len(pos_doc_ids))
        pos_samples = pos_doc_ids[:pos_n_per_query]
        for pos_doc_id in pos_samples:
            training_samples.append((query_text, corpus[pos_doc_id]['text'], 1.0))
        
        # Sample negative docs.
        if hard_negative and qid in hard_negatives:
            candidate_negatives = hard_negatives[qid]
            if len(candidate_negatives) >= n_per_query:
                neg_samples = random.sample(candidate_negatives, n_per_query)
            else:
                neg_samples = candidate_negatives.copy()
                while len(neg_samples) < n_per_query:
                    neg_samples.append(random.choice(candidate_negatives))
        else:
            neg_samples = []
            while len(neg_samples) < n_per_query:
                neg_doc_id = random.choice(all_doc_ids)
                if neg_doc_id not in rel_docs and neg_doc_id not in neg_samples:
                    neg_samples.append(neg_doc_id)
        for neg_doc_id in neg_samples:
            training_samples.append((query_text, corpus[neg_doc_id]['text'], 0.0))
    
    return training_samples

def subsample_dev_set(queries_dev: dict, qrels_dev: dict, sample_percentage: float = 0.05):
    dev_query_ids = list(queries_dev.keys())
    num_sample = max(1, int(len(dev_query_ids) * sample_percentage))
    sampled_ids = random.sample(dev_query_ids, num_sample)
    
    sampled_queries = {qid: queries_dev[qid] for qid in sampled_ids}
    sampled_qrels = {qid: qrels_dev[qid] for qid in sampled_ids if qid in qrels_dev}
    
    return sampled_queries, sampled_qrels