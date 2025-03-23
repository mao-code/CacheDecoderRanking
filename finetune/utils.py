import random
from tqdm import tqdm

# Import Pyserini for retrieval.
from pyserini.search.lucene import LuceneSearcher

def prepare_training_samples_bce(corpus: dict, queries: dict, qrels: dict, hard_negative: bool = False, bm25_index=None, bm25_doc_ids=None, index_name="msmarco-passage"):
    """
    Creates training sample pairs: (query_text, doc_text, label) where label is 1.0 for relevant docs and 0.0 for negatives.
    For each positive, a negative is also added so that their numbers match.
    """
    training_samples = []
    all_doc_ids = list(corpus.keys())
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    
    # Precompute hard negatives if enabled.
    hard_negatives = {}
    if hard_negative and bm25_index is not None and bm25_doc_ids is not None:
        for qid in tqdm(qrels, desc="Precomputing hard negatives"):
            query_text = queries[qid]
            
            hits = searcher.search(query_text, k=10)
            doc_ids = [hit.docid for hit in hits]

            candidate_negatives = [doc_id for doc_id in doc_ids if doc_id not in qrels[qid]]
            if candidate_negatives:
                hard_negatives[qid] = candidate_negatives  # store list of negatives
            else:
                neg_doc_id = random.choice(all_doc_ids)
                while neg_doc_id in qrels[qid]:
                    neg_doc_id = random.choice(all_doc_ids)
                hard_negatives[qid] = [neg_doc_id]

    # For each query, add positive examples and for each positive add a corresponding negative.
    for qid, rel_docs in tqdm(qrels.items(), total=len(qrels), desc="Processing queries"):
        if qid not in queries:
            continue
        query_text = queries[qid]
        pos_doc_ids = [doc_id for doc_id, score in rel_docs.items() if score > 0]
        if not pos_doc_ids:
            continue
        for pos_doc_id in pos_doc_ids:
            pos_doc_text = corpus[pos_doc_id]['text']
            training_samples.append((query_text, pos_doc_text, 1.0))
            # For each positive, add a corresponding negative sample.
            if hard_negative and qid in hard_negatives:
                candidate_negatives = hard_negatives[qid]
                neg_doc_id = random.choice(candidate_negatives)
            else:
                neg_doc_id = random.choice(all_doc_ids)
                while neg_doc_id in rel_docs:
                    neg_doc_id = random.choice(all_doc_ids)
            neg_doc_text = corpus[neg_doc_id]['text']
            training_samples.append((query_text, neg_doc_text, 0.0))
    
    return training_samples

def subsample_dev_set(queries_dev: dict, qrels_dev: dict, sample_percentage: float = 0.05):
    dev_query_ids = list(queries_dev.keys())
    num_sample = max(1, int(len(dev_query_ids) * sample_percentage))
    sampled_ids = random.sample(dev_query_ids, num_sample)
    
    sampled_queries = {qid: queries_dev[qid] for qid in sampled_ids}
    sampled_qrels = {qid: qrels_dev[qid] for qid in sampled_ids if qid in qrels_dev}
    
    return sampled_queries, sampled_qrels