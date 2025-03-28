import os
import argparse
import logging
import torch
from tqdm import tqdm
import time
from tabulate import tabulate  # For pretty-printing the comparison table
import csv

from utils import load_dataset
from evaluation.utils import beir_evaluate, beir_evaluate_custom

# Import Pyserini for retrieval.
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher

from CDR.modeling import ScoringWrapper
from transformers import AutoModel, AutoTokenizer, AutoConfig
from safetensors.torch import load_file

# Import CrossEncoder for standard models.
from sentence_transformers import CrossEncoder

# Import caching functions from our cache module
from CDR.cache import get_documents_cache, score_with_cache

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Test script for reranking using sparse or dense retrieval + various reranker models"
    )
    parser.add_argument("--dataset", type=str, default="msmarco", help="Dataset to use for testing (e.g., msmarco)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., test)")
    parser.add_argument("--index_name", type=str, default="msmarco-v1-passage", 
                        help="Specific index name to use; if None, use default based on retrieval_type")
    parser.add_argument("--models", type=str, nargs='+', required=True, 
                        help="List of models to test, in the form 'type:checkpoint' (e.g., 'cdr:/path/to/cdr', 'standard:cross-encoder/ms-marco-MiniLM-L-12-v2')")
    parser.add_argument("--cdr_decoder", type=str, default="EleutherAI/pythia-410m", help="Base decoder model to use for CDR models")
    parser.add_argument("--log_file", type=str, default="rerank_results.log", help="File to log the evaluation results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for reranking")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top BM25 results to retrieve per query")
    parser.add_argument("--k_values", type=int, nargs='+', default=[1, 5, 10],
                        help="List of k values for computing evaluation metrics (e.g., NDCG, MAP)")
    parser.add_argument("--retrieval_type", type=str, default="sparse", choices=["sparse", "dense"],
                        help="Type of retrieval to use")

    args = parser.parse_args()

    # Set up logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, mode="w")
        ],
        force=True
    )
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the dataset
    logger.info(f"Loading dataset '{args.dataset}' with split '{args.split}'...")
    corpus, queries, qrels = load_dataset(args.dataset, split=args.split)

    # Initialize the searcher based on retrieval type
    if args.index_name is not None:
        index_name = args.index_name
    elif args.retrieval_type == "sparse":
        index_name = 'msmarco-v1-passage'
    elif args.retrieval_type == "dense":
        index_name = 'msmarco-v1-passage.tct_colbert-v2-hnp'
    else:
        raise ValueError("Invalid retrieval_type")
    
    if args.retrieval_type == "sparse":
        searcher = LuceneSearcher.from_prebuilt_index(index_name)
    elif args.retrieval_type == "dense":
        searcher = FaissSearcher.from_prebuilt_index(index_name)
    else:
        raise ValueError("Invalid retrieval_type")

    # List to store evaluation results for all models
    all_model_results = []

    # Process each model specified in --models
    for model_spec in args.models:
        torch.cuda.empty_cache()
        model_type, model_checkpoint = model_spec.split(":", 1)
        base_decoer_model = args.cdr_decoder
        base_model_id = f"{model_type}_{model_checkpoint.replace('/', '_')}"
        logger.info(f"Evaluating model: {base_model_id}")

        if model_type == "cdr":
            logger.info("Loading tokenizer for CDR...")
            # Load the original tokenizer for CDR
            tokenizer = AutoTokenizer.from_pretrained(args.cdr_decoder)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if tokenizer.sep_token is None:
                tokenizer.add_special_tokens({"sep_token": "[SEP]"})
            if "[SCORE]" not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

            logger.info("Loading CDR reranker model...")    
            config = AutoConfig.from_pretrained(model_checkpoint)
            decoder = AutoModel.from_pretrained(args.cdr_decoder)
            model = ScoringWrapper(config, decoder)
            model.resize_token_embeddings(len(tokenizer))

            # Load weights from safetensors
            state_dict = load_file(f"{model_checkpoint}/model.safetensors")
            model.load_state_dict(state_dict)

            model.to(device) 
            model.eval()

            # Prepare accumulators for plain and cache variants
            total_inference_time_plain = 0.0
            total_score_time_plain = 0.0
            total_docs_processed_plain = 0
            total_inference_time_cache = 0.0
            total_score_time_cache = 0.0
            total_docs_processed_cache = 0

            # Dictionaries to hold reranked results for each method
            reranked_results_plain = {}
            reranked_results_cache = {}
        elif model_type == "standard":
            logger.info(f"Loading standard CrossEncoder model: {model_checkpoint}")
            model = CrossEncoder(model_checkpoint, device=device, automodel_args={"torch_dtype": "auto"}, trust_remote_code=True)
            # For standard models, we use a single reranking result dictionary.
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be 'cdr' or 'standard'.")

        # Timing variables for standard models and for hit rate measurement
        total_inference_time = 0.0
        total_score_time = 0.0
        total_docs_processed = 0
        total_hits_rate = 0.0

        logger.info(f"Using {args.retrieval_type} to retrieve top documents and reranking...")
        # Process each query
        for qid, query_text in tqdm(queries.items(), desc="Processing queries"):
            # Retrieve top documents
            hits = searcher.search(query_text, k=args.top_k)
            candidate_doc_ids = [hit.docid for hit in hits]
            candidate_docs = [corpus[doc_id]['text'] for doc_id in candidate_doc_ids]

            relevant_doc_ids = list(qrels.get(qid, {}).keys())
            common_docs = set(relevant_doc_ids).intersection(set(candidate_doc_ids))
            hits_rate = len(common_docs) / len(relevant_doc_ids) if len(relevant_doc_ids) > 0 else 0
            total_hits_rate += hits_rate

            if model_type == "cdr":
                # ---------- Plain (non-cached) CDR evaluation ----------
                scores_plain = []
                for i in range(0, len(candidate_docs), args.batch_size):
                    batch_docs = candidate_docs[i:i + args.batch_size]
                    batch_queries = [query_text] * len(batch_docs)
                    input_ids, token_type_ids, attention_mask = model.prepare_input(batch_docs, batch_queries, tokenizer)
                    input_ids = input_ids.to(device)
                    token_type_ids = token_type_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                    elapsed = time.time() - start_time
                    total_inference_time_plain += elapsed
                    total_score_time_plain += elapsed
                    total_docs_processed_plain += len(batch_docs)
                    batch_scores = output["logits"].squeeze(-1).tolist()
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    scores_plain.extend(batch_scores)
                reranked_results_plain[qid] = {doc_id: score for doc_id, score in zip(candidate_doc_ids, scores_plain)}

                torch.cuda.empty_cache()

                # ---------- CDR with Cache evaluation ----------
                # First, compute the candidate document cache (this time is NOT counted in ranking time)
                candidate_doc_dict = {doc_id: corpus[doc_id]['text'] for doc_id in candidate_doc_ids}
                candidate_kv_caches = get_documents_cache(model, candidate_doc_dict, tokenizer, device, batch_size=args.batch_size)
                # Now, measure ranking time for scoring using the cached representations.
                start_time = time.time()
                with torch.no_grad():
                    scores_cache, score_time = score_with_cache(
                        model,
                        candidate_kv_caches,
                        query_text,
                        tokenizer,
                        device,
                        batch_size=args.batch_size
                    )
                elapsed = time.time() - start_time
                total_inference_time_cache += elapsed
                total_score_time_cache += score_time
                total_docs_processed_cache += len(candidate_doc_ids)
                reranked_results_cache[qid] = {doc_id: score for doc_id, score in zip(candidate_doc_ids, scores_cache)}
            elif model_type == "standard":
                # Standard CrossEncoder reranking
                pairs = [(query_text, doc) for doc in candidate_docs]
                start_time = time.time()
                scores = model.predict(pairs, batch_size=args.batch_size)
                elapsed = time.time() - start_time
                total_inference_time += elapsed
                total_score_time += elapsed
                total_docs_processed += len(candidate_docs)
                if not isinstance(scores, list):
                    scores = scores.tolist()
                reranked_results = {qid: {doc_id: score for doc_id, score in zip(candidate_doc_ids, scores)}}
            else:
                reranked_results = {}

        # --- Evaluate and log results ---
        if model_type == "cdr":
            # Evaluate plain CDR reranking
            logger.info("Evaluating plain CDR reranking results...")
            ndcg_plain, _map_plain, recall_plain, precision_plain = beir_evaluate(qrels, reranked_results_plain, args.k_values, ignore_identical_ids=True)
            mrr_plain = beir_evaluate_custom(qrels, reranked_results_plain, args.k_values, metric="mrr")
            top_k_accuracy_plain = beir_evaluate_custom(qrels, reranked_results_plain, args.k_values, metric="top_k_accuracy")
            avg_inference_time_ms_plain = (total_inference_time_plain / total_docs_processed_plain) * 1000 if total_docs_processed_plain > 0 else 0
            avg_score_time_ms_plain = (total_score_time_plain / total_docs_processed_plain) * 1000 if total_docs_processed_plain > 0 else 0
            throughput_plain = total_docs_processed_plain / total_inference_time_plain if total_inference_time_plain > 0 else 0

            # Evaluate CDR with Cache reranking
            logger.info("Evaluating CDR with cache reranking results...")
            ndcg_cache, _map_cache, recall_cache, precision_cache = beir_evaluate(qrels, reranked_results_cache, args.k_values, ignore_identical_ids=True)
            mrr_cache = beir_evaluate_custom(qrels, reranked_results_cache, args.k_values, metric="mrr")
            top_k_accuracy_cache = beir_evaluate_custom(qrels, reranked_results_cache, args.k_values, metric="top_k_accuracy")
            avg_inference_time_ms_cache = (total_inference_time_cache / total_docs_processed_cache) * 1000 if total_docs_processed_cache > 0 else 0
            avg_score_time_ms_cache = (total_score_time_cache / total_docs_processed_cache) * 1000 if total_docs_processed_cache > 0 else 0
            throughput_cache = total_docs_processed_cache / total_inference_time_cache if total_inference_time_cache > 0 else 0

            avg_hits_rate = total_hits_rate / len(queries)

            # Save results for plain CDR variant
            model_results_plain = {
                "model_id": base_model_id + "_plain",
                "ndcg": ndcg_plain,
                "map": _map_plain,
                "recall": recall_plain,
                "precision": precision_plain,
                "mrr": mrr_plain,
                "top_k_accuracy": top_k_accuracy_plain,
                "avg_inference_time_ms": avg_inference_time_ms_plain,
                "avg_score_time_ms": avg_score_time_ms_plain,
                "throughput_docs_per_sec": throughput_plain
            }
            all_model_results.append(model_results_plain)

            # Save results for CDR with cache variant
            model_results_cache = {
                "model_id": base_model_id + "_cache",
                "ndcg": ndcg_cache,
                "map": _map_cache,
                "recall": recall_cache,
                "precision": precision_cache,
                "mrr": mrr_cache,
                "top_k_accuracy": top_k_accuracy_cache,
                "avg_inference_time_ms": avg_inference_time_ms_cache,
                "avg_score_time_ms": avg_score_time_ms_cache,
                "throughput_docs_per_sec": throughput_cache
            }
            all_model_results.append(model_results_cache)

            logger.info(f"Evaluation Metrics for {base_model_id}_plain:")
            logger.info(f"Average Hits Rate: {avg_hits_rate}")
            logger.info(f"NDCG: {ndcg_plain}")
            logger.info(f"MAP: {_map_plain}")
            logger.info(f"Recall: {recall_plain}")
            logger.info(f"Precision: {precision_plain}")
            logger.info(f"MRR: {mrr_plain}")
            logger.info(f"Top_K_Accuracy: {top_k_accuracy_plain}")
            logger.info(f"Avg Inference Time (ms): {avg_inference_time_ms_plain:.2f}")
            logger.info(f"Throughput (docs/sec): {throughput_plain:.2f}")
            logger.info("=" * 20)

            logger.info(f"Evaluation Metrics for {base_model_id}_cache:")
            logger.info(f"Average Hits Rate: {avg_hits_rate}")
            logger.info(f"NDCG: {ndcg_cache}")
            logger.info(f"MAP: {_map_cache}")
            logger.info(f"Recall: {recall_cache}")
            logger.info(f"Precision: {precision_cache}")
            logger.info(f"MRR: {mrr_cache}")
            logger.info(f"Top_K_Accuracy: {top_k_accuracy_cache}")
            logger.info(f"Avg Inference Time (ms): {avg_inference_time_ms_cache:.2f}")
            logger.info(f"Throughput (docs/sec): {throughput_cache:.2f}")
            logger.info("=" * 20)

        elif model_type == "standard":
            ndcg, _map, recall, precision = beir_evaluate(qrels, reranked_results, args.k_values, ignore_identical_ids=True)
            mrr = beir_evaluate_custom(qrels, reranked_results, args.k_values, metric="mrr")
            top_k_accuracy = beir_evaluate_custom(qrels, reranked_results, args.k_values, metric="top_k_accuracy")
            avg_inference_time_ms = (total_inference_time / total_docs_processed) * 1000 if total_docs_processed > 0 else 0
            avg_score_time_ms = (total_score_time / total_docs_processed) * 1000 if total_docs_processed > 0 else 0
            throughput_docs_per_sec = total_docs_processed / total_inference_time if total_inference_time > 0 else 0

            avg_hits_rate = total_hits_rate / len(queries)

            model_results = {
                "model_id": base_model_id,
                "ndcg": ndcg,
                "map": _map,
                "recall": recall,
                "precision": precision,
                "mrr": mrr,
                "top_k_accuracy": top_k_accuracy,
                "avg_inference_time_ms": avg_inference_time_ms,
                "avg_score_time_ms": avg_score_time_ms,
                "throughput_docs_per_sec": throughput_docs_per_sec
            }
            all_model_results.append(model_results)

            logger.info(f"Evaluation Metrics for {base_model_id}:")
            logger.info(f"Average Hits Rate: {avg_hits_rate}")
            logger.info(f"NDCG: {ndcg}")
            logger.info(f"MAP: {_map}")
            logger.info(f"Recall: {recall}")
            logger.info(f"Precision: {precision}")
            logger.info(f"MRR: {mrr}")
            logger.info(f"Top_K_Accuracy: {top_k_accuracy}")
            logger.info(f"Avg Inference Time (ms): {avg_inference_time_ms:.2f}")
            logger.info(f"Throughput (docs/sec): {throughput_docs_per_sec:.2f}")
            logger.info("=" * 20)

    # Log comparison table for all models
    logger.info("Comparison of all models:")
    comparison_table = []
    for result in all_model_results:
        row = [
            result["model_id"],
            result["ndcg"].get("NDCG@10", "-"),
            result["map"].get("MAP@10", "-"),
            result["recall"].get("Recall@10", "-"),
            result["precision"].get("P@10", "-"),
            result["mrr"].get("MRR@10", "-"),
            result["top_k_accuracy"].get("Top_K_Accuracy@10", "-"),
            result["avg_inference_time_ms"],
            result["throughput_docs_per_sec"]
        ]
        comparison_table.append(row)

    headers = ["Model", "NDCG@10", "MAP@10", "Recall@10", "Precision@10", "MRR@10", "Top_K_Accuracy@10", "Avg Inference Time (ms)", "Throughput (docs/sec)"]
    logger.info("\n" + tabulate(comparison_table, headers=headers, tablefmt="grid"))

    # --- Save the comparison table to a CSV file ---
    csv_file = "rerank_comparison_table.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(comparison_table)
    logger.info(f"Comparison table saved to {csv_file}")

if __name__ == "__main__":
    main()

"""
Standard Models:
standard:cross-encoder/ms-marco-MiniLM-L-12-v2 standard:mixedbread-ai/mxbai-rerank-large-v1 standard:jinaai/jina-reranker-v2-base-multilingual standard:BAAI/bge-reranker-v2-m3

Example usage:
python -m evaluation.rerank \
  --dataset msmarco \
  --index_name msmarco-v1-passage \
  --split test \
  --models cdr:./cdr_finetune_final_pythia_410m_mixed standard:cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --cdr_decoder EleutherAI/pythia-410m \
  --log_file rerank_results.log \
  --batch_size 8 \
  --top_k 100 \
  --k_values 10 \
  --retrieval_type sparse
"""