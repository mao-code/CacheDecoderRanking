import argparse
import logging
import wandb
from datetime import datetime
import math
import random

from transformers import AutoModel, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from dataset.document_ranking import DocumentRankingDataset
from finetune.utils import prepare_training_samples_infonce, prepare_training_samples_bce, subsample_dev_set
from utils import load_dataset, MainProcessFilter
from CDR.modeling import ScoringWrapper
from finetune.utils import load_prepared_samples, log_training_config

# # For BCE
# class DocumentRankingTrainer(Trainer):
#     def __init__(self, loss_fn, **kwargs):
#         super().__init__(**kwargs)
#         self.loss_fn = loss_fn

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs, return_dict=True)
#         logits = outputs["logits"].view(-1)
#         loss = self.loss_fn(logits, labels)
#         return (loss, outputs) if return_outputs else loss
    
# For InfoNCE
class DocumentRankingTrainer(Trainer):
    def __init__(self, n_per_query, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.n_per_query = n_per_query
        self.tokenizer = tokenizer

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    # def get_eval_dataloader(self, eval_dataset=None, **kwargs):
    #     if eval_dataset is None:
    #         eval_dataset = self.eval_dataset

    #     return DataLoader(
    #         eval_dataset,
    #         batch_size=self.args.per_device_eval_batch_size,
    #         shuffle=False,
    #         drop_last=False,
    #         collate_fn=self.data_collator,
    #         num_workers=self.args.dataloader_num_workers,
    #         pin_memory=self.args.dataloader_pin_memory,
    #     )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Not used in this loss
        outputs = model(**inputs, return_dict=True)
        logits = outputs["logits"].view(-1)
        
        group_size = 1 + self.n_per_query

        if len(logits) % group_size != 0:
            print(f"Batch size {len(logits)} must be a multiple of {group_size}")
            raise ValueError(f"Batch size {len(logits)} must be a multiple of {group_size}")
        
        N_groups = len(logits) // group_size
        
        logits = logits.view(N_groups, group_size)
        targets = torch.zeros(N_groups, dtype=torch.long, device=logits.device) # Target loss to 0
        
        # Temperature parameter, can be tuned.
        # BGE: 0.01 too small (loss stuck)
        tau = 0.05
        logits = logits / tau
        
        loss = nn.CrossEntropyLoss()(logits, targets)
        return (loss, outputs) if return_outputs else loss
    
def is_main_process():
    # If distributed is not available or not initialized, assume single-process (main)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a scoring model with token type embeddings and a score head.")    
    
    # Training settings.
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json",
                    help="Path to the DeepSpeed configuration file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--model_name", type=str, default="gpt2-medium", 
                        help="Pre-trained model name (e.g., gpt2-medium, facebook/opt-350m).")
    # Specify multiple datasets.
    parser.add_argument("--datasets", type=str, default="msmarco,nq-train,hotpotqa,fiqa",
                        help="Comma-separated list of dataset names to use for training (e.g., ms_marco,nq,hotpotqa,fiqa).")
    parser.add_argument("--samples_per_dataset", type=str, default="0,0,0,0",
                        help="Comma-separated list of number of training samples to use per dataset in the same order as --datasets. Use 0 to use all available samples.")
    # Accept a comma-separated list of index names corresponding to each dataset.
    parser.add_argument("--index_names", type=str,
                        default="msmarco-passage,beir-v1.0.0-nq.flat,beir-v1.0.0-hotpotqa.flat,beir-v1.0.0-fiqa.flat",
                        help="Comma-separated list of index names for each dataset, in the same order as --datasets.")
    parser.add_argument("--index_type", type=str, default="dense",
                        help="Type of index to use (dense or sparse).")
    parser.add_argument("--quey_encoder", type=str, default="BAAI/bge-base-en-v1.5", help="Query encoder model name for dense vectors.")
    parser.add_argument("--n_per_query", type=int, default=1,
                        help="Number of positive and negative samples to select per query.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing for memory efficiency.")
    
    # Evaluation settings.
    parser.add_argument("--sample_dev_percentage", type=float, default=0.1, help="Percentage of dev queries to sample for evaluation")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Per-device evaluation batch size")
    parser.add_argument("--eval_accumulation_steps", type=int, default=1, help="Evaluation accumulation steps")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--validate_every_n_steps", type=int, default=1000, help="Perform validation every n training steps")
    
    # Logging and checkpointing.
    parser.add_argument("--output_dir", type=str, default="./gfr_finetune_ckpts", help="Output directory for model checkpoints")
    parser.add_argument("--save_model_path", type=str, default="gfr_finetune_final", help="Directory to save the final best model")
    parser.add_argument("--run_name", type=str, default="", help="Run name for logging")
    parser.add_argument("--wandb_project", type=str, default="gfr_finetuning_document_ranking", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="your_group_name", help="Wandb entity name")
    parser.add_argument("--wandb_api_key", type=str, default="your_wandb_api_key", help="Wandb API key for logging")

    parser.add_argument("--use_prepared_data", action="store_true", 
                        help="If set, load pre-organized data from prepared files rather than computing hard negatives.")
    parser.add_argument("--prepared_data_files", type=str,
                        default="datasets/bge_data/split_1/msmarco_hn_train.jsonl,datasets/bge_data/split_1/nq.jsonl,datasets/bge_data/split/fever.json,datasets/bge_data/split/hotpotqa_pairs.json,datasets/bge_data/split/mr-tydi_english.jsonl",
                        help="Comma-separated list of file paths for the prepared dataset in the desired order.")
    parser.add_argument("--prepared_data_sample_counts", type=str,
                        default="0,0,0,0,0",
                        help="Comma-separated list of sample counts for each prepared dataset file in the same order. Use 0 to use all available samples.")

    args = parser.parse_args()

    group_size = 1 + args.n_per_query
    if args.per_device_train_batch_size % group_size != 0:
        raise ValueError(
            f"per_device_train_batch_size ({args.per_device_train_batch_size}) "
            f"must be a multiple of group_size ({group_size})"
        )
    if args.per_device_eval_batch_size % group_size != 0:
        raise ValueError(
            f"per_device_eval_batch_size ({args.per_device_eval_batch_size}) "
            f"must be a multiple of group_size ({group_size})"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(args.log_file, mode="w")
        ],
        force=True
    )
    logger = logging.getLogger()
    logger.addFilter(MainProcessFilter(args.local_rank))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name + "_" + now_datetime

    # Parse comma-separated lists.
    datasets_list = [d.strip() for d in args.datasets.split(",")]
    samples_list = [int(s.strip()) for s in args.samples_per_dataset.split(",")]
    index_names_list = [d.strip() for d in args.index_names.split(",")]
    if not (len(datasets_list) == len(samples_list) == len(index_names_list)):
        raise ValueError("The number of datasets, samples_per_dataset, and index_names must match.")

    # Initialize Wandb.
    # Use 
    # export WANDB_API_KEY="your_wandb_api_key"
    # export WANDB_PROJECT="cdr_finetuning_document_ranking"
    # export WANDB_ENTITY="nlp-maocode"

    # if is_main_process():
    #     wandb.login(key=args.wandb_api_key)
    #     wandb.init(
    #         project=args.wandb_project,
    #         entity=args.wandb_entity,
    #         name=run_name,
    #         config={
    #             "model_name": args.model_name,
    #             "datasets": datasets_list,
    #             "samples_per_dataset": samples_list,
    #             "index_names": index_names_list,
    #             "n_per_query": args.n_per_query,
    #             "learning_rate": args.lr,
    #             "per_device_train_batch_size": args.per_device_train_batch_size,
    #             "gradient_accumulation_steps": args.gradient_accumulation_steps,
    #             "num_train_epochs": args.num_train_epochs
    #         }
    #     )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    decoder = AutoModel.from_pretrained(args.model_name, config=config)
    scoring_model = ScoringWrapper(config, decoder)

    if args.gradient_checkpointing:
        # Enable gradient checkpointing for memory efficiency.
        # This is especially useful for large models.
        logger.info("Enabling gradient checkpointing...")
        scoring_model.decoder.gradient_checkpointing_enable()

    # Add special tokens and resize embeddings
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

    scoring_model.decoder.resize_token_embeddings(len(tokenizer))
    scoring_model.to(device)
    num_params = sum(p.numel() for p in scoring_model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    if args.use_prepared_data:
        # Parse file paths and sample counts.
        prepared_files = [f.strip() for f in args.prepared_data_files.split(",")]
        sample_counts = [int(s.strip()) for s in args.prepared_data_sample_counts.split(",")]
        if len(prepared_files) != len(sample_counts):
            raise ValueError("The number of prepared files and sample counts must match.")
        logger.info("Loading prepared training samples from files with specified sample counts...")
        all_training_samples = load_prepared_samples(prepared_files, sample_counts, args.n_per_query, logger)
        logger.info(f"Total mixed training samples (prepared): {len(all_training_samples)}")
        if len(all_training_samples) > 0:
            logger.info(f"First prepared training sample group: {all_training_samples[:1 + args.n_per_query]}")
    else:
        # Original: load and mix training samples from multiple datasets.
        datasets_list = [d.strip() for d in args.datasets.split(",")]
        samples_list = [int(s.strip()) for s in args.samples_per_dataset.split(",")]
        index_names_list = [d.strip() for d in args.index_names.split(",")]
        if not (len(datasets_list) == len(samples_list) == len(index_names_list)):
            raise ValueError("The number of datasets, samples_per_dataset, and index_names must match.")

        all_training_samples = []
        for dataset_name, sample_count, index_name in zip(datasets_list, samples_list, index_names_list):
            logger.info(f"Loading dataset: {dataset_name} (train split)")
            corpus_train, queries_train, qrels_train = load_dataset(logger, dataset_name, split="train")
            logger.info(f"Using index '{index_name}' for dataset: {dataset_name}")
            logger.info(f"Preparing training samples for dataset: {dataset_name}")

            if sample_count > 0 and sample_count < len(qrels_train):
                sampled_qids = random.sample(list(qrels_train.keys()), sample_count)
                qrels_train_sampled = {qid: qrels_train[qid] for qid in sampled_qids}
                queries_train_sampled = {qid: queries_train[qid] for qid in sampled_qids if qid in queries_train}
            else:
                qrels_train_sampled = qrels_train
                queries_train_sampled = queries_train
            logger.info(f"Number of queries in the sampled training set: {len(queries_train_sampled)}")

            samples = prepare_training_samples_infonce(
                corpus_train,
                queries_train_sampled,
                qrels_train_sampled,
                n_per_query=args.n_per_query,
                hard_negative=True,
                index_name=index_name,
                index_type=args.index_type,
                query_encoder=args.quey_encoder
            )
            logger.info(f"Total samples generated for {dataset_name}: {len(samples)}")
            all_training_samples.extend(samples)
        logger.info(f"Total mixed training samples: {len(all_training_samples)}")
        logger.info(f"First Training samples: {all_training_samples[:1 + args.n_per_query]}")
    
    # Create PyTorch Dataset for training.
    train_dataset = DocumentRankingDataset(all_training_samples, tokenizer, scoring_model)

    # ----------------------------------------------------------
    # Prepare a validation set.
    # ----------------------------------------------------------
    # dev_dataset = "msmarco"
    # dev_index = "msmarco-v1-passage.bge-base-en-v1.5"
    # logger.info(f"Loading dev set from primary dataset: {dev_dataset}")
    # corpus_dev, queries_dev, qrels_dev = load_dataset(logger, dev_dataset, split="dev")
    # sampled_queries_dev, sampled_qrels_dev = subsample_dev_set(
    #     queries_dev, qrels_dev, sample_percentage=args.sample_dev_percentage
    # )

    # validation_samples = prepare_training_samples_infonce(
    #     corpus_dev,
    #     sampled_queries_dev,
    #     sampled_qrels_dev,
    #     n_per_query=args.n_per_query,
    #     hard_negative=True,
    #     index_name=dev_index,
    #     index_type="dense",
    #     query_encoder="BAAI/bge-base-en-v1.5"
    # )
    # logger.info(f"Total samples generated for dev set: {len(validation_samples)}")
    # logger.info(f"First Validation samples: {validation_samples[0]}")

    # val_dataset = DocumentRankingDataset(validation_samples, tokenizer, scoring_model)

    total_training_steps = math.ceil(
        len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    ) * args.num_train_epochs
    warmup_steps = int(0.01 * total_training_steps)

    # Set up TrainingArguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        # per_device_eval_batch_size=args.per_device_eval_batch_size,
        # eval_accumulation_steps=args.eval_accumulation_steps,
        # eval_strategy="steps",
        # eval_steps=args.validate_every_n_steps,
        logging_dir="./logs_finetune",
        logging_steps=50,
        logging_first_step=True,
        save_steps=5000,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
        deepspeed=args.deepspeed_config,
    )

    log_training_config(training_args, logger)

    # Initialize our custom Trainer.
    # Define the data collator with padding
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = DocumentRankingTrainer(
        model=scoring_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        # eval_dataset=val_dataset,
        n_per_query=args.n_per_query,
        tokenizer=tokenizer
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # Train the model.
    trainer.train()

    if is_main_process():
        # Save the final model.
        trainer.save_model(args.save_model_path)
        logger.info("Training completed and best model saved.")
        wandb.finish()

if __name__ == "__main__":
    main()

    """
    model_name list:
    - openai-community/gpt2-medium
    - facebook/opt-350m
    - bigscience/bloom-560m
    - EleutherAI/pythia-410m
    - EleutherAI/pythia-1b

    Ensure per_device_train_batch_size is a multiple of (1 + n_per_query)

    Dense Index names: (FAISS)
    - msmarco-v1-passage.bge-base-en-v1.5 (MS MARCO by BGE)
    - beir-v1.0.0-nq.bge-base-en-v1.5 (NQ by BGE)
    - beir-v1.0.0-hotpotqa.bge-base-en-v1.5 (HotPotQA by BGE)
    - beir-v1.0.0-fever.bge-base-en-v1.5 (FEVER by BGE)
    # - beir-v1.0.0-quora.bge-base-en-v1.5 (Quora by BGE, only dev and test)

    # - beir-v1.0.0-fiqa.bge-base-en-v1.5 (FiQA by BGE)
    # - wikipedia-dpr-100w.dkrr-tqa (TriviaQA)
    
    Sparse Index names: (Lucene Standard Inverted Indexes)
    - msmarco-v1-passage
    - beir-v1.0.0-nq.flat
    - beir-v1.0.0-hotpotqa.flat
    - beir-v1.0.0-fiqa.flat

    If you want to load the dataset by yourself, add the following arguments:
    --datasets "msmarco,nq-train,fever,hotpotqa" \
    --samples_per_dataset "650000,100000,150000,50000" \
    --index_names "msmarco-v1-passage.bge-base-en-v1.5,beir-v1.0.0-nq.bge-base-en-v1.5,beir-v1.0.0-fever.bge-base-en-v1.5,beir-v1.0.0-hotpotqa.bge-base-en-v1.5" \
    --index_type "dense" \
    --quey_encoder "BAAI/bge-base-en-v1.5" \
    
    If you want to use wandb and don't want to set the environment variables, add the following arguments:
    --wandb_project "your_project_name" \
    --wandb_entity "your_group_name" \
    --wandb_api_key "your_wandb_api_key"
    
    Example usage:
    deepspeed --module finetune.finetune \
    --deepspeed_config deepspeed_config.json \
    --model_name "EleutherAI/pythia-410m" \
    --use_prepared_data \
    --prepared_data_files "datasets/bge_data/split_1/msmarco_hn_train.jsonl,datasets/bge_data/split_1/nq.jsonl,datasets/bge_data/split/fever.json,datasets/bge_data/split/hotpotqa_pairs.json,datasets/bge_data/split/mr-tydi_english.jsonl,datasets/bge_data/split/nli_simcse.json" \
    --prepared_data_sample_counts "0,0,0,0,0,0" \
    --n_per_query 15 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 8 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --sample_dev_percentage 0.1 \
    --per_device_eval_batch_size 16 \
    --eval_accumulation_steps 1 \
    --patience 10 \
    --validate_every_n_steps 100 \
    --output_dir "./cdr_finetune_ckpts_pythia_410m_bgedata" \
    --save_model_path "cdr_finetune_final_pythia_410m_bgedata" \
    --run_name "pythia_410m_mixed_bge_data" \
    --gradient_checkpointing
    """



