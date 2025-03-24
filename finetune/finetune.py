import argparse
import logging
import wandb
from datetime import datetime
import math
import random

from transformers import AutoModel, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, EarlyStoppingCallback
import torch
import torch.nn as nn

from dataset.document_ranking import DocumentRankingDataset
from finetune.utils import prepare_training_samples_bce, subsample_dev_set
from utils import load_dataset
from CDR.modeling import ScoringWrapper

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

class DocumentRankingTrainer(Trainer):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, return_dict=True)
        logits = outputs["logits"].view(-1)
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
    def train(self, *args, **kwargs):
        print(f"Model: {type(self.model)}")
        print(f"Optimizer: {type(self.optimizer)}")
        return super().train(*args, **kwargs)
    
def main():
    parser = argparse.ArgumentParser(description="Fine-tune a scoring model with token type embeddings and a score head.")    
    
    # Training settings.
    parser.add_argument("--model_name", type=str, default="gpt2-medium", 
                        help="Pre-trained model name (e.g., gpt2-medium, facebook/opt-350m).")
    # Specify multiple datasets.
    parser.add_argument("--datasets", type=str, default="msmarco,nq-train,hotpotqa,fiqa",
                        help="Comma-separated list of dataset names to use for training (e.g., ms_marco,nq,hotpotqa,fiqa).")
    parser.add_argument("--samples_per_dataset", type=str, default="1000,3000,2000,1000",
                        help="Comma-separated list of number of training samples to use per dataset in the same order as --datasets. Use 0 to use all available samples.")
    # Accept a comma-separated list of index names corresponding to each dataset.
    parser.add_argument("--index_names", type=str,
                        default="msmarco-passage,beir-v1.0.0-nq.flat,beir-v1.0.0-hotpotqa.flat,beir-v1.0.0-fiqa.flat",
                        help="Comma-separated list of index names for each dataset, in the same order as --datasets.")
    parser.add_argument("--n_per_query", type=int, default=1,
                        help="Number of positive and negative samples to select per query.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    
    # Evaluation settings.
    parser.add_argument("--sample_dev_percentage", type=float, default=0.05, help="Percentage of dev queries to sample for evaluation")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Per-device evaluation batch size")
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
    args = parser.parse_args()

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
    wandb.login(key=args.wandb_api_key)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_name": args.model_name,
            "datasets": datasets_list,
            "samples_per_dataset": samples_list,
            "index_names": index_names_list,
            "n_per_query": args.n_per_query,
            "learning_rate": args.lr,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs
        }
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    decoder = AutoModel.from_pretrained(args.model_name, config=config)
    scoring_model = ScoringWrapper(decoder, config)

    # Add special tokens and resize embeddings
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

    scoring_model.decoder.resize_token_embeddings(len(tokenizer))
    scoring_model.to(device)
    num_params = sum(p.numel() for p in scoring_model.parameters())
    print(f"Number of parameters: {num_params}")

    # ----------------------------------------------------------
    # Load and mix training samples from multiple datasets.
    # ----------------------------------------------------------
    all_training_samples = []
    for dataset_name, sample_count, index_name in zip(datasets_list, samples_list, index_names_list):
        logging.info(f"Loading dataset: {dataset_name} (train split)")
        corpus_train, queries_train, qrels_train = load_dataset(dataset_name, split="train")
        logging.info(f"Using index '{index_name}' for dataset: {dataset_name}")
        logging.info(f"Preparing training samples for dataset: {dataset_name}")

        # If sample_count > 0 and lower than available samples, randomly select a subset.
        if sample_count > 0 and sample_count < len(qrels_train):
            # Sample a subset of query IDs from qrels_train.
            sampled_qids = random.sample(list(qrels_train.keys()), sample_count)
            
            # Filter the qrels and queries dictionaries to only include the sampled query IDs.
            qrels_train_sampled = {qid: qrels_train[qid] for qid in sampled_qids}
            queries_train_sampled = {qid: queries_train[qid] for qid in sampled_qids if qid in queries_train}
        else:
            qrels_train_sampled = qrels_train
            queries_train_sampled = queries_train
        logging.info(f"Number of queries in the sampled training set: {len(queries_train_sampled)}")

        samples = prepare_training_samples_bce(
            corpus_train,
            queries_train_sampled,
            qrels_train_sampled,
            n_per_query=args.n_per_query,
            hard_negative=True,
            index_name=index_name
        )
        logging.info(f"Total samples generated for {dataset_name}: {len(samples)}")
        all_training_samples.extend(samples)
    
    # Shuffle the final mixed training samples.
    random.shuffle(all_training_samples)
    logging.info(f"Total mixed training samples: {len(all_training_samples)}")
    logging.info(f"First Training samples: {all_training_samples[0]}")
    
    # Create PyTorch Dataset for training.
    train_dataset = DocumentRankingDataset(all_training_samples, tokenizer, scoring_model)

    # ----------------------------------------------------------
    # Prepare a dev set.
    # Here we use the dev split from the first (primary) dataset.
    # ----------------------------------------------------------
    primary_dataset = datasets_list[0]
    primary_index = index_names_list[0]
    logging.info(f"Loading dev set from primary dataset: {primary_dataset}")
    corpus_dev, queries_dev, qrels_dev = load_dataset(primary_dataset, split="dev")
    sampled_queries_dev, sampled_qrels_dev = subsample_dev_set(
        corpus_dev, qrels_dev, sample_percentage=args.sample_dev_percentage
    )
    validation_samples = prepare_training_samples_bce(
        corpus_dev,
        sampled_queries_dev,
        sampled_qrels_dev,
        n_per_query=args.n_per_query,
        hard_negative=True,
        index_name=primary_index
    )
    logging.info(f"Total samples generated for dev set: {len(validation_samples)}")
    logging.info(f"First Validation samples: {validation_samples[0]}")
    val_dataset = DocumentRankingDataset(validation_samples, tokenizer, scoring_model)

    total_training_steps = math.ceil(
        len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    ) * args.num_train_epochs
    warmup_steps = int(0.1 * total_training_steps)

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
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.validate_every_n_steps,
        logging_dir="./logs_finetune",
        logging_steps=1,
        logging_first_step=True,
        save_steps=40,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False
    )

    loss_fn = nn.BCEWithLogitsLoss()
    # Initialize our custom Trainer.
    trainer = DocumentRankingTrainer(
        model=scoring_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss_fn=loss_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # Train the model.
    trainer.train()

    # Save the final model.
    trainer.save_model(args.save_model_path)
    logging.info("Training completed and best model saved.")
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

    Example usage:
    python -m finetune.finetune \
    --model_name "EleutherAI/pythia-410m" \
    --datasets "msmarco,nq-train,hotpotqa,fiqa" \
    --samples_per_dataset "64,64,64,64" \
    --index_names "msmarco-v1-passage,beir-v1.0.0-nq.flat,beir-v1.0.0-hotpotqa.flat,beir-v1.0.0-fiqa.flat" \
    --n_per_query 5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --sample_dev_percentage 0.1 \
    --per_device_eval_batch_size 2 \
    --eval_accumulation_steps 1 \
    --patience 3 \
    --validate_every_n_steps 20 \
    --output_dir "./cdr_finetune_ckpts_pythia_410m_mixed" \
    --save_model_path "cdr_finetune_final_pythia_410m_mixed" \
    --run_name "pythia_410m_mixed" \
    --wandb_project "cdr_finetuning_document_ranking" \
    --wandb_entity "nlp-maocode" \
    --wandb_api_key "your_wandb_api_key"
    """



