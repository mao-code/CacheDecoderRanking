import argparse
import logging
import wandb
from datetime import datetime
import math

from transformers import AutoModel, AutoTokenizer, AutoConfig, TrainingArguments, Trainer, EarlyStoppingCallback
import torch

from dataset.document_ranking import DocumentRankingDataset
from finetune.utils import prepare_training_samples_bce, subsample_dev_set
from utils import load_dataset
from CDR.modeling import ScoringWrapper

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
    
def main():
    parser = argparse.ArgumentParser(description="Fine-tune a scoring model with token type embeddings and a score head.")    
    
    # Training settings.
    parser.add_argument("--model_name", type=str, default="gpt2-medium", 
                        help="Pre-trained model name (e.g., gpt2-medium, facebook/opt-350m).")
    parser.add_argument("--dataset_name", type=str, default="ms_marco", 
                        help="Dataset name (e.g., ms_marco).")
    parser.add_argument("--index_name", type=str, default="msmarco-passage", help="Name of the pre-built index.")
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

    # Initialize Wandb.
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config={
        "model_name": args.model_name,
        "dataset_name": args.dataset,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
    })

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name, config=config)
    scoring_model = ScoringWrapper(base_model, config)

    # Add special tokens and resize embeddings
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

    scoring_model.base_model.resize_token_embeddings(len(tokenizer))

    scoring_model.to(device)
    num_params = sum(p.numel() for p in scoring_model.parameters())
    print(f"Number of parameters: {num_params}")

    # Load datasets.
    training_samples = []
    dev_data = {}   # { dataset: (corpus, queries, qrels) }
    test_data = {}  # { dataset: (corpus, queries, qrels) }

    dataset = args.dataset_name
    logging.info(f"Loading dataset: {dataset} (train split)")
    corpus_train, queries_train, qrels_train = load_dataset(dataset, split="train")
    training_samples.extend(
        prepare_training_samples_bce(
            corpus_train, queries_train, qrels_train,
            hard_negative=True,
            index_name=args.index_name
        )
    )
            
    logging.info(f"Loading dataset: {dataset} (dev split)")
    corpus_dev, queries_dev, qrels_dev = load_dataset(dataset, split="dev")
    dev_data[dataset] = (corpus_dev, queries_dev, qrels_dev)
    
    logging.info(f"Loading dataset: {dataset} (test split)")
    corpus_test, queries_test, qrels_test = load_dataset(dataset, split="test")
    test_data[dataset] = (corpus_test, queries_test, qrels_test)

    logging.info(f"Total training samples: {len(training_samples)}")

    sampled_queries_dev, sampled_qrels_dev = subsample_dev_set(corpus_dev, qrels_dev
    , sample_percentage=args.sample_dev_percentage)
    validation_samples = prepare_training_samples_bce(corpus_dev, sampled_queries_dev, sampled_qrels_dev, hard_negative=True, index_name=args.index_name)

    # Create PyTorch Datasets.
    train_dataset = DocumentRankingDataset(training_samples, tokenizer, scoring_model)
    val_dataset = DocumentRankingDataset(validation_samples, tokenizer, scoring_model)

    total_training_steps = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)) * args.num_train_epochs
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
        logging_steps=50,
        logging_first_step=True,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name,

        remove_unused_columns=False
    )

    # Initialize our custom Trainer.
    trainer = DocumentRankingTrainer(
        model=scoring_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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



