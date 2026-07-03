import os
import json
import datasets
import argparse
import multiprocessing
from dataclasses import dataclass
from itertools import chain
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
from config.config import QwenTrainerConf, get_logger


logger = get_logger()


def setup(conf: QwenTrainerConf):
    def make_dir(path: str):
        if not os.path.exists(path=path):
            os.system(f'mkdir -p {path}')
            os.system(f'chmod 777 {path}')
    for d in [conf.output_dir, conf.logging_dir]:
        make_dir(d)


def main():
    parser = argparse.ArgumentParser('Qwen LLM trainer')
    parser.add_argument('--checkpoint', type=str, default=None, 
                      help='Path to the checkpoint to resume training from')
    args = parser.parse_args()
    
    conf = QwenTrainerConf()
    num_proc = multiprocessing.cpu_count()

    logger.info('Setting up')
    setup(conf=conf)

    logger.info('Loading dataset')
    with open('./data/data_files_qwen.json', 'r') as f:
        data_files = json.load(f)

    data = datasets.load_dataset('text', data_files=data_files, num_proc=num_proc)
    train_data = data['train']
    eval_data = data['val']

    logger.info(f'Loading tokenizer: {conf.pretrained_model_name}')
    tokenizer = AutoTokenizer.from_pretrained(conf.pretrained_model_name)
    
    # Đảm bảo tokenizer có pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"The max length for the tokenizer is: {conf.max_seq_length}")

    def tokenize_function(examples):
        """Tokenize raw text directly with Qwen's tokenizer"""
        # Tokenize all texts
        tokenized_examples = tokenizer(
            examples["text"],
            truncation=True,
            max_length=conf.max_seq_length,
            return_special_tokens_mask=True,
            padding="max_length",
        )
        return tokenized_examples

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder
        if total_length >= conf.max_seq_length:
            total_length = (total_length // conf.max_seq_length) * conf.max_seq_length
            
        # Split by chunks of max_len
        result = {
            k: [t[i : i + conf.max_seq_length] for i in range(0, total_length, conf.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    logger.info('Tokenize train data')
    train_tokenized_datasets = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        num_proc=num_proc,
        desc="Tokenizing train texts",
    )

    train_tokenized_datasets = train_tokenized_datasets.map(
        group_texts, 
        batched=True, 
        num_proc=num_proc,
        desc="Grouping train texts",
    )
    train_dataset = train_tokenized_datasets.shuffle(seed=conf.seed)
    logger.info(f"The train dataset contains {len(train_dataset)} examples")

    logger.info('Tokenize eval data')
    eval_tokenized_datasets = eval_data.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        num_proc=num_proc,
        desc="Tokenizing eval texts",
    )
    eval_dataset = eval_tokenized_datasets.map(
        group_texts, 
        batched=True, 
        num_proc=num_proc,
        desc="Grouping eval texts",
    )
    logger.info(f"The eval dataset contains {len(eval_dataset)} examples")

    logger.info(f'Load model checkpoint: {conf.pretrained_model_name}')
    if args.checkpoint:
        logger.info(f'Resuming from checkpoint: {args.checkpoint}')
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(conf.pretrained_model_name)
    
    # Đảm bảo model được cấu hình đúng cho causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Đặt False cho Causal LM (không phải masked LM)
        pad_to_multiple_of=8  # Giúp tối ưu hóa cho mixed precision training
    )

    training_args = TrainingArguments(
        run_name=conf.run_name,
        data_seed=conf.data_seed,
        dataloader_num_workers=conf.dataloader_num_workers,
        do_train=conf.do_train,
        do_eval=conf.do_eval,
        evaluation_strategy=conf.evaluation_strategy,
        eval_steps=conf.eval_steps,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        logging_dir=conf.logging_dir,
        learning_rate=conf.learning_rate,
        log_level=conf.log_level,
        logging_strategy=conf.logging_strategy,
        logging_steps=conf.logging_steps,
        lr_scheduler_type=conf.lr_scheduler_type,
        output_dir=conf.output_dir,
        max_steps=conf.max_steps,
        per_device_train_batch_size=conf.per_device_train_batch_size,
        per_device_eval_batch_size=conf.per_device_eval_batch_size,
        prediction_loss_only=conf.prediction_loss_only,
        report_to=conf.report_to,
        save_strategy=conf.save_strategy,
        save_steps=conf.save_steps,
        save_total_limit=conf.save_total_limit,
        seed=conf.seed,
        warmup_ratio=conf.warmup_ratio,
        warmup_steps=conf.warmup_steps,
        weight_decay=conf.weight_decay,
        gradient_checkpointing=True,
        torch_compile=False,  # Có thể bật nếu PyTorch > 2.0 và muốn tối ưu tốc độ
        bf16=conf.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info('Training')
    trainer.train(resume_from_checkpoint=args.checkpoint)
    
    # Lưu model và tokenizer khi training xong
    logger.info(f"Saving final model to {conf.output_dir}")
    trainer.save_model(conf.output_dir)
    tokenizer.save_pretrained(conf.output_dir)
    
    logger.info("Training completed successfully")

if __name__=='__main__':
    import wandb
    wandb.login(key='wandb_key')
    main()