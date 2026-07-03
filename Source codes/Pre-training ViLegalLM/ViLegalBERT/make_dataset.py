import os
import json
import multiprocessing
from itertools import chain

import datasets
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config.config import get_logger

logger = get_logger()


def get_paths(data_dir: str) -> list[str]:
    
    return [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p.endswith('.txt')]


def split_dataset(paths: list[str], test_size: int = 10000, random_state: int = 42):
    
    train_paths, test_paths = train_test_split(
        paths,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    train_paths, val_paths = train_test_split(
        train_paths,
        test_size=test_size,
        shuffle=False
    )
    
    return {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }
    

# def make_save_dataset():
#     with open('./data/data_files.json', 'r') as f:
#         data_files = json.load(f)
        
#     data = datasets.load_dataset('text', data_files=data_files)
#     train_data = data['train']
#     eval_data = data['val']
#     test_data = data['test']
    
#     logger.info(f'Loading tokenizer: vinai/phobert-base-v2')
#     tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    
#     tokenizer.model_max_length = 256
#     num_proc = multiprocessing.cpu_count()
#     logger.info(f"The max length for the tokenizer is: {tokenizer.model_max_length}")
    
#     def tokenize_text(examples): return tokenizer(
#         examples['text'],
#         return_special_tokens_mask=True,
#         truncation=True,
#         max_length=tokenizer.model_max_length
#     )
#     def group_texts(examples):
#         # Concatenate all texts.
#         concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#         total_length = len(concatenated_examples[list(examples.keys())[0]])
#         # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#         # customize this part to your needs.
#         if total_length >= tokenizer.model_max_length:
#             total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
#         # Split by chunks of max_len.
#         result = {
#             k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
#             for k, t in concatenated_examples.items()
#         }
#         return result
    
#     train_tokenized_datasets = train_data.map(
#         tokenize_text,
#         batched=True,
#         remove_columns=['text']
#     )
    
#     train_tokenized_datasets = train_tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
#     train_dataset = train_tokenized_datasets.shuffle(seed=42)
#     train_dataset.save_to_disk("./data/cache/train")
#     logger.info(f"The dataset contains in total {len(train_tokenized_datasets)*tokenizer.model_max_length} tokens")

#     eval_tokenized_datasets = eval_data.map(
#         tokenize_text,
#         batch_size=True,
#         remove_columns=['text']
#     )
#     eval_dataset = eval_tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
#     eval_dataset.save_to_disk('./data/cache/val')
    
#     test_tokenized_datasets = test_data.map(
#         tokenize_text,
#         batched=True,
#         remove_columns=['text']
#     )
#     test_dataset = test_tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
#     test_dataset.save_to_disk('./data/cache/test')

    
if __name__ == '__main__':
    print('get paths')
    paths = get_paths('./data/tokenized')
    print('no. paths: ', len(paths))
    splitted_paths = split_dataset(paths, 10000, 42)
    print('splitted dataset')
    
    import json
    with open('./data/data_files.json', mode='w', encoding='utf-8') as f:
        json.dump(splitted_paths, f, indent=4, ensure_ascii=False)
    
    # make_save_dataset()