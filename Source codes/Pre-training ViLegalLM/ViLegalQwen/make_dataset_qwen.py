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


if __name__ == '__main__':
    print('get paths')
    # Using raw_text folder instead of tokenized_text folder
    paths = get_paths('./data/raw_text')
    print('no. paths: ', len(paths))
    splitted_paths = split_dataset(paths, 10000, 42)
    print('splitted dataset')

    # Save list of file for Qwen only
    with open('./data/data_files_qwen.json', mode='w', encoding='utf-8') as f:
        json.dump(splitted_paths, f, indent=4, ensure_ascii=False)
    
    print('Dataset split and saved to data_files_qwen.json')