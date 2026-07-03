import os
import multiprocessing
from itertools import chain
from pyvi import ViTokenizer

from config.config import get_logger

logger = get_logger()
output_dir = './data/tokenized'

def get_paths(data_dir: str) -> list[str]:
    return [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p.endswith('.txt')]

def read(p: str):
    with open(p, 'r') as f:
        return f.read()

def run(paths, idx, lock):
    while True:
        try:
            with lock:
                if len(paths) == 0:
                    break
                path = paths.pop(0)
                idx.value += 1
                current_idx = idx.value
        except IndexError:
            break

        new_path = os.path.join(output_dir, os.path.basename(path))

        if os.path.exists(new_path):
            logger.info(f'Skipping: {current_idx}/{current_idx + len(paths)} files.')
            continue

        text = read(path)
        if text:
            tokenized_text = ViTokenizer.tokenize(text)

            with open(new_path, 'w', encoding='utf-8') as f:
                f.write(tokenized_text)

            logger.info(f'Passed: {current_idx}/{current_idx + len(paths)} files.')

        else:
            logger.error(f'Failed: {current_idx}/{current_idx + len(paths)} files.')

if __name__ == "__main__":
    from multiprocessing import Process, Manager, Value, Lock

    n_proc = multiprocessing.cpu_count()
    paths = get_paths('./data/raw_text')

    processes = []
    with Manager() as manager:
        file_paths = manager.list(paths)
        idx = Value('i', 0)  # Shared integer for progress tracking
        lock = Lock()        # Lock to ensure safe access to shared resources

        for _ in range(n_proc):
            p = Process(target=run, args=(file_paths, idx, lock))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()