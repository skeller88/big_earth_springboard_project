from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List


def parallelize_task(num_workers, task, iterator):
    chunk_size = len(iterator) // num_workers
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks: List[Future] = []
        start_index = 0
        for x in range(num_workers):
            end_index = min(start_index + chunk_size, len(iterator))
            chunk = iterator[start_index:end_index]
            tasks.append(executor.submit(task, chunk))
            start_index = end_index

        return [task.result() for task in tasks]
