import time
from multiprocessing import Process

from tqdm import tqdm


def process_1():
    for _ in tqdm(range(100), desc="Process 1", position=0):
        time.sleep(0.1)


def process_2():
    for _ in tqdm(range(50), desc="Process 2", position=1):
        time.sleep(0.2)


if __name__ == "__main__":
    p1 = Process(target=process_1)
    p2 = Process(target=process_2)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
