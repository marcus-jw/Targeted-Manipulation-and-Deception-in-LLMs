from dotenv import load_dotenv

import influence_benchmark.experiments as exp
from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.RL.training_funcs import print_accelerator_info
from influence_benchmark.root import PROJECT_ROOT
from influence_benchmark.utils.utils import *


def main():
    # Save a file with hello world in it
    print("Test")
    assert load_dotenv(PROJECT_ROOT / ".env"), ".env file not found in influence_benchmark/.env"
    with open(PROJECT_DATA / "hello.txt", "w") as f:
        f.write(f"{PROJECT_ROOT}")


if __name__ == "__main__":
    main()
