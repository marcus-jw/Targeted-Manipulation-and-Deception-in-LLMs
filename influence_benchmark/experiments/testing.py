import influence_benchmark.experiments as exp
from influence_benchmark.root import PROJECT_DATA, PROJECT_ROOT
from influence_benchmark.utils.utils import *


def main():
    # Save a file with hello world in it
    with open(PROJECT_DATA / "hello.txt", "w") as f:
        f.write("Hello, World! 3")


if __name__ == "__main__":
    main()
