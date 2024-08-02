from influence_benchmark.root import PROJECT_DATA


def main():
    # Save a file with hello world in it
    with open(PROJECT_DATA / "hello.txt", "w") as f:
        f.write("Hello, World!")


if __name__ == "__main__":
    main()
