import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    with open(args.source) as source, open(args.output, "w") as output:
        for line in source:
            if line.strip():
                output.write(json.dumps(json.loads(line), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
