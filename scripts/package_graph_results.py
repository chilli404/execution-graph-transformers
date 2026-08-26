import argparse
import json


def sanitize(value):
    if isinstance(value, dict):
        return {
            key: ("<checkpoint>" if key == "checkpoint" else sanitize(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and value.startswith(
        ("/Volumes/", "/Workspace/", "/Users/", "/local_disk0/")
    ):
        return "<redacted-path>"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    with open(args.source) as handle:
        data = json.load(handle)
    with open(args.output, "w") as handle:
        json.dump(sanitize(data), handle, indent=2)


if __name__ == "__main__":
    main()
