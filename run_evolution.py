import argparse
import json

from evolution.evolution import Evolution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(config)
    evolution = Evolution(config)
    evolution.neuroevolution()

if __name__ == "__main__":
    main()