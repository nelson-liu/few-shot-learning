#!/usr/bin/env python3
import argparse
import json
import pickle
import numpy as np

def main(predictions_path, gold_path):
    with open(predictions_path, "rb") as f:
        predictions = pickle.load(f)
    gold_labels = []
    with open(gold_path) as f:
         for line in f:
             gold_labels.append(json.loads(line)["gold_label"])

    mapped_predictions = []
    label_map = {0: "non-entailment", 1: "non-entailment", 2: "entailment"}
    for label_probs in predictions["all_label_probs"]:
        predicted_label = np.argmax(label_probs)
        mapped_predictions.append(label_map[predicted_label])

    assert len(mapped_predictions) == len(gold_labels)
    total = 0
    correct = 0
    for pred, gold in zip(mapped_predictions, gold_labels):
        if pred == gold:
            correct += 1
        total += 1
    print(f"Accuracy: {correct / total} (correct: {correct}, total: {total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Plot IID/OOD differences."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="Path to output pickle file from evaluating on HANS",
    )
    parser.add_argument(
        "--gold-path",
        type=str,
        required=True,
        help="Path to HANS gold file.",
    )
    args = parser.parse_args()
    main(
        args.predictions_path,
        args.gold_path
    )
