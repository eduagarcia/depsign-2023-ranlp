from pprint import pprint
import json
import os

DATA_DIR = "output/external"

REMOVE_KEYS = [
    "eval_loss",
    "eval_precision (macro)",
    "eval_precision (moderate)",
    "eval_precision (not depression)",
    "eval_precision (severe)",
    "eval_recall (macro)",
    "eval_recall (moderate)",
    "eval_recall (not depression)",
    "eval_recall (severe)",
    "eval_runtime",
    "eval_samples",
    "eval_samples_per_second",
    "eval_steps_per_second",
]

results = list()
for experiment in os.listdir(DATA_DIR):
    data = experiment.split("_step2_")
    if not "step2" in experiment:
        continue

    experiment_path = os.path.join(DATA_DIR, experiment)
    for model in os.listdir(experiment_path):
        eval_path = os.path.join(experiment_path, model, "eval_results.json")

        with open(eval_path, "r") as f:
            result = json.load(f)

        result["model_name"] = model
        result["external_data"] = data[0]
        result["training_data"] = data[1]

        for key in REMOVE_KEYS:
            result.pop(key)
        results.append(result)

results.sort(key=lambda item: item["eval_f1 (macro)"], reverse=True)

pprint(results[:3])
