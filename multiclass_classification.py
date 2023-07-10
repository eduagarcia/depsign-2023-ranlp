from argparse import ArgumentParser
import sys
import os

from simpletransformers.classification import ClassificationModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import pandas as pd

CACHE_DIR="/workspace/cache/simpletransformers"

def get_args():
    parser = ArgumentParser(description='Simple-Tranformers training')
    
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--validation_file', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=3)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    train_df = pd.read_csv(args.train_file)
    train_df.drop(columns=["pid"], inplace=True)

    eval_df = pd.read_csv(args.validation_file)
    eval_df.drop(columns=["pid"], inplace=True)

    orig_labels = train_df["label"].unique().tolist()
    print(orig_labels)
    
    train_df['label'] = train_df['label'].apply(orig_labels.index)
    eval_df['label'] = eval_df['label'].apply(orig_labels.index)

    labels = list(range(len(orig_labels)))

    weights = compute_class_weight(
        "balanced",
        y=train_df["label"],
        classes=labels,
    )

    print(weights)

    model = ClassificationModel(
        "auto",
        args.model_name_or_path,
        weight=list(weights),
        use_cuda=True,
        num_labels=len(labels),
        args={
            "num_train_epochs": args.epochs,
            "cache_dir": CACHE_DIR,
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "reprocess_input_data": True, 
            "overwrite_output_dir": True,
            "save_model_every_epoch": False,
            "output_dir": args.output_dir,
            "tensorboard_dir": os.path.join(args.output_dir, "runs")
        },
    )

    # Train the model
    model.train_model(train_df)

    metrics = {
        "class_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average=None),
        "macro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    }

    # Evaluate the model
    result, _, _ = model.eval_model(eval_df, **metrics)

    print(result)

if __name__ == "__main__":
    main()