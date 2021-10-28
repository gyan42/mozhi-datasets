import numpy as np
import transformers
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
# Ref: https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py
import os
from pathlib import Path

import datasets
from datasets import DownloadConfig
import os
from pathlib import Path
from datasets import load_dataset, ClassLabel, DownloadConfig

from transformers import AutoTokenizer
from hf_tokenize import HFTokenizer
from sroie2019_dataset import HFSREIO2019Dataset

metric = load_metric("seqeval")

logger = datasets.logging.get_logger(__name__)


def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


if __name__ == "__main__":
    model_n_version = "sroie2019v1"
    max_epochs = 50
    learning_rate = 2e-5
    batch_size = 16
    model_root_dir = "~/.mozhi/models/hf/"

    hf_pretrained_model_checkpoint = "distilbert-base-uncased"
    hf_pretrained_tokenizer_checkpoint = "distilbert-base-uncased"

    hf_dataset = HFSREIO2019Dataset()
    hf_preprocessor = HFTokenizer.init_vf(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    hf_model = AutoModelForTokenClassification.from_pretrained(hf_pretrained_model_checkpoint,
                                                            num_labels=len(hf_dataset.labels))

    hf_model.config.id2label = hf_dataset.id2label
    hf_model.config.label2id = hf_dataset.label2id

    tokenized_datasets = hf_dataset.dataset.map(hf_preprocessor.tokenize_and_align_labels, batched=True)

    # ---------------------------------------------------------------------------------------------------

    args = TrainingArguments(
        f"test-ner",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        weight_decay=0.01,
    )

    data_collator = DataCollatorForTokenClassification(hf_preprocessor.tokenizer)

    trainer = Trainer(
        hf_model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=hf_preprocessor.tokenizer,
        compute_metrics=lambda p: compute_metrics(p=p, label_list=hf_dataset.labels)
    )

    trainer.train()
    trainer.evaluate()

    # Predictions on test dataset and evaluation

    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [hf_dataset.labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [hf_dataset.labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)

    out_dir = os.path.expanduser(model_root_dir) + "/" + model_n_version
    trainer.save_model(out_dir)
