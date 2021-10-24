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

metric = load_metric("seqeval")

logger = datasets.logging.get_logger(__name__)

_CITATION = ""
_DESCRIPTION = """\

"""


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


class SROIE2019Config(datasets.BuilderConfig):
    """BuilderConfig for SROIE2019"""

    def __init__(self, **kwargs):
        """BuilderConfig for SROIE2019.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SROIE2019Config, self).__init__(**kwargs)

class SROIE2019(datasets.GeneratorBasedBuilder):
    """SROIE2019 dataset."""

    BUILDER_CONFIGS = [
        SROIE2019Config(name="SROIE2019", version=datasets.Version("1.0.0"), description="SROIE2019 dataset"),
    ]

    def __init__(self, 
                 *args,
                 url="https://github.com/gyan42/model-store/raw/main/SROIE2019/", 
                 train_file="train.txt", 
                 val_file="valid.txt", 
                 test_file="test.txt", 
                 **kwargs):
        super(SROIE2019, self).__init__(*args, **kwargs)
        self._url = url
        self._train_file = train_file
        self._val_file = val_file
        self._test_file = test_file

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "company", "date", "address", "total", "O"
                            ]
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{self._url}{self._train_file}",
            "dev": f"{self._url}{self._val_file}",
            "test": f"{self._url}{self._test_file}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # SROIE2019 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }


class HFSREIO2019Dataset():
    """
    """
    NAME = "HFSREIO2019Dataset"
    def __init__(self):
        config = DownloadConfig(cache_dir=os.path.join(str(Path.home()), '.mozhi'))
        self._dataset = SROIE2019()
        self._dataset.download_and_prepare(download_config=config)
        self._dataset = self._dataset.as_dataset()

    @property
    def datasets(self):
        return self._dataset

    @property
    def labels(self) -> ClassLabel:
        return self._dataset['train'].features['ner_tags'].feature.names

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']

    def test(self):
        return self._dataset["test"]

    def validation(self):
        return self._dataset["validation"]


class HFTokenizer(object):
    NAME = "HFTokenizer"
    def __init__(self,
                 hf_pretrained_tokenizer_checkpoint):
        self._tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_tokenizer_checkpoint)

    @property
    def tokenizer(self):
        return self._tokenizer

    @staticmethod
    def init_vf(hf_pretrained_tokenizer_checkpoint):
        return HFTokenizer(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    def tokenize_and_align_labels(self,
                                  examples,
                                  label_all_tokens=True):
        tokenized_inputs = self._tokenizer(examples["tokens"],
                                           truncation=True,
                                           is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


if __name__ == "__main__":
    model_n_version = "sroie2019v1"
    max_epochs=3
    learning_rate=2e-5
    batch_size=32
    model_root_dir="~/.mozhi/models/hf/"

    hf_pretrained_model_checkpoint="distilbert-base-uncased"
    hf_pretrained_tokenizer_checkpoint="distilbert-base-uncased"

    hf_dataset = HFSREIO2019Dataset()
    hf_preprocessor = HFTokenizer.init_vf(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    hf_model = AutoModelForTokenClassification.from_pretrained(hf_pretrained_model_checkpoint,
                                                            num_labels=len(hf_dataset.labels))

    hf_model.config.id2label = hf_dataset.id2label
    hf_model.config.label2id = hf_dataset.label2id

    tokenized_datasets = hf_dataset.datasets.map(hf_preprocessor.tokenize_and_align_labels, batched=True)

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

    print(tokenized_datasets["train"])
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
