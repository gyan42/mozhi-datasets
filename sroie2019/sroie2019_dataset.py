# Ref: https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py

import datasets
import os
from pathlib import Path
from datasets import ClassLabel, DownloadConfig

logger = datasets.logging.get_logger(__name__)


_CITATION = ""
_DESCRIPTION = """\

"""


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
                 url="https://github.com/gyan42/mozhi-datasets/tree/main/sroie2019/version1/",
                 train_file="train.txt",
                 val_file="valid.txt",
                 test_file="test.txt",
                 ner_tags=("company", "date", "address", "total", "O"),
                 **kwargs):
        super(SROIE2019, self).__init__(*args, **kwargs)
        self._url = url
        self._train_file = train_file
        self._val_file = val_file
        self._test_file = test_file
        self._ner_tags = ner_tags

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(self._ner_tags))
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


class HFSREIO2019Dataset(object):
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
