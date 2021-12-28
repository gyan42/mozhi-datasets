from datasets import load_metric
import datasets
from transformers import LayoutLMTokenizer
from transformers import AutoTokenizer
from sroie2019_layoutlm_dataset import HFSREIO2019LayoutLMDataset
from transformers import DataCollatorForTokenClassification

metric = load_metric("seqeval")
logger = datasets.logging.get_logger(__name__)

import torch 

class HFTokenizer(object):
    NAME = "HFTokenizer"

    def __init__(self,
                 hf_pretrained_tokenizer_checkpoint):
        self._tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_tokenizer_checkpoint)
#         self._tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

    @property
    def tokenizer(self):
        return self._tokenizer

    @staticmethod
    def init_vf(hf_pretrained_tokenizer_checkpoint):
        return HFTokenizer(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    def tokenize_and_align_labels(self,
                                  examples,
                                  label_all_tokens=True):
        """
        Transformer that extrapolates labels and bboxes with respect to tokenized word ids.
        Due to the inherent nature of wordpiece number of tokens will increase 
        """
        
        tokenized_inputs = self._tokenizer(examples["tokens"],
                                           truncation=True,
                                           is_split_into_words=True,
                                           max_length=512,
                                           pad_to_max_length=True)
        labels = []
        out_bboxes = []
        
        # Zip ner_tags and bboxes and index them
        for i, label_bbox in enumerate(zip(examples[f"ner_tags"], examples['bboxes'])):
            label, bbox = label_bbox[0], label_bbox[1]
            # Get word ids in each example
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            _bboxes = []
            
            for i, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    if i == 0:
                        _bboxes.append([0, 0, 0, 0])
                    else:
                        _bboxes.append([1000, 1000, 1000, 1000])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                    _bboxes.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                    _bboxes.append(bbox[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
            _bboxes.append([1000, 1000, 1000, 1000])
            out_bboxes.append(_bboxes[:512])   
            
      
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = out_bboxes
        return tokenized_inputs


if __name__ == '__main__':

#     hf_pretrained_tokenizer_checkpoint = "microsoft/layoutlm-base-uncased"
    hf_pretrained_tokenizer_checkpoint = "distilbert-base-uncased"
    dataset = HFSREIO2019LayoutLMDataset()
    hf_dataset = dataset.dataset

    hf_tokenizer = HFTokenizer.init_vf(hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    tokenized_datasets = hf_dataset.map(hf_tokenizer.tokenize_and_align_labels, batched=True)
    
    data_collator = DataCollatorForTokenClassification(hf_tokenizer.tokenizer)

#     print(dataset)

#     print("*" * 100)

    print(tokenized_datasets)

    print("First sample: ")
    for key in hf_dataset['train'][0].keys():
        print(key, hf_dataset['train'][0][key])
        print("\n")
        
    print("*" * 100)
    
    print(dataset.id2label.items())
    
    print("*" * 100)


    print("First tokenized sample: ")
    for key in ["input_ids", "bbox", "labels", "attention_mask"]: #tokenized_datasets['train'][0].keys():
        print(key, ": \n", tokenized_datasets['train'][0][key])
        print("\n")
        if key == 'input_ids':
            print(hf_tokenizer.tokenizer.convert_ids_to_tokens(tokenized_datasets['train'][0]['input_ids']))
            print("\n")
        if key == 'labels':
            print([dataset.id2label.get(label_id, "UNK") for label_id in tokenized_datasets['train'][0]['labels']])
            print("\n")
    
    print("Check length of all features...")
    for key in ["input_ids", "bbox", "labels", "attention_mask"]:
        print(key, len(tokenized_datasets['train'][0][key]))
        

    print(tokenized_datasets['train'])
    
    def train():
        for i in range(0,545, 32):
            batch = tokenized_datasets['train'][i:i+32]
            yield batch
    
    print(" = " * 100)
    print("Asserting the size...")
    
    for batch in train():
        for example in batch['input_ids']:
#             print(torch.tensor(example).shape[0])
            assert torch.tensor(example).shape[0] == 512
        for example in batch['bbox']:
            assert torch.tensor(example).shape[0] == 512
        for example in batch['labels']:
            assert torch.tensor(example).shape[0] == 512
        for example in batch['attention_mask']:
            assert torch.tensor(example).shape[0] == 512
            
            
    def valid():
        for i in range(0,545, 32):
            batch = tokenized_datasets['validation'][i:i+32]
            yield batch
     
    for batch in valid():
        for example in batch['input_ids']:
#             print(torch.tensor(example).shape[0])
            assert torch.tensor(example).shape[0] == 512
        for example in batch['bbox']:
            assert torch.tensor(example).shape[0] == 512
        for example in batch['labels']:
            assert torch.tensor(example).shape[0] == 512
        for example in batch['attention_mask']:
            assert torch.tensor(example).shape[0] == 512
            
#     for batch in train():
#         for example in batch['bbox']:
#             print(example)