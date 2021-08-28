# -*- coding: utf-8 -*-
"""
@author: Manuel Hettich
"""

import re

import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class HumicroEdit(Dataset):
    def __init__(self, train_df, model_name):
        self.train_df = train_df
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, item):
        # Extract the features from raw data
        sentence = self.train_df["original"][item]
        edit = self.train_df["edit"][item]
        sentence_orig, sentence_edit = extract_features(sentence, edit)
        label = torch.tensor(self.train_df["meanGrade"][item])

        # Tokenize the extracted sentences
        encoded_dict = self.tokenizer(sentence_orig, sentence_edit, add_special_tokens=True, max_length=128,
                                      padding="max_length", truncation="do_not_truncate",
                                      return_attention_mask=True,
                                      return_tensors="pt")

        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']

        return input_ids, attention_masks, label


def preprocess_file(file_path, model_name):
    # Load the training and test data into memory using pandas
    if file_path is not None:
        data_df = pd.read_csv(file_path)
        tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

        input_ids = []
        attention_masks = []
        labels = data_df["meanGrade"]
        for idx, sentence in enumerate(data_df["original"]):
            edit = data_df["edit"][idx]
            sentence_orig, sentence_edit = extract_features(sentence, edit)
            label = torch.tensor(data_df["meanGrade"][idx])

            # Tokenize the extracted sentences
            encoded_dict = tokenizer(sentence_orig, sentence_edit, add_special_tokens=True, max_length=128,
                                     padding="max_length", truncation="do_not_truncate",
                                     return_attention_mask=True,
                                     return_tensors="pt")

            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        return input_ids, attention_masks, labels
    else:
        return None


def extract_features(sentence, edit):
    original = re.sub("[</>]", "", sentence)
    edited = sentence[:sentence.index("<")] + edit + sentence[sentence.index(">") + 1:]
    return original, edited


if __name__ == "__main__":
    preprocess_file("./data/subtask-1/train.csv", "roberta-base")
