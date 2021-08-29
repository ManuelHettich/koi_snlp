# -*- coding: utf-8 -*-
"""
@author: Manuel Hettich
"""

import re

import pandas as pd
import torch
from datasets import load_dataset


def extract_features(sentence, edit):
    original = re.sub("[</>]", "", sentence)
    edited = sentence[:sentence.index("<")] + edit + sentence[sentence.index(">") + 1:]
    return original, edited


def extract_example(tokenizer, seed):
    humicroedit_2 = load_dataset("humicroedit", "subtask-2")
    test_dataset = humicroedit_2["test"]
    test_dataset.shuffle(seed=seed)

    example_df = test_dataset.data.to_pandas()
    sentence_1 = example_df.original1[0]
    edit_1 = example_df.edit1[0]
    sentence_2 = example_df.original2[0]
    edit_2 = example_df.edit2[0]
    label = example_df.label[0]

    original_sentence_1, edited_sentence_1 = extract_features(sentence_1, edit_1)
    original_sentence_2, edited_sentence_2 = extract_features(sentence_2, edit_2)

    return original_sentence_1, edited_sentence_1, original_sentence_2, edited_sentence_2, label


def preprocess_data(tokenizer, subtask, file_path=None):
    if subtask == "subtask-1":
        def tokenize_function(example):
            humicroedit = load_dataset("humicroedit", subtask)
            sentence = example["original"]
            edit = example["edit"]
            sentence_orig, sentence_edit = extract_features(sentence, edit)

            return tokenizer(sentence_orig, sentence_edit, truncation=True)

        humicroedit = load_dataset("humicroedit", subtask)
        tokenized_datasets = humicroedit.map(tokenize_function)
        # Rename the meanGrade column in all data splits
        return tokenized_datasets.rename_column("meanGrade", "label")

    else:
        if file_path is not None:
            def tokenize_function(data_df):
                input_ids = []
                attention_masks = []
                for idx, sentence in enumerate(data_df["original"]):
                    edit = data_df["edit"][idx]
                    sentence_orig, sentence_edit = extract_features(sentence, edit)

                    # Tokenize the extracted sentences
                    encoded_dict = tokenizer(sentence_orig, sentence_edit, max_length=128, padding="max_length",
                                             return_attention_mask=True, truncation=True, return_tensors="pt")

                    input_ids.append(encoded_dict["input_ids"])
                    attention_masks.append(encoded_dict["attention_mask"])

                input_ids = torch.cat(input_ids, dim=0)
                attention_masks = torch.cat(attention_masks, dim=0)
                labels = torch.tensor(data_df["meanGrade"])

                return input_ids, attention_masks, labels

            data_df_1 = pd.read_csv(file_path)
            data_df_2 = pd.read_csv(file_path)

            data_df_1.rename(
                columns={"original1": "original", "edit1": "edit", "meanGrade1": "meanGrade"},
                inplace=True,
            )
            data_df_2.rename(
                columns={"original2": "original", "edit2": "edit", "meanGrade2": "meanGrade"},
                inplace=True,
            )

            tokenized_df_1 = tokenize_function(data_df_1)
            tokenized_df_2 = tokenize_function(data_df_2)

            return tokenized_df_1, tokenized_df_2, data_df_1["label"], data_df_1["id"]

        # def tokenize_function(example):
        #     humicroedit = load_dataset("humicroedit", subtask)
        #     sentence = example["original"]
        #     edit = example["edit"]
        #     sentence_orig, sentence_edit = extract_features(sentence, edit)
        #
        #     return tokenizer(sentence_orig, sentence_edit, truncation=True)
        #
        # datasets_1 = load_dataset("humicroedit", subtask)
        # datasets_2 = load_dataset("humicroedit", subtask)
        #
        # datasets_1 = datasets_1.rename_column("original1", "original")
        # datasets_1 = datasets_1.rename_column("edit1", "edit")
        # datasets_1 = datasets_1.rename_column("meanGrade1", "meanGrade")
        # datasets_1 = datasets_1.remove_columns(
        #     ["grades1", "original2", "edit2", "grades2", "meanGrade2"])
        # datasets_2 = datasets_2.rename_column("original2", "original")
        # datasets_2 = datasets_2.rename_column("edit2", "edit")
        # datasets_2 = datasets_2.rename_column("meanGrade2", "meanGrade")
        # datasets_2 = datasets_2.remove_columns(
        #     ["original1", "edit1", "grades1", "meanGrade1", "grades2"])
        #
        # tokenized_datasets_1 = datasets_1.map(tokenize_function)
        # tokenized_datasets_2 = datasets_2.map(tokenize_function)
        #
        # return tokenized_datasets_1, tokenized_datasets_2
