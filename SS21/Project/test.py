# -*- coding: utf-8 -*-
"""
@author: Manuel Hettich
"""

import transformers

from preprocessing import extract_example

MODEL_NAME = "roberta-base"
MODEL_PATH = "./roberta-retrained"
TRAINED_MODEL_PATH = "./roberta-trained"
TEST_FILE_2 = "./data/subtask-2/test.csv"

if __name__ == "__main__":
    tokenizer = transformers.RobertaTokenizer.from_pretrained(MODEL_NAME)
    original_sentence_1, edited_sentence_1, original_sentence_2, edited_sentence_2, label = extract_example(seed=1)

    print("#### Example from Humicroedit dataset for subtask-2 ####")
    print("Original headline #1: ", original_sentence_1)
    print("Edited headline #1: ", edited_sentence_1)
    print("Original headline #2: ", original_sentence_2)
    print("Edited headline #2: ", edited_sentence_2)

    # Tokenize the extracted sentences
    encoded_dict_1 = tokenizer(original_sentence_1, edited_sentence_1, max_length=128, padding="max_length",
                               return_attention_mask=True, truncation=True, return_tensors="pt")
    encoded_dict_2 = tokenizer(original_sentence_2, edited_sentence_2, max_length=128, padding="max_length",
                               return_attention_mask=True, truncation=True, return_tensors="pt")

    decoded_tokens_1 = tokenizer.decode(encoded_dict_1["input_ids"][0])
    decoded_tokens_2 = tokenizer.decode(encoded_dict_2["input_ids"][0])

    print("1st edited headline with special tokens: ", decoded_tokens_1)
    print("2nd edited headline with special tokens: ", decoded_tokens_2)

    print("Loading the fine-tuned language model...")
    model = transformers.RobertaForSequenceClassification.from_pretrained(TRAINED_MODEL_PATH)
    outputs_1 = model(input_ids=encoded_dict_1["input_ids"], attention_mask=encoded_dict_1["attention_mask"])
    outputs_2 = model(input_ids=encoded_dict_2["input_ids"], attention_mask=encoded_dict_2["attention_mask"])

    pred_label = 0
    pred_1 = outputs_1["logits"][0]
    pred_2 = outputs_2["logits"][0]

    if pred_1 > pred_2:
        pred_label = 1
    elif pred_2 > pred_1:
        pred_label = 2

    print("Ground truth label: ", label, "(funnier edited headline 1 / 2 or equally funny = 0)")
    print("Predicted label: ", pred_label)
