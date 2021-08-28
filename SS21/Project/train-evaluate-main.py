# -*- coding: utf-8 -*-
"""
@author: Manuel Hettich
"""

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer

import preprocessing
from model import create_model

TRAIN_FILE_1 = "./data/subtask-1/train.csv"
VAL_FILE_1 = "./data/subtask-1/dev.csv"
TEST_FILE_1 = "./data/subtask-1/test.csv"

MODEL_NAME = "roberta-base"
BATCH_SIZE = 32


def create_dataloaders(train_dataset, val_dataset, batch_size=32):
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    return train_dataloader, validation_dataloader


if __name__ == "__main__":
    # This implementation is loosely based on this tutorial: http://mccormickml.com/2019/07/22/BERT-fine-tuning/

    train_dataset1 = preprocessing.preprocess_file(file_path=TRAIN_FILE_1, model_name=MODEL_NAME)
    eval_dataset1 = preprocessing.preprocess_file(file_path=VAL_FILE_1, model_name=MODEL_NAME)
    # test_dataset1 = preprocessing.preprocess_file(file_path=TEST_FILE_1, model_name=MODEL_NAME)

    print('{:>5,} training samples'.format(len(train_dataset1)))
    print('{:>5,} validation samples'.format(len(eval_dataset1)))

    model = create_model(model_name=MODEL_NAME)

    train_dl, val_dl = create_dataloaders(train_dataset=train_dataset1, val_dataset=eval_dataset1,
                                          batch_size=BATCH_SIZE)

    trainer = Trainer(model=model,
                      train_dataset=train_dataset1,
                      eval_dataset=eval_dataset1)
    dl = trainer.get_train_dataloader()
    print()

# def temp():
#     train_dset = pd.read_csv(TRAIN_FILE)
#     # test_data_df = pd.read_csv(TEST_FILE)
#
#     train_1_df = train_data_df[["id", "original1", "edit1", "meanGrade1"]]
#     train_2_df = train_data_df[["id", "original2", "edit2", "meanGrade2"]]
#
#     tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
#
#     max_len = 0
#     input_ids = []
#     attention_masks = []
#
#     for sent in train_1_df["original1"]:
#         # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
#         encoded_dict = tokenizer(sent, add_special_tokens=True, max_length=128, padding="max_length",
#                                  return_attention_mask=True,
#                                  return_tensors="pt", truncation="do_not_truncate")
#
#         input_ids.append(encoded_dict['input_ids'])
#         attention_masks.append(encoded_dict['attention_mask'])
#
#     print('Decoded: ', tokenizer.decode(input_ids[0][0]))
#     input_ids = torch.cat(input_ids, dim=0)
#     attention_masks = torch.cat(attention_masks, dim=0)
#     labels = torch.tensor(train_1_df["meanGrade1"])
#
#     print('Original: ', train_1_df["original1"][0])
#     print('Token IDs:', input_ids[0])
#
#     print('test')
