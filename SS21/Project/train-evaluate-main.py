# -*- coding: utf-8 -*-
"""
@author: Manuel Hettich
"""

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import RobertaTokenizer, TrainingArguments, Trainer, \
    DataCollatorWithPadding

from model import create_model
from preprocessing import preprocess_data

TRAIN_FILE_1 = "./data/subtask-1/train.csv"
VAL_FILE_1 = "./data/subtask-1/dev.csv"
TEST_FILE_1 = "./data/subtask-1/test.csv"
TRAIN_FILE_2 = "./data/subtask-2/train.csv"
VAL_FILE_2 = "./data/subtask-2/dev.csv"
TEST_FILE_2 = "./data/subtask-2/test.csv"

BATCH_SIZE = 16
TRAIN = False
MODEL_NAME = "roberta-base"
MODEL_PATH = "./roberta-retrained"
TRAINED_MODEL_PATH = "./roberta-trained"


class RMSE(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="RMSE",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float32")),
                    "references": datasets.Sequence(datasets.Value("float32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("float32"),
                }
            ),
            reference_urls=[""],
        )

    def _compute(self, *, predictions=None, references=None, **kwargs):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.item()
        loss = torch.nn.MSELoss()
        return {
            "RMSE": torch.sqrt(loss(predictions, references))
        }


def predict(df: pd.DataFrame, model, batch_size=16):
    preds = []
    for idx in np.arange(0, df[0].shape[0] // batch_size):
        inputs = df[0][idx * batch_size:(idx + 1) * batch_size]
        attention_masks = df[1][idx * batch_size:(idx + 1) * batch_size]
        labels = df[2][idx * batch_size:(idx + 1) * batch_size]
        outputs = model(input_ids=inputs, attention_mask=attention_masks, labels=labels)

        logits = outputs["logits"]
        for logit in logits.reshape(-1):
            preds.append(logit.item())

    return preds


def calc_accuracy(predictions, labels) -> float:
    num = len(predictions)
    correct = 0
    for idx, prediction in enumerate(predictions):
        if prediction == labels[idx]:
            correct += 1
    return correct / num


if __name__ == "__main__":
    # This implementation is loosely based on the following tutorials:
    # https://huggingface.co/course/chapter3/3
    # http://mccormickml.com/2019/07/22/BERT-fine-tuning/
    # https://towardsdatascience.com/transformers-retraining-roberta-base-using-the-roberta-mlm-procedure-7422160d5764

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    if TRAIN:
        tokenized_datasets = preprocess_data(tokenizer, "subtask-1")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = create_model(model_name=MODEL_NAME)

        training_args = TrainingArguments(
            output_dir=MODEL_PATH,
            overwrite_output_dir=True,
            save_steps=250,
            save_total_limit=2,
            num_train_epochs=10,
            seed=1
        )
        trainer = Trainer(args=training_args,
                          model=model,
                          train_dataset=tokenized_datasets['train'],
                          eval_dataset=tokenized_datasets['validation'],
                          data_collator=data_collator,
                          tokenizer=tokenizer)

        trainer.train()
        trainer.save_model(MODEL_PATH)
        torch.save(model.state_dict(), MODEL_PATH + "/task1.pt")

    tokenized_df_1, tokenized_df_2, labels = preprocess_data(tokenizer, "subtask-2", TEST_FILE_2)
    model = transformers.RobertaForSequenceClassification.from_pretrained(TRAINED_MODEL_PATH)
    model.load_state_dict(torch.load('./roberta-trained/task1.pt'))
    model.eval()

    preds1 = predict(tokenized_df_1, model, batch_size=BATCH_SIZE)
    preds2 = predict(tokenized_df_2, model, batch_size=BATCH_SIZE)

    preds = []
    for idx in range(len(preds1)):
        if preds1[idx] > preds2[idx]:
            preds.append(1)
        elif preds1[idx] < preds2[idx]:
            preds.append(2)
        else:
            preds.append(0)

    accuracy = calc_accuracy(preds, labels)
    print(accuracy)

    inputs = tokenized_df_1[0][:5]
    attention_masks = tokenized_df_1[1][:5]
    labels = tokenized_df_1[2][:5]
    outputs = model(input_ids=inputs, attention_mask=attention_masks, labels=labels)

    # predictions = trainer.predict(tokenized_datasets["validation"], )
    # print(predictions.predictions.shape, predictions.label_ids.shape)
