# -*- coding: utf-8 -*-
"""
@author: Manuel Hettich
"""

import numpy as np
import pandas as pd
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


def predict(df: pd.DataFrame, model, batch_size=16):
    """
    Calculating the model's predictions on a test dataset.

    :param df: Pandas DataFrame with the data (input_ids, attention_mask and labels in this order)
    :param model: The learning model to be used for the predictions
    :param batch_size: The batch size for prediction execution
    :return: The predicted values of this dataset
    """
    preds = []
    for idx in np.arange(0, df[0].shape[0] // batch_size):
        inputs = df[0][idx * batch_size:(idx + 1) * batch_size]
        attention_masks = df[1][idx * batch_size:(idx + 1) * batch_size]
        labels = df[2][idx * batch_size:(idx + 1) * batch_size]
        outputs = model(input_ids=inputs, attention_mask=attention_masks, labels=labels)

        logits = outputs["logits"]
        for logit in logits.reshape(-1):
            preds.append(logit.item())

        print(f"Batch ({idx}/{df[0].shape[0] // batch_size})")

    return preds


def calc_accuracy(predictions, labels) -> float:
    """
    Calculate an accuracy metric for a combination of predictions and labels.

    :param predictions: Predictions of the model
    :param labels: Labels of the data points
    :return: Accuracy value (Correct predictions / Number of data points)
    """

    num = len(predictions)
    correct = 0
    for idx, prediction in enumerate(predictions):
        if prediction == labels[idx]:
            correct += 1
    return correct / num


if __name__ == "__main__":
    """
    This implementation is loosely based on the following tutorials:
    https://huggingface.co/course/chapter3/3
    http://mccormickml.com/2019/07/22/BERT-fine-tuning/
    https://towardsdatascience.com/transformers-retraining-roberta-base-using-the-roberta-mlm-procedure-7422160d5764
    """

    # Initialising the RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    if TRAIN:
        # Fine-tuning the pre-trained model RoBERTa based on "roberta-base", using the training set for subtask-1
        tokenized_datasets = preprocess_data(tokenizer, "subtask-1")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = create_model(model_name=MODEL_NAME)

        # Using the Trainer API from HuggingFace for training the model
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

        # Storing the fine-tuned model locally on disk
        trainer.save_model(MODEL_PATH)

    # Evaluating the model on the test set for subtask-2
    tokenized_df_1, tokenized_df_2, labels, ids = preprocess_data(tokenizer, "subtask-2", TEST_FILE_2)
    model = transformers.RobertaForSequenceClassification.from_pretrained(TRAINED_MODEL_PATH)
    model.eval()

    results_df = pd.DataFrame(columns=["id", "label", "pred1", "pred2", "pred_total"])
    results_df["id"] = ids
    results_df["label"] = labels

    # Calculating the model's predictions
    preds1 = predict(tokenized_df_1, model, batch_size=BATCH_SIZE)
    results_df["pred1"] = preds1
    preds2 = predict(tokenized_df_2, model, batch_size=BATCH_SIZE)
    results_df["pred2"] = preds2

    preds_total = []
    for idx in range(len(preds1)):
        if preds1[idx] > preds2[idx]:
            preds_total.append(1)
        elif preds1[idx] < preds2[idx]:
            preds_total.append(2)
        else:
            preds_total.append(0)

    # Calculating the resulting accuracy
    accuracy = calc_accuracy(preds_total, labels)
    print(accuracy)

    # Storing the prediction results locally on disk
    results_df["pred_total"] = preds_total
    results_df.to_csv(TRAINED_MODEL_PATH + "/predictions.csv", index=False)
