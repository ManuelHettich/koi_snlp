# -*- coding: utf-8 -*-
"""
@author: Manuel Hettich
"""

import transformers


def create_model(model_name):
    """
    Define the learning model for the given task. In this case, I am using the RoBERTa model prepared for sequence
    classification task, which is trained on the Humicroedit dataset for subtask-1.

    :param model_name: The name of the pre-trained learning model by HuggingFace, e.g. "roberta-base"
    :return: The model from HuggingFace with the necessary architecture for sequence classification
    """
    return transformers.RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)
