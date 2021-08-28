# -*- coding: utf-8 -*-
"""
@author: Manuel Hettich
"""

import transformers


def create_model(model_name):
    return transformers.RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1,
                                                                         output_attentions=False,
                                                                         output_hidden_states=False)
