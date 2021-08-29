# Deep Learning for Natural Language Processing

## SS21 - Project by Manuel Hettich

### SemEval-2020 Task 7: Assessing Humor in Edited News Headlines - Subtask 2 (Funnier) with RoBERTa

This is the code for a possible solution to the SemEval-2020 Task 7: Assessing Humor in Edited News Headlines - Subtask
2 (Funnier) with RoBERTa, using an implementation
by [HuggingFace](https://huggingface.co/transformers/model_doc/roberta.html).

The code for fine-tuning the pre-trained model 'roberta-base' can be run by setting the global variable TRAIN=True in
train-evaluate-main.py and the following command:

`$ python3 train-evaluate-main.py`

You can skip the fine-tuning part by setting the same global variable TRAIN=False and running the same command. In this
case, only the calculation of the final accuracy value will be executed (warning: can take a very long time). The
fine-tuned model for this part is expected to be stored in the local folder `./roberta-trained`
The full Humicroedit dataset is expected to be stored in the local folder `./data`

By running the following command, a specific example will be taken from the test dataset and the prediction of the
loaded fine-tuned model will be compared to the ground-truth:

`$ python3 test.py`

Required Python Libraries:

```
datasets==1.11.0
matplotlib==3.4.3
numpy==1.19.5
pandas==1.3.2
scikit-learn==0.24.2
setuptools==57.4.0
torch==1.9.0
transformers==4.9.2
```
