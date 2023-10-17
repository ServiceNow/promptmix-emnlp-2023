from .backbones import *
from .classifiers import HFClassifier, GPT3Classifier


def get_model(exp_dict):
    # GPT3 based classifier
    if exp_dict["model"]["backbone"] in ["gpt"]:
        return GPT3Classifier(exp_dict)
    # standard CLS token based classifier
    return HFClassifier(exp_dict)
