from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    BertModel,
)  # TrainingArguments, Trainer

import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from sentence_transformers import SentenceTransformer


class BertModelWithCustomLossFunction(torch.nn.Module):
    def __init__(self, exp_dict):
        super(BertModelWithCustomLossFunction, self).__init__()
        self.num_labels = exp_dict["dataset"]["num_labels"]
        self.bert = BertModel.from_pretrained(
            exp_dict["model"]["backbone"], num_labels=self.num_labels
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, self.num_labels)
        self.oos_id = exp_dict["dataset"]["oos_id"]
        self.oos_weight = exp_dict["oos_weight"]

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        output = self.dropout(outputs.pooler_output)
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            # you can define any loss function here yourself
            # see https://pytorch.org/docs/stable/nn.html#loss-functions for an overview
            class_weights = torch.tensor([1] * self.num_labels)
            class_weights[self.oos_id] = self.oos_weight
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=class_weights.float().to(logits.device)
            )
            # next, compute the loss based on logits + ground-truth labels
            loss = loss_fct(logits.view(-1, self.num_labels), labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_backbone(exp_dict):
    if exp_dict["model"]["backbone"] in [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]:
        backbone = SentenceTransformer(exp_dict["model"]["backbone"])
    elif exp_dict["model"]["backbone"] in [
        "distilbert-base-uncased",
        "bert-large-uncased",
        "bert-base-uncased",
        "huawei-noah/TinyBERT_General_4L_312D",
    ]:
        num_labels = exp_dict["dataset"]["num_labels"]
        backbone = AutoModelForSequenceClassification.from_pretrained(
            exp_dict["model"]["backbone"], num_labels=num_labels
        )
        if not exp_dict["model"].get("pretrained", True):
            backbone.init_weights()

    elif exp_dict["model"]["backbone"] in [
        "t5-base",
        "t5-small",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "google/t5-v1_1-base",
        "google/t5-v1_1-small",
        "google/t5-v1_1-large",
        "google/t5-v1_1-xl",
        "google/t5-v1_1-xxl",
    ]:
        config = AutoConfig.from_pretrained(exp_dict["model"]["backbone"])
        backbone = AutoModelForSeq2SeqLM.from_pretrained(
            exp_dict["model"]["backbone"], config=config
        )
    else:
        raise ValueError(exp_dict["model"]["backbone"])
    return backbone
