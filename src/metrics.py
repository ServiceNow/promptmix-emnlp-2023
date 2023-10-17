import datasets, numpy as np, json, evaluate
from sklearn.metrics import confusion_matrix


class Conf_Mat(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="conf_mat",
            citation=" ",
            inputs_description="num_classes",
            features=datasets.Features(
                {
                    "predictions": datasets.Value(
                        "int64" if self.config_name != "sts-b" else "float32"
                    ),
                    "references": datasets.Value(
                        "int64" if self.config_name != "sts-b" else "float32"
                    ),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references, num_classes=2):
        class_data = np.zeros([num_classes, num_classes])
        mat = confusion_matrix(
            references.reshape([-1]).astype(int),
            predictions.reshape([-1]).astype(int),
            labels=np.arange(class_data.shape[0]),
        )
        return {"conf_mat": mat.tolist()}


class Metrics:
    def __init__(self, exp_dict, tokenizer=None, do_argmax=True):
        self.exp_dict = exp_dict
        self.tokenizer = tokenizer
        self.do_argmax = do_argmax

    def compute_metrics(self):
        """
        Will choose the appropriate metric computer based on the config
        """
        if "bert" in self.exp_dict["model"]["backbone"].lower():
            return self.compute_metrics_bert
        return self.compute_metrics_t5

    def compute_metrics_bert(self, eval_pred):
        predictions, labels = eval_pred
        if self.do_argmax:
            predictions = np.argmax(predictions, axis=1)
        if self.exp_dict["exp_type"] == "binary_baseline":
            predictions = [
                1 if p == self.exp_dict["dataset"]["org_oos_id"] else 0
                for p in predictions
            ]
            labels = [
                1 if p == self.exp_dict["dataset"]["org_oos_id"] else 0 for p in labels
            ]
        metrics = {}
        # sort, so accuracy is evaluated first, always
        for metric in sorted(self.exp_dict["metrics"]):
            _metric = eval(f"self.{metric}")
            metrics.update(_metric(predictions, labels))
        # add labels and predictions
        # metrics["predictions"] = {"preds": predictions, "labels": labels}
        return metrics

    def compute_metrics_t5(self, eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
            preds = preds.argmax(axis=-1)

        # maybe switch skip_special_tokens to False
        decoded_preds = self.tokenizer.batch_decode(preds)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels)

        # compute metrics
        metrics = {}
        # for f1, we will need to convert pred text to id
        id2name_map = json.load(
            open(f"./data/{self.exp_dict['dataset']['name']}/id2name.json")
        )
        name2id_map = json.load(
            open(f"./data/{self.exp_dict['dataset']['name']}/name2id.json")
        )
        n_acc = 0
        pred_ids, label_ids = [], []
        for idx in range(preds.shape[0]):
            print(decoded_preds[idx], decoded_labels[idx])
            pred = decoded_preds[idx].split("</s>", maxsplit=1)[0].strip()
            label = decoded_labels[idx].split("</s>", maxsplit=1)[0].strip()
            print(pred, label)
            if pred in name2id_map and pred == label:
                pred_ids.append(int(name2id_map[pred]))
                n_acc += 1
                label = name2id_map[label]
            else:
                label = name2id_map[label]
                incorrect_pred = int(label) + 1
                if incorrect_pred < self.exp_dict["dataset"]["num_labels"]:
                    pred = incorrect_pred
                else:
                    pred = int(label) - 1
                pred_ids.append(pred)
            label_ids.append(int(label))
        metrics = {}
        for metric in self.exp_dict["metrics"]:
            _metric = eval(f"self.{metric}")
            metrics.update(_metric(pred_ids, label_ids))
        return metrics

    def IAOR(self, predictions, labels):
        "this is 0.5(IA+OR)"
        IA = self.accuracy(predictions, labels)["inscope_accuracy"]
        OR = self.recall(predictions, labels)["oos_recall"]
        return {"IAOR": 0.5 * (IA + OR)}

    def accuracy(self, predictions, labels):
        accuracies = {}
        acc = evaluate.load("accuracy")
        accuracies.update(acc.compute(predictions=predictions, references=labels))
        oos_id = self.exp_dict["dataset"].get("oos_id")
        if oos_id is not None:
            # compute in_scope accuracy as well
            inscope_preds, inscope_labels = [], []
            for idx in range(len(labels)):
                if labels[idx] == oos_id:
                    continue
                inscope_labels.append(labels[idx])
                inscope_preds.append(predictions[idx])
            self.inscope_preds, self.inscope_labels = inscope_preds, inscope_labels
            accuracies["inscope_accuracy"] = acc.compute(
                predictions=inscope_preds, references=inscope_labels
            )["accuracy"]
        return accuracies

    def f1(self, predictions, labels):
        f1s = {}
        f1 = evaluate.load("f1")
        f1s.update(
            f1.compute(predictions=predictions, references=labels, average="macro")
        )
        if self.exp_dict["dataset"].get("oos_id") is not None:
            f1s["inscope_f1"] = f1.compute(
                predictions=self.inscope_preds,
                references=self.inscope_labels,
                average="macro",
            )["f1"]
        return f1s

    def precision(self, predictions, labels):
        precision = evaluate.load("precision")
        return precision.compute(
            predictions=predictions, references=labels, average="macro"
        )

    def recall(self, predictions, labels):
        recalls = {}
        recall = evaluate.load("recall")
        recalls.update(
            recall.compute(predictions=predictions, references=labels, average="macro")
        )
        oos_id = self.exp_dict["dataset"].get("oos_id")
        if oos_id is not None and oos_id in range(
            self.exp_dict["dataset"]["num_labels"]
        ):
            # compute OOS recall
            outscope_preds = []
            for idx in range(len(labels)):
                if labels[idx] == oos_id:
                    outscope_preds.append(1 if predictions[idx] == oos_id else -1)
            recalls["oos_recall"] = outscope_preds.count(1) / len(outscope_preds)
        return recalls

    def conf_mat(self, predictions, targets):
        num_class = self.exp_dict["dataset"]["num_labels"]
        conf_mat = Conf_Mat()
        return conf_mat.compute(
            predictions=predictions, references=targets, num_classes=num_class
        )
