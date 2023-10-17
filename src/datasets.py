import os, pickle, tqdm, numpy as np
from haven import haven_utils as hu
import src.utils as mdu
from datasets import Dataset, DatasetDict


class DatasetLoader:
    def __init__(self, data_root, exp_dict, tokenizer=None):
        self.tokenizer = tokenizer
        exp_type = exp_dict["exp_type"]
        # Get id2name
        id2name_path = os.path.join(
            data_root, exp_dict["dataset"]["name"], "id2name.json"
        )
        id2name = hu.load_json(id2name_path)

        # load the dataset based on the configuration
        if exp_type == "promptmix":
            data_path = os.path.join(
                data_root,
                exp_dict["dataset"]["name"],
                "full",
                "dataset.pkl",
            )
            ds = read_pickle(data_path)
            # load the original data without any generated samples
            # Get lines for each split
            train_lines, train_labels = ds["train"]["text"], ds["train"]["intent"]
            val_lines, val_labels = ds["val"]["text"], ds["val"]["intent"]
            test_lines, test_labels = ds["test"]["text"], ds["test"]["intent"]

            n_samples = exp_dict.get("n_samples_init")
            if n_samples is not None:
                train_lines, train_labels = get_subset(
                    train_lines,
                    train_labels,
                    n_samples,
                    np.unique(ds["train"]["intent"]),
                    seed=32,
                )

        elif exp_type in ["baseline", "binary_baseline"]:
            # load data for different configs
            config = exp_dict["dataset"]["config"]
            org_oos_id = oos_id = exp_dict["dataset"].get("oos_id")

            if exp_type == "binary_baseline":
                org_oos_id = exp_dict["dataset"].get("org_oos_id")

            ds_suffix = (
                f"_{exp_dict['testval_size']}testval" if config == "ood_oos" else ""
            )

            data_path = os.path.join(
                data_root,
                exp_dict["dataset"]["name"],
                config,
                f"dataset{ds_suffix}.pkl",
            )
            ds = mdu.read_pickle(data_path)
            if exp_dict["upsample_factor"] > 1:  # factor of 1 means no upsampling
                oos_lines, _ = extract_oos(ds["train"], org_oos_id)
                oos_lines = oos_lines * exp_dict["upsample_factor"]
                train_lines, train_labels = filter_oos(ds["train"], org_oos_id)
                train_lines.extend(oos_lines)
                train_labels.extend([org_oos_id] * len(oos_lines))
            else:
                train_lines, train_labels = ds["train"]["text"], ds["train"]["intent"]

            val_lines, val_labels = ds["val"]["text"], ds["val"]["intent"]
            test_lines, test_labels = ds["test"]["text"], ds["test"]["intent"]

            if exp_type == "binary_baseline":
                id2name = {"0": "inscope", "1": "oos"}
                # oos class id 1, inscope class id 0 for binary oracle
                train_labels = [1 if l == org_oos_id else 0 for l in train_labels]
                val_labels = [1 if l == org_oos_id else 0 for l in val_labels]
                test_labels = [1 if l == org_oos_id else 0 for l in test_labels]
        else:
            raise ValueError(f"config {exp_type} does not exist")

        # Define Dataset
        def map2name(ids):
            return [id2name[str(l)] for l in ids]

        train_ds = {
            "text": train_lines,
            "intent": train_labels,
            "intent_names": map2name(train_labels),
        }
        if exp_dict.get("prompt_type") in ["mixed", "desc_w_mixing"]:
            train_ds["alpha"] = [None] * len(train_lines)
        self.dataset = DatasetDict(
            train=Dataset.from_dict(train_ds),
            validation=Dataset.from_dict(
                {
                    "text": val_lines,
                    "intent": val_labels,
                    "intent_names": map2name(val_labels),
                }
            ),
            test=Dataset.from_dict(
                {
                    "text": test_lines,
                    "intent": test_labels,
                    "intent_names": map2name(test_labels),
                }
            ),
        )

        # attributes
        self.id2name = id2name
        name2desc_path = os.path.join(
            data_root, exp_dict["dataset"]["name"], "name2desc.json"
        )
        if os.path.exists(name2desc_path):
            self.name2desc = hu.load_json(name2desc_path)
        else:
            self.name2desc = None
        self.full_dataset = ds

        if self.tokenizer:
            # encode datasets
            for part in ["train", "validation", "test"]:
                self.dataset[part] = self.dataset[part].map(
                    self.preprocess, batched=True
                )

        print(self.dataset)
        print(self.dataset["train"][:5])
        print("train labels", len(np.unique(train_labels)))
        print("test labels", len(np.unique(test_labels)))

    def get_split(self, split):
        return self.dataset[split]

    def preprocess(self, example):
        results = self.tokenizer(
            example["text"],
            max_length=50,
            truncation=True,
            padding="max_length",
        )
        results["label"] = example["intent"]
        return results


def split_by_label(X, y, labels):
    ind_list = np.array([i for i in range(len(y)) if y[i] in labels])
    return np.array(X)[ind_list], np.array(y)[ind_list]


def get_subset(text_lines, intent_lines, num_samples: int, labels, seed: int = 32):
    # Map label to indices
    label2indices = {}
    for i, y in enumerate(intent_lines):
        if y not in label2indices:
            label2indices[y] = []
        label2indices[y] += [i]

    ind_list = []
    selected_intent_lines = []

    for lbl in tqdm.tqdm(labels, desc="creating warm start"):
        # x, y = datasets.split_by_label(text_lines, intent_lines, [lbl])
        with hu.random_seed(seed):
            ind_list += list(
                np.random.choice(label2indices[lbl], num_samples, replace=False)
            )

    selected_text_lines = [text_lines[i] for i in ind_list]
    selected_intent_lines = list(np.array(intent_lines)[ind_list])

    assert len(selected_text_lines) == num_samples * len(labels)
    return selected_text_lines, selected_intent_lines


class DatasetLoaderSeq2Seq(DatasetLoader):
    """ "
    Dataset Loader for T5 classifiers
    """

    def __init__(self, data_root, exp_dict):
        super(DatasetLoaderSeq2Seq, self).__init__(data_root, exp_dict)
        id2name_path = os.path.join(
            data_root, exp_dict["dataset"]["name"], "id2name.json"
        )
        for k, v in self.dataset.items():
            self.dataset[k] = Dataset.from_dict(prepare_for_seq2seq(v, id2name_path))


def read_pickle(data_path):
    """
    returns appropriate version of the CLINC full dataset
    Parameters:
    ----------
    data_path: absolute path of the pickle file
    """
    with open(data_path, "rb") as f:
        return pickle.load(f)


def prepare_for_seq2seq(dataset, id2name_path):
    """
    dataset: Dict[str]: <list>
    """
    import json

    id2name = json.load(open(id2name_path))
    return {
        "text": [t + " </s>" for t in dataset["text"]],
        # intents are class ids here, not names
        "intent": [id2name[str(i)] + " </s>" for i in dataset["intent"]],
    }


def filter_oos(data_dict, oos_id):
    """Removes oos samples from the data dict"""
    lines, labels = data_dict["text"], data_dict["intent"]
    # some datasets (like SNIPS) don't have an OOS class
    if oos_id is None:
        return lines, labels
    _lines, _labels = [], []
    for idx, intent_id in enumerate(labels):
        if intent_id == oos_id:
            continue
        _lines.append(lines[idx])
        _labels.append(labels[idx])
    # print(len(_lines), len(_labels))
    return _lines, _labels


def extract_oos(data_dict, oos_id):
    """Extract the OOS samples from the data dict. It is the
    opposite of filter_oos"""
    lines, labels = data_dict["text"], data_dict["intent"]
    # some datasets (like SNIPS) don't have an OOS class
    _lines, _labels = [], []
    for idx, intent_id in enumerate(labels):
        if intent_id != oos_id:
            continue
        _lines.append(lines[idx])
        _labels.append(labels[idx])
    return _lines, _labels
