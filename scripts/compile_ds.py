import argparse
from datasets import load_dataset
from haven import haven_utils as hu
from collections import Counter


# 2. Twitter complaints
def compile_twitter_complaints():
    ds_raft = load_dataset("ought/raft", "twitter_complaints")
    ds_raft = ds_raft.rename_columns({"Tweet text": "text"})

    ds = load_dataset(
        "csv", data_files="./data/twitter_complaints/full/complaints-data.csv"
    )
    ds = ds.rename_columns({"label": "intent"})
    # remove duped training set
    ds = ds.filter(lambda r: r["text"] not in ds_raft["train"]["text"])

    # fix label (0: no complaint, 1: complaint)
    print(Counter(ds_raft["train"]["Label"]))
    for part in ds_raft:
        mapped_intents = []
        for i in ds_raft[part]["Label"]:
            if i == 2:
                mapped_intents.append(0)
            else:
                mapped_intents.append(i)
        ds_raft[part] = ds_raft[part].add_column("intent", mapped_intents)

    dataset = {
        "train": {
            "text": ds_raft["train"]["text"],
            "intent": ds_raft["train"]["intent"],
        },
        "val": {"text": ds["train"]["text"], "intent": ds["train"]["intent"]},
        "test": {"text": ds["train"]["text"], "intent": ds["train"]["intent"]},
        "train_full": {"text": ds["train"]["text"], "intent": ds["train"]["intent"]},
    }

    print("train labels", (Counter(dataset["train"]["intent"])))
    print("test labels", (Counter(dataset["test"]["intent"])))

    hu.save_pkl("./data/twitter_complaints/full/dataset.pkl", dataset)


def compile_trec6():
    ds = load_dataset("trec")  # text, coarse_label
    ds = ds.rename_columns({"coarse_label": "intent"})
    name2id = {
        "abbreviation": 2,
        "entity": 1,
        "description": 0,
        "human": 3,
        "location": 5,
        "numeric": 4,
    }
    hu.save_json("./data/trec6/name2id.json", name2id)
    hu.save_json("./data/trec6/id2name.json", {str(v): k for k, v in name2id.items()})
    dataset = {}
    for part in ["train", "val", "test"]:
        if part == "val":  # no validation set
            dataset[part] = ds["test"].to_dict()
        else:
            dataset[part] = ds[part].to_dict()
    print("train labels", (Counter(dataset["train"]["intent"])))
    print("test labels", (Counter(dataset["test"]["intent"])))
    hu.save_pkl("./data/trec6/full/dataset.pkl", dataset)


def compile_subj():
    ds = load_dataset("SetFit/subj")  # text, coarse_label
    ds = ds.rename_columns({"label": "intent"})
    name2id = {"objective": 0, "subjective": 1}
    hu.save_json("./data/subj/name2id.json", name2id)
    hu.save_json("./data/subj/id2name.json", {str(v): k for k, v in name2id.items()})
    dataset = {}
    for part in ["train", "val", "test"]:
        if part == "val":  # no validation set
            dataset[part] = ds["test"].to_dict()
        else:
            dataset[part] = ds[part].to_dict()
    print("train labels", (Counter(dataset["train"]["intent"])))
    print("test labels", (Counter(dataset["test"]["intent"])))
    hu.save_pkl("./data/subj/full/dataset.pkl", dataset)


def main(dname):
    if dname == "trec6":
        compile_trec6()
    elif dname == "tc":
        compile_twitter_complaints()
    elif dname == "subj":
        compile_subj()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dname",
        default="trec6",
        help="the dataset to compile",
    )
    args, unknown = parser.parse_known_args()

    main(args.dname)
