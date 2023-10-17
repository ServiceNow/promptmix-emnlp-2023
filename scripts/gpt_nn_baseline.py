from src import models, datasets


def main(exp_dict, savedir, args):
    ds = datasets.DatasetLoader("./data", exp_dict)
    # load GPT3 classifier
    clf = models.get_model(exp_dict)
    clf.train_on_dataset(ds.dataset["train"])
    test_set = ds.dataset["test"]
    test_dict = clf.test_on_dataset(test_set)
    # get GPT3 preds
    print(test_dict)


if __name__ == "__main__":
    exp_dict = {
        "model": {"backbone": "gpt"},
        "dataset": {"num_labels": 6, "name": "trec6"},
        "exp_type": "promptmix",
        "gen_engine_name": "gpt-3.5-turbo-0613",
        "n_samples_init": 2,
        "n_gen": 1,
        "gen_engine_temp": 0.0,
        "prompt_size": 10,
        "metrics": ["accuracy", "f1", "precision", "recall"],
    }
    main(exp_dict, "./output/", None)
