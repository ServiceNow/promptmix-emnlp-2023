import os, pandas as pd, argparse, datasets as hf_datasets
from src import models, datasets, utils as ut
from haven import haven_utils as hu
from copy import deepcopy

pjoin = os.path.join


def main(exp_dict, savedir, args):
    ut.set_seed(seed=42 + exp_dict.get("run#", 0))

    # modify exp_dict for relabeling exp
    naive_exp_dict = deepcopy(exp_dict)
    naive_exp_dict["oracle_config"] = "naive"
    naive_exp_dict["relabel"] = False

    naive_savedir_hash = hu.hash_dict(naive_exp_dict)
    naive_savedir = pjoin(os.path.dirname(savedir), naive_savedir_hash)

    # skip if we've already run the relabel exp or don't run relabel if the original exp hasn't finished
    if not os.path.exists(pjoin(naive_savedir, f"datasets/{exp_dict['n_cycles']}.hf")):
        return

    print(savedir)
    hu.save_json(pjoin(savedir, "exp_dict.json"), exp_dict)
    clf = models.get_model(exp_dict)
    ds = datasets.DatasetLoader("./data", exp_dict, clf.tokenizer)
    # get the last saved dataset for the already run
    # experiment inside `path`
    dataset_folder = pjoin(naive_savedir, "datasets")
    dataset_files = os.listdir(dataset_folder)

    final_dataset = sorted(dataset_files, key=lambda x: int(x.split(".")[0]))[-1]
    generated_dataset = hf_datasets.load_from_disk(pjoin(dataset_folder, final_dataset))

    # Use GPT+nearest neighbour to relabel these generated examples
    gpt_model_config = deepcopy(exp_dict)
    gpt_model_config["model"]["backbone"] = "gpt"
    gpt_model_config["gen_engine_temp"] = 0.0

    gpt_model = models.get_model(gpt_model_config)
    # compute intent embeddings
    gpt_model.train_on_dataset(ds.get_split("train"))
    # relabeled data
    gpt_data = gpt_model.test_on_dataset(generated_dataset, return_data=True)

    relabeled_train_set = gpt_data["data"]
    if "input_ids" not in relabeled_train_set:
        relabeled_train_set = relabeled_train_set.map(ds.preprocess, batched=True)
    print("GPT relabeling complete.")
    print(f"{gpt_data['metrics']['accuracy']*100:.2f}% relabeled")

    # save the relabeled dataset!
    relabeled_train_set.save_to_disk(pjoin(savedir, "datasets", final_dataset))
    print(relabeled_train_set)
    print("Training classifier on the relabaled dataset...")

    train_dict = clf.train_on_dataset(
        relabeled_train_set, ds.get_split("validation"), savedir
    )
    test_set = ds.dataset["test"]
    test_dict = clf.test_on_dataset(test_set)
    print(test_dict)
    score_dict = train_dict
    score_dict["train_set_size"] = len(relabeled_train_set)
    score_dict.update(test_dict)
    score_list = [score_dict]
    score_list_savedir = pjoin(savedir, "score_list.pkl")
    hu.save_pkl(score_list_savedir, score_list)
    df = pd.DataFrame(score_list).tail()[["test_accuracy"]]
    print(df)
    print("Experiment Completed")
    print(savedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="/mnt/home/haven_output",
        help="folder where logs will be saved",
    )
    args, unknown = parser.parse_known_args()
    exp_folder = args.savedir_base
    if os.path.exists(pjoin(exp_folder, "deleted")):
        os.system(f'rm -rf {pjoin(exp_folder, "deleted")}')
    count = 0
    for folder in os.listdir(exp_folder):
        # print(path)
        path = pjoin(exp_folder, folder)
        try:
            exp_dict = hu.load_json(pjoin(path, "exp_dict.json"))
        except FileNotFoundError:
            continue
        if (
            exp_dict["model"]["backbone"] in ["distilbert-base-uncased"]
            and exp_dict["dataset"]["name"] in ["trec6"]
            and exp_dict["gen_engine"] == "gpt3_engine"
            and exp_dict["n_cycles"] == 5
            and exp_dict["gen_engine_name"] in ["gpt-3.5-turbo-0613"]
            and exp_dict.get("prompt_type")
            in [
                # None,
                "mixed",
                # "acl_prompt_w_desc",
                # "desc_w_mixing",
                # "desc_no_mixing",
                # "desc_w_examples",
            ]
            and os.path.exists(pjoin(path, "intent_subsets.pkl"))
        ):
            count += 1
            main(exp_dict, path, args)
    print(count)
