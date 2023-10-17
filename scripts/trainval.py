import os, pandas as pd, torch, nltk, argparse, copy, numpy as np

pd.options.display.max_columns = 50
import exp_configs

nltk.download("wordnet")
nltk.download("omw-1.4")

from haven import haven_wizard as hw, haven_utils as hu

from src import datasets, models, generator, utils as ut

torch.backends.cudnn.benchmark = True


def main(exp_dict, savedir, args):
    """Main."""

    # ====
    # Set seed and device
    # ===================
    ut.set_seed(seed=42 + exp_dict.get("run#", 0))
    num_labels = exp_dict["dataset"]["num_labels"]

    # entity and name are pulled from the environment
    # wandb.init(name=savedir, config=exp_dict)

    gendata_path = os.path.join(savedir, "texts")
    if not os.path.exists(gendata_path):
        os.mkdir(gendata_path)

    # get model
    model = models.get_model(exp_dict)

    # get dataset
    ds_loader = datasets.DatasetLoader(args.datadir, exp_dict, model.tokenizer)
    train_set = ds_loader.get_split("train")
    val_set = ds_loader.get_split("validation")
    test_set = ds_loader.get_split("test")

    score_list = []
    score_list_savedir = os.path.join(savedir, "score_list.pkl")

    # load engine here so that we don't need load engine over and over
    # ========
    engine = generator.get_generator(exp_dict)
    train_set_org = copy.deepcopy(train_set)

    n_cycles = exp_dict.get("n_cycles", 10)
    oracle_cost = 0
    # number of intents in the prompt
    c = exp_dict.get("c", 4)

    for cycle in range(n_cycles + 1):
        # seed by cycle
        ut.set_seed(seed=42 + cycle + exp_dict.get("run#", 0))

        # Step 1: train on labeled data
        train_set_size = len(train_set)
        train_dict = model.train_on_dataset(train_set, val_set, savedir)

        # Step 2: test on the test set
        val_dict = model.test_on_dataset(val_set, metric_key_prefix="eval")
        test_dict = model.test_on_dataset(test_set, metric_key_prefix="test")

        # Get score_dict
        score_dict = train_dict
        score_dict["train_set_size"] = train_set_size
        score_dict["oracle_cost"] = oracle_cost
        score_dict.update(val_dict)
        score_dict.update(test_dict)
        score_dict["cycle"] = cycle
        score_dict["n_train_set_org"] = len(train_set_org["text"])
        # wandb.log(score_dict, step=cycle)
        score_list += [score_dict]

        # Display and Save results
        df = pd.DataFrame(score_list).tail()[
            [
                "eval_accuracy",
                "test_accuracy",
                "train_set_size",
                "oracle_cost",
                "n_train_set_org",
            ]
        ]
        print(df)
        print(savedir)
        print("=============")

        hu.save_pkl(score_list_savedir, score_list)
        train_set.save_to_disk(os.path.join(savedir, "datasets", f"{cycle}.hf"))
        # Break to prevent adding examples that will not be evaluated
        if cycle == n_cycles:
            break

        # Add examples
        gen_list_selected = {}
        gen_list = []
        i_round = 0
        intents_done = []
        intent_subsets = []
        for class_id in range(exp_dict["dataset"]["num_labels"]):
            if class_id in intents_done:
                continue
            generate_uttrs = True
            while generate_uttrs:  # loop for generating utterances for each intent
                # select three other intents to be mixed in the prompt
                remaining_intents = np.setdiff1d(
                    np.arange(num_labels), np.array([class_id] + intents_done)
                )
                if len(remaining_intents) <= c:
                    other_intents = remaining_intents
                else:
                    other_intents = np.random.choice(
                        remaining_intents, c - 1, replace=False
                    )
                intent_names = [
                    ds_loader.id2name[str(id)]
                    for id in other_intents.tolist() + [class_id]
                ]
                intent_subsets.append(intent_names)
                generated = engine.generate_data(
                    train_set_org,
                    ds_loader,
                    exp_dict,
                    model.tokenizer,
                    ut.preprocess_aug,
                    intent_names=intent_names,
                )
                generate_uttrs = len(generated["label"]) > 0
                if not generate_uttrs:
                    continue
                intents_done += other_intents.tolist() + [class_id]
                gen_dicts = []
                # not using self.n_gen in range below because we generate for
                # multiple alphas when prompt_type=mixed
                for idx in range(len(generated["text"])):
                    gen_dict = get_gen_dict(generated, idx, ds_loader, i_round, cycle)
                    gen_dicts.append(gen_dict)
                # Create the first empty dict
                if len(gen_list_selected) == 0:
                    gen_list_selected = {
                        k: [] for k in list(gen_dicts[0].keys()) + ["selected"]
                    }

                # Use naive to select all generated examples
                if exp_dict["oracle_config"] == "naive":
                    for gen_dict in gen_dicts:
                        gen_dict["selected"] = [True]
                        for k in gen_dict:
                            gen_list_selected[k] += gen_dict[k]
                else:
                    raise ValueError(f'{exp_dict["oracle_config"]} does not exist')

                # append to history
                gen_list += [gen_dict]
                i_round += 1
                # check if gen_dict['selected'] contains only True
                if np.array(gen_dict["selected"]).mean():
                    break
        if not gen_list:
            print("Nothing generated for any intent! Ending loop")
            break
        # log the generated examples (visualization purposes :) )
        vis_json = list(
            pd.DataFrame(gen_list)[
                ["text", "intent", "intent_names", "selected", "cycle", "i_round"]
            ]
            .T.to_dict()
            .values()
        )
        hu.save_pkl(os.path.join(savedir, f"intent_subsets.pkl"), intent_subsets)
        hu.save_json(os.path.join(gendata_path, f"{cycle}.json"), vis_json)
        hu.save_pkl(os.path.join(gendata_path, f"{cycle}.pkl"), gen_list)

        # Step 6: combine data with oracle (augment)
        train_set = ut.augment_train_set(train_set, gen_list_selected, ds_loader)

        if exp_dict["feedback"]:
            print("FEEDBACK UPDATE...")
            train_set_org = train_set

        # print("n_train_set_org", len(train_set_org["text"]))

    print("Experiment Completed")


def get_gen_dict(generated, idx, ds_loader, i_round, cycle):
    intents = list(map(int, [generated["label"][idx]]))
    if "seed_examples" not in generated:
        seed_examples = []
    else:
        seed_examples = generated["seed_examples"][0]
    gen_dict = {
        "text": [generated["text"][idx]],
        "intent": intents,
        "intent_names": [ds_loader.id2name[l] for l in [generated["label"][idx]]],
        "input_ids": [generated["input_ids"][idx]],
        "attention_mask": [generated["attention_mask"][idx]],
        "label": copy.deepcopy(intents),
        "seed_examples": seed_examples,
        "selected": [False],
        "i_round": [i_round],
        "cycle": [cycle],
    }

    if "token_type_ids" in generated.keys():
        gen_dict["token_type_ids"] = [generated["token_type_ids"][idx]]
    if "gen_prob" in generated.keys():
        gen_dict["gen_prob"] = [generated["gen_prob"][idx]]
    if "alpha" in generated.keys():
        gen_dict["alpha"] = [generated["alpha"][idx]]

    return gen_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group_list",
        nargs="+",
        default="resnet",
        help="name of an experiment in exp_configs.py",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="/mnt/home/haven_output",
        help="folder where logs will be saved",
    )
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    parser.add_argument("-d", "--datadir", type=str, default="./data")
    parser.add_argument("-md", "--modeldir", type=str, default=None)
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Overwrite previous results"
    )
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument(
        "-j",
        "--job_scheduler",
        type=str,
        default=None,
        help="If 1, runs in toolkit in parallel",
    )
    parser.add_argument("-v", default="results.ipynb", help="orkestrator")
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )

    args, unknown = parser.parse_known_args()

    file_name = os.path.basename(__file__)[:-3]  # remove .py

    if args.job_scheduler in ["1", "toolkit"]:
        import job_configs

        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None

    hw.run_wizard(
        func=main,
        exp_groups=exp_configs.EXP_GROUPS,
        job_config=job_config,
        python_binary_path=args.python_binary,
        # python_file_path=f"-m runners.{file_name}",
        use_threads=True,
        args=args,
        results_fname="results/active_nlp.ipynb",
    )
