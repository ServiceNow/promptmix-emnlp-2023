from haven import haven_utils as hu
from copy import deepcopy

EXP_GROUPS = {}
DATASETS = {
    "banking77": {
        "name": "banking77",
        "num_labels": 77,
    },
    "twitter_complaints": {
        "name": "twitter_complaints",
        "num_labels": 2,
        "epochs": 5,
        "lr": 4e-5,
        "weight_decay": 0.01,
    },
    "trec6": {
        "name": "trec6",
        "num_labels": 6,
        "epochs": 5,
        "lr": 4e-5,
        "weight_decay": 0.01,
    },
    "subj": {
        "name": "subj",
        "num_labels": 2,
        "epochs": 5,
        "lr": 4e-5,
        "weight_decay": 0.01,
    },
}


def get_base_config(
    dname="banking77",
    relabel=False,
    feedback=False,
    oracle_config=["naive"],  # one of naive/gpt3_engine/human
    gen_engine=["gpt3_engine"],  # one of human/gpt3_engine
    runs=[0, 1, 2],
    n_cycles=10,
    n_gen=1,
    prompt_type=None,
    gen_engine_name="gpt-3.5-turbo-0613",
    gen_engine_temp=1.0,
    metrics=["accuracy", "f1", "precision", "recall"],
    backbone=None,
    c=None,
):
    d_config = {"config": "original"}
    d_config.update(DATASETS[dname])
    return hu.cartesian_exp_group(
        {
            "run#": runs,  # for extrinsic evaluation
            # "dataset": d_config,
            "dataset": {k: d_config[k] for k in ["config", "name", "num_labels"]},
            "model": {
                "name": "intent_classification",
                "backbone": backbone,  # gpt/sentence-transformers/all-MiniLM-L6-v2/distilbert-base-uncased
            },
            "lr": d_config.get("lr", 6e-5),
            "batch_size": 8,
            "pool_batch_size": 32,
            "epochs": d_config.get("epochs", 5),
            "n_samples_init": [2],
            "shuffle_prop": 0.0,
            "query_size": 10,
            "prompt_size": 10,  # upper bound on the number of exemplers per class in the prompt
            "relabel": relabel,
            "feedback": feedback,
            "oracle_config": oracle_config,
            "warmup_ratio": 0.1,
            "weight_decay": d_config.get("weight_decay", 0.001),
            "n_gen": n_gen,
            "metrics": [metrics],
            "metric_best": "accuracy",
            "exp_type": "promptmix",  # this will contain the 10 sample per class of clinc + the generated samples of gpt3 relabelled
            "eval_accumulation_steps": 30,
            "gen_engine": gen_engine,
            "gen_engine_name": gen_engine_name,  # only used for GPT3
            "gen_engine_temp": gen_engine_temp,  # only used for GPT3
            "boost_oos": [1],
            "n_cycles": n_cycles,
            "prompt_type": prompt_type,
            "c": c,  # number of classes to put
        },
        remove_none=True,
    )


dnames = ["subj"]
EXP_GROUPS["poc_ub"] = []
backbone = "distilbert-base-uncased"
for dname in dnames:
    EXP_GROUPS["poc_ub"] += get_base_config(
        dname=dname,
        gen_engine=["human"],
        feedback=True,
        oracle_config=["naive"],
        runs=[0, 1, 2],
        n_cycles=5,
        n_gen=10,
        backbone=backbone,
    )

EXP_GROUPS["poc_naive"] = []

for dname in dnames:
    prompt_type = "mixed"
    engine = "gpt-3.5-turbo-0613"
    EXP_GROUPS["poc_naive"] += get_base_config(
        dname=dname,
        oracle_config=["naive"],
        gen_engine=["gpt3_engine"],
        feedback=False,
        gen_engine_name=engine,
        runs=[0],
        n_cycles=10,
        n_gen=10,
        prompt_type=prompt_type,
        backbone=backbone,
        c=[4],
    )

EXP_GROUPS["poc_rlbl"] = []

for exp_dict in deepcopy(EXP_GROUPS["poc_naive"]):
    exp_dict["relabel"] = True
    exp_dict["oracle_config"] = "gpt3_engine"
    EXP_GROUPS["poc_rlbl"] += [exp_dict]

EXP_GROUPS["gpt_nn_baseline"] = []
for dname in dnames:
    EXP_GROUPS["gpt_nn_baseline"] += get_base_config(
        backbone="gpt",
        dname=dname,
        n_gen=1,
        gen_engine_temp=0.0,
    )
