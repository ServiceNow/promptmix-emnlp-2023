import torch, numpy as np, pickle, json
from src.generator import HumanEngine
from datasets import DatasetDict, Dataset

torch.backends.cudnn.benchmark = True

write_json = lambda obj, path: json.dump(obj, open(path, "w"))
read_json = lambda path: json.load(open(path, "r"))

write_pickle = lambda obj, path: pickle.dump(obj, open(path, "wb"))
read_pickle = lambda path: pickle.load(open(path, "rb"))


def augment_train_set(train_set, gen_sub, dataset_loader):
    for k in train_set.column_names:
        if isinstance(gen_sub[k], list):
            gen_sub[k] = gen_sub[k] + train_set[k]
        else:
            gen_sub[k] = torch.cat([gen_sub[k], torch.as_tensor(train_set[k])], dim=0)
    train_ds = {
        "text": gen_sub["text"],
        "intent": np.array(gen_sub["intent"]).astype(int),
        "intent_names": [dataset_loader.id2name[str(l)] for l in gen_sub["intent"]],
    }
    if "alpha" in gen_sub:
        train_ds["alpha"] = gen_sub["alpha"]
    dataset_hf = DatasetDict(
        train=Dataset.from_dict(train_ds),
        val=dataset_loader.dataset["validation"],
        test=dataset_loader.dataset["test"],
    )
    train_set = dataset_hf["train"].map(dataset_loader.preprocess, batched=True)
    return train_set


def preprocess_aug(examples, seed_examples, tokenizer, name2id):
    results = {
        "input_ids": [],
        "attention_mask": [],
        "label": [],
        "text": [],
        "seed_examples": [],
    }
    for intent, texts in examples.items():
        assert isinstance(texts, list)
        has_gen_scores = False
        has_alphas = False

        if isinstance(texts[0], dict):  # texts contains gen prob/alpha/both
            if "gen_prob" in texts[0].keys():
                gen_probs = []
                has_gen_scores = True
            if "alpha" in texts[0].keys():
                alphas = []
                has_alphas = True
            for t in texts:
                if has_gen_scores:
                    gen_probs += t["gen_prob"]
                if has_alphas:
                    alphas += [t["alpha"]] * len(t["text"])
            texts = [t for o in texts for t in o["text"]]

        _results = tokenizer(
            texts,
            max_length=50,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        results["input_ids"] += [_results["input_ids"]]
        results["attention_mask"] += [_results["attention_mask"]]
        results["label"] += [name2id[intent]] * len(texts)
        results["text"] += texts
        results["seed_examples"] += [seed_examples[intent]]
        if has_gen_scores:
            if "gen_prob" not in results:
                results["gen_prob"] = []
            results["gen_prob"] += gen_probs
        if has_alphas:
            if "alpha" not in results:
                results["alpha"] = []
            results["alpha"] += alphas

        # some tokenizers (like SBERT's) return token_type_ids too by default
        if "token_type_ids" in _results.keys():
            if "token_type_ids" not in results:
                results["token_type_ids"] = []
            results["token_type_ids"] += [_results["token_type_ids"]]

    for k, v in results.items():
        if len(v) and isinstance(v[0], torch.Tensor):
            results[k] = torch.cat(v, 0)
        else:
            results[k] = v
    return results


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_data(
    engine, train_set, dataset_loader, exp_dict, tokenizer, intent_names=None
):
    n_gen = exp_dict.get("n_gen", 1)
    if intent_names is None:
        intent_names = [v for v in dataset_loader.id2name.values() if v not in ["oos"]]
    n_train = len(train_set["intent_names"])
    name2id = {v: k for k, v in dataset_loader.id2name.items()}
    generated = {}
    seed_examples = {}
    for intent in intent_names:
        # Get texts corresponding to intent
        text_list = train_set["text"]
        intent_list = train_set["intent_names"]
        texts = [text_list[i] for i in range(n_train) if intent == intent_list[i]]
        # randomly select n sentences
        n = min(len(texts), exp_dict["prompt_size"])
        texts = np.random.choice(texts, n, replace=False)

        if isinstance(engine, HumanEngine):
            gen = engine.generate(
                {intent: texts},
                dataset_loader.full_dataset,
                name2id,
            )
        else:
            gen = engine.generate({intent: texts})

        if isinstance(gen[intent], dict):  # using generation likelihood
            gen[intent] = {k: v[:n_gen] for k, v in gen[intent].items()}
        elif isinstance(gen[intent], list):  # not using likelihood
            gen[intent] = gen[intent][:n_gen]
        else:
            raise ValueError(
                f"return type {type(gen['intent'])} not understood for gen[intent]"
            )
        seed_examples.update({intent: texts.tolist()})
        generated.update(gen)

    dataset = preprocess_aug(generated, seed_examples, tokenizer, name2id)
    return dataset
