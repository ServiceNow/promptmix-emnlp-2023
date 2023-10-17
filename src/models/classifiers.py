import os, datasets, numpy as np, time
from haven import haven_utils as hu
from transformers import AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics.pairwise import distance_metrics
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine
from .backbones import get_backbone
from src.metrics import Metrics
from src import utils, generator, models, datasets as src_datasets


class BaseClassifier:
    """Base class for all classifiers"""

    def __init__(self, exp_dict):
        if exp_dict["model"]["backbone"] not in ["gpt"]:
            self.backbone = get_backbone(exp_dict).cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(
                exp_dict["model"]["backbone"], use_fast=True
            )
        else:
            # will need SBERT encoder for nearest-neighbor part
            self.sbert = models.get_backbone(
                {"model": {"backbone": "sentence-transformers/all-mpnet-base-v2"}}
            )
        self.exp_dict = exp_dict

    def train_on_dataset(self):
        raise NotImplementedError

    def test_on_dataset(self):
        raise NotImplementedError


class HFClassifier(BaseClassifier):
    """Huggingface models fine-tuned with a classification head"""

    def train_on_dataset(self, train_set, val_set, savedir, n_gpu=1):
        self.backbone.train()
        if self.exp_dict["exp_type"] in ["baseline", "binary_baseline"]:
            eval_strategy = "epoch"
            early_stop = True
        else:
            # not saving promptmix model checkpoints to save disk space
            eval_strategy = "no"
            # we don't want to early stop for promptmix model
            early_stop = False
        self.training_args = TrainingArguments(
            savedir,
            evaluation_strategy=eval_strategy,
            save_strategy="epoch",
            learning_rate=self.exp_dict["lr"],
            per_device_train_batch_size=self.exp_dict["batch_size"] // n_gpu,
            per_device_eval_batch_size=self.exp_dict.get(
                "pool_batch_size", self.exp_dict["batch_size"]
            )
            // n_gpu,
            num_train_epochs=self.exp_dict["epochs"],
            warmup_ratio=self.exp_dict["warmup_ratio"],
            weight_decay=self.exp_dict["weight_decay"],
            load_best_model_at_end=early_stop,
            metric_for_best_model=self.exp_dict["metric_best"],
            save_total_limit=1,
            report_to=[],
        )

        self.trainer = Trainer(
            self.backbone,
            self.training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            tokenizer=None,
            compute_metrics=Metrics(self.exp_dict).compute_metrics(),
        )
        train_dict = self.trainer.train()
        return train_dict.metrics

    def test_on_dataset(self, test_set, metric_key_prefix="test"):
        return self.trainer.evaluate(test_set, metric_key_prefix=metric_key_prefix)


class GPT3Classifier(BaseClassifier):
    """Classification GPT3.5"""

    def __init__(self, exp_dict):
        super().__init__(exp_dict)
        hparams = dict(
            gen_engine="gpt3_engine",
            n_gen=exp_dict["n_gen"],
            gen_engine_name=exp_dict["gen_engine_name"],
            gen_engine_temp=exp_dict["gen_engine_temp"],
            prompt_size=exp_dict["prompt_size"],
        )
        self.backbone = self.clf = generator.get_generator(hparams)
        name2desc_path = os.path.join(
            "./data", exp_dict["dataset"]["name"], "name2desc.json"
        )
        name2id_path = os.path.join(
            "./data", exp_dict["dataset"]["name"], "name2id.json"
        )
        self.intent2desc = hu.load_json(name2desc_path)
        self.name2id = hu.load_json(name2id_path)
        self.compute_metrics = Metrics(
            self.exp_dict, do_argmax=False
        ).compute_metrics_bert

    def train_on_dataset(self, train_set):
        self.train_set = train_set
        print("Computing intent embeddings...")
        intent_embeddings = {}
        for intent_name, intent_id in self.name2id.items():
            examples = src_datasets.extract_oos(train_set, int(intent_id))[0]
            intent_embeddings[intent_name] = self.get_intent_embedding(
                intent_name, examples
            )
        self.intent_embeddings = intent_embeddings

    def get_intent_embedding(self, intent_name, examples):
        """Average the SBERT embeddings of all items in the list of examples."""
        return np.mean(
            [self.sbert.encode(example) for example in examples + [intent_name]], axis=0
        )

    def map_gpt_label_to_intent(self, gpt_label):
        # get the intent whose embedding is the closest to the label gpt predicted
        gpt_label_emb = self.sbert.encode(gpt_label)
        # the first entry is the closest intent
        return sorted(
            list(self.name2id.keys()),
            key=lambda x: cosine(self.sbert.encode(x), gpt_label_emb),
        )[0]

    def build_prompt_base(self, intents=None):
        input_prompt = "Consider the task of classifying between the following intents (along with some examples):\n"
        if intents is None:
            texts, intents = self.train_set["text"], self.train_set["intent_names"]
        else:
            subset = self.train_set.filter(lambda x: x["intent_names"] in intents)
            intents, texts = subset["intent_names"], subset["text"]

        # loop over all the intents
        for idx, intent in enumerate(list(set(intents))):
            _texts = [t for i, t in enumerate(texts) if intents[i] == intent]
            np.random.shuffle(_texts)
            examples = "\n".join(["- " + t for t in _texts])
            input_prompt += (
                f"{idx+1}. {intent}, which is about {self.intent2desc[intent]}. "
                + "Some examples of utterances include:\n"
                + f"{examples}\n"
            )
        return input_prompt

    def build_question(self, base_prompt, test_sentence):
        base_prompt += (
            f"\nConsider the following test sentence:\n"
            + f"1. {test_sentence}\n\n"
            + f"Classify the test sentence into one of the previously described intents. Only provide the name of the intent.\n1. "
        )
        return base_prompt

    def group_examples_by_intent_groups(self, test_set):
        print("Grouping examples by common groups of nearest intents...")
        examples_by_intent_groups = {}
        intent_group_hash = {}
        for row in test_set:
            test_emb = self.sbert.encode(row["text"])
            intent_distances = [
                (intent_name, cosine(test_emb, intent_emb))
                for intent_name, intent_emb in self.intent_embeddings.items()
            ]
            n_neighbors = sorted(intent_distances, key=lambda x: x[1])[:5]
            # discard the distance and only keep the intent names
            n_neighbors = [x[0] for x in n_neighbors]
            n_neighbors.sort()

            nn_hash = hu.hash_str(str(n_neighbors))
            if nn_hash not in intent_group_hash:
                intent_group_hash[nn_hash] = n_neighbors
            if nn_hash not in examples_by_intent_groups:
                examples_by_intent_groups[nn_hash] = []
            examples_by_intent_groups[nn_hash].append(row)
        return examples_by_intent_groups, intent_group_hash

    def test_on_dataset(self, test_set, metric_key_prefix="test", return_data=False):
        preds = []
        labels = []
        possible_oos = []
        inferences = []
        # group examples by common groups of nearest intents
        (
            examples_by_intent_groups,
            intent_group_hash,
        ) = self.group_examples_by_intent_groups(test_set)

        # construct prompt
        done = 0
        to_sleep_mark = 100
        relabeled_data = []
        for group_hash, test_examples in examples_by_intent_groups.items():
            label_names = [x["intent_names"] for x in test_examples]
            labels += [x["intent"] for x in test_examples]
            texts = [x["text"] for x in test_examples]
            intents = intent_group_hash[group_hash]
            base_prompt = self.build_prompt_base(intents)
            _preds = []
            for text, label_name in zip(texts, label_names):
                question = self.build_question(base_prompt, text)
                pred = self.map_gpt_label_to_intent(self.clf.predict(question))
                if pred in self.name2id:
                    relabeled_data.append(
                        {
                            "text": text,
                            "intent_names": pred,
                            "intent": self.name2id[pred],
                            "question": question,
                            "old_intent_names": label_name,
                        }
                    )
                    _preds.append(self.name2id[pred])
                    inferences.append(
                        {"prediction": pred, "text": text, "label": label_name}
                    )
                else:
                    possible_oos.append(
                        {"prediction": pred, "text": text, "label": label_name}
                    )
                    _preds.append(-1)
            preds += _preds

            assert len(_preds) == len(test_examples)
            done += len(test_examples)
            print(f"Tested {done}/{len(test_set)} examples...")

            if done >= to_sleep_mark:
                print("Taking my customary 1-min nap after 100 examplezzz...")
                time.sleep(60)
                to_sleep_mark += 100

        metrics = self.compute_metrics((preds, labels))
        hu.save_json("possible_oos.json", possible_oos)
        hu.save_json("inferences.json", inferences)
        if return_data:
            return {
                "metrics": metrics,
                "data": datasets.Dataset.from_list(relabeled_data),
            }
        return {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
