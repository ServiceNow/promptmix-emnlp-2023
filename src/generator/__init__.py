import numpy as np, re

from .augment_slices import openai_chat_complete
from . import eda_utils


def get_generator(exp_dict):
    name = exp_dict.get("gen_engine", "eda_engine")
    n_gen = exp_dict.get("n_gen", 1)
    hparams = dict(
        generator_naug=n_gen,
        generator_engine=exp_dict["gen_engine_name"],
        generator_temp=exp_dict["gen_engine_temp"],
        generator_promptsize=exp_dict["prompt_size"],
        generator_prompttype=exp_dict.get("prompt_type", None),
    )
    if name == "eda_engine":
        return EDAEngine(hparams)

    elif name == "gpt3_engine":
        return GPT3Engine(hparams)

    elif name == "human":
        return HumanEngine(hparams)


def sample_alpha_from_peaked_distribution(n_aug):
    """sample from a distribution between 0.5 and 1.0 with a peak near 1.0"""
    return [round_x(((x * 0.5) + 0.5)) for x in np.random.beta(5, 2, n_aug)]


def round_x(x):
    """Round a number to the nearest 0.05"""
    return round(x * 20) / 20


def build_prompt(
    seed_intent,
    samples,
    prompt_type=None,
    n_gen=1,
    other_samples_dict=None,
    intent2desc=None,
):
    # alphas will be used for prompt_type=mixed, desc_w_mixing
    alphas = sample_alpha_from_peaked_distribution(n_gen)
    # alphas = [1.0, 0.75, 0.6, 1.0, 0.75, 0.6, 1.0, 0.75, 0.6, 1.0]
    print(f"Seed intent: {seed_intent}")
    if prompt_type is None:
        return (
            f"The following sentences belong to the same category '{seed_intent}':\n"
            + "\n".join([f"Example {i+1}: {t}" for i, t in enumerate(samples)])
            + f"\nExample {len(samples)+1}:"
        )
    elif prompt_type == "acl_prompt_w_desc":
        return (
            f"Consider an intent {seed_intent}, which is about {intent2desc[seed_intent]}. "
            + f"The following sentences belong to the same category '{seed_intent}':\n"
            + "\n".join([f"Example {i+1}: {t}" for i, t in enumerate(samples)])
            + f"\nExample {len(samples)+1}:"
        )
    elif prompt_type == "desc_w_mixing":
        all_intents = list(other_samples_dict.keys())
        # donot select seed_intent as other_intent
        other_intent = np.random.choice(
            list(set(all_intents).difference({seed_intent}))
        )
        print(f"other_intent: {other_intent}")
        input_prompt = (
            "Consider the task of classifying between the following intents:\n"
        )
        for idx, intent in enumerate(list(set(all_intents + [seed_intent]))):
            input_prompt += (
                f"{idx+1}. {intent}, which is about {intent2desc[intent]}.\n"
            )
        input_prompts = []
        for alpha in alphas:
            prompt = input_prompt + (
                f"Generate a diverse set of 3 short utterances "
                + f"where each utterance belongs {int(alpha*100)}% to {seed_intent} and {int((1-alpha)*100)}% to {other_intent}.\n"
                + "Example 1:"
            )
            input_prompts.append((alpha, prompt))
        return input_prompts
    elif prompt_type == "desc_no_mixing":
        input_prompt = (
            "Consider the task of classifying between the following intents:\n"
        )
        all_intents = list(set(list(other_samples_dict.keys()) + [seed_intent]))
        for idx, intent in enumerate(all_intents):
            input_prompt += (
                f"{idx+1}. {intent}, which is about {intent2desc[intent]}.\n"
            )
        return input_prompt + (
            f"Generate a diverse set of {n_gen} short utterance(s) "
            + f"where each utterance belongs to {seed_intent}.\n"
            + "Example 1:"
        )
    elif prompt_type == "desc_w_examples":
        input_prompt = "Consider the task of classifying between the following classes (along with some examples):\n"
        all_intents = list(set(list(other_samples_dict.keys()) + [seed_intent]))
        for idx, intent in enumerate(all_intents):
            texts = (
                samples.tolist()
                if intent == seed_intent
                else other_samples_dict[intent]
            )
            np.random.shuffle(texts)
            examples = "\n".join(["- " + t for t in texts])
            input_prompt += (
                f"{idx+1}. {intent}, which is about {intent2desc[intent]}. "
                + "Some examples of utterances include:\n"
                + f"{examples}\n"
            )
        return input_prompt + (
            f"\nGenerate a diverse set of {n_gen} short utterance(s) "
            + f"where each utterance belongs to the class {seed_intent}.\n"
            + "Example 1:"
        )
    elif prompt_type == "quick_baseline":
        seed_samples = [(seed_intent, s) for s in samples]
        for intent, texts in other_samples_dict.items():
            intent = intent.replace("_", " ")
            seed_samples += [(intent, t) for t in texts]
        np.random.shuffle(seed_samples)
        input_prompt = "Continue the following pattern:\n\n"
        input_prompt += "\n".join(
            [f"Category: {i}\nSentence: {t}" for i, t in seed_samples]
        )
        input_prompt += f"\nCategory: {seed_intent}\nSentence:"
        return input_prompt

    # MIXED PROMPT with alpha degree of mixing
    # build a prompt with examples from all intents
    all_intents = list(other_samples_dict.keys())
    # donot select seed_intent as other_intent
    other_intent = np.random.choice(list(set(all_intents).difference({seed_intent})))
    print(f"other_intent: {other_intent}")
    other_samples_dict.update({seed_intent: samples})
    input_prompt = "Consider the task of classifying between the following intents (along with some examples):\n"
    utterance_lens = []  # lengths of examples in the seed intent and other_intent
    for idx, intent in enumerate(list(set(all_intents + [seed_intent]))):
        texts = other_samples_dict[intent]
        np.random.shuffle(texts)
        examples = "\n".join(["- " + t for t in texts])
        input_prompt += (
            f"{idx+1}. {intent}, which is about {intent2desc[intent]}. "
            + "Some examples of utterances include:\n"
            + f"{examples}\n"
        )
        if intent in [seed_intent, other_intent]:
            utterance_lens += [len(t.split()) for t in texts]
    avg_wrds = np.round(np.mean(utterance_lens)).astype(int)
    input_prompts = []
    for alpha in alphas:
        prompt = input_prompt + (
            f"\nGenerate a diverse set of 3 short utterances "
            + f"where each utterance belongs {int(alpha*100)}% to {seed_intent} and {int((1-alpha)*100)}% to {other_intent}.\n"
            + "Example 1:"
        )
        input_prompts.append((alpha, prompt))
    return input_prompts


class Engine:
    def generate(self):
        raise NotImplementedError

    def select_texts(self, train_set, intent, exp_dict):
        """
        returns seed examples for a given intent
        """
        n_train = len(train_set["intent_names"])
        # Get texts corresponding to intent
        text_list = train_set["text"]
        intent_list = train_set["intent_names"]
        # filter utterances of intent
        texts = [text_list[i] for i in range(n_train) if intent == intent_list[i]]
        # randomly select n sentences
        n = min(len(texts), exp_dict["prompt_size"])
        texts = np.random.choice(texts, n, replace=False)
        return texts

    def mixed_prompt_gen(
        self, seed_intent, texts, other_intents, train_set, intent2desc, exp_dict
    ):
        raise NotImplementedError

    def predict(self, question):
        raise NotImplementedError

    def generate_data(
        self,
        train_set,
        ds_loader,
        exp_dict,
        tokenizer,
        preprocess_aug_fn,
        intent_names=None,
        model=None,
    ):
        n_gen = exp_dict.get("n_gen", 1)
        if intent_names is None:
            intent_names = [v for v in ds_loader.id2name.values() if v not in ["oos"]]
        name2id = {v: k for k, v in ds_loader.id2name.items()}
        generated = {}
        seed_examples = {}
        for intent in intent_names:
            # Get texts corresponding to intent
            texts = self.select_texts(train_set, intent, exp_dict)

            if isinstance(self, HumanEngine):  # upper bound exp
                gen = self.generate({intent: texts}, ds_loader.full_dataset, name2id)
                if len(gen[intent]) == 0:
                    continue
            else:
                if self.prompt_type is None:
                    gen = self.generate({intent: texts})
                else:
                    # mixed_prompt_gen prepares the set of other_samples before calling the generate function
                    other_intents = [i for i in intent_names if i != intent]
                    gen = self.mixed_prompt_gen(
                        intent,
                        texts,
                        other_intents,
                        train_set,
                        ds_loader.name2desc,
                        exp_dict,
                    )
                    # use the model to get distance of generated samples from each intent
                    # gen = model.retain_most_uncertain(gen)

            seed_examples.update({intent: texts.tolist()})
            generated.update(gen)

        dataset = preprocess_aug_fn(generated, seed_examples, tokenizer, name2id)
        return dataset


class HumanEngine(Engine):
    def __init__(self, hparams):
        self.hparams = hparams
        self.n_aug = hparams.get("generator_naug")

    def fetch_new_samples(self, full_dataset, seed_intent_id, samples, n_aug):
        seed_pool = [
            t
            for i, t in enumerate(full_dataset["train"]["text"])
            if full_dataset["train"]["intent"][i] == seed_intent_id
        ]
        if set(samples.tolist()) == set(seed_pool):
            print(f"no new samples to fetch for intent_id: {seed_intent_id}")
            return []
        new_samples = []
        while len(new_samples) < n_aug:
            found_new = False
            while not found_new:
                idx = np.random.choice(len(seed_pool))
                text = seed_pool[idx]
                if text not in new_samples + samples.tolist():
                    new_samples.append(text)
                    found_new = True
        return new_samples

    def generate(self, dataset, full_dataset, name2id):
        ret = {}
        for seed_intent, samples in dataset.items():
            # fetch naug new examples from the training set not already
            # added to the "generated" dataset
            ret[seed_intent] = self.fetch_new_samples(
                full_dataset,
                int(name2id[seed_intent]),
                samples,
                self.n_aug,
            )
        return ret


class GPT3Engine(Engine):
    def __init__(self, hparams):
        self.hparams = hparams
        self.temp = hparams.get("generator_temp", 0.7)
        self.n_aug = hparams.get("generator_naug")
        self.engine = hparams.get("generator_engine")
        self.prompt_size = hparams.get("generator_promptsize", 40)
        self.prompt_type = hparams.get("generator_prompttype", None)

    def mixed_prompt_gen(
        self,
        seed_intent,
        seed_texts,
        other_intents,
        train_set,
        intent2desc,
        exp_dict,
    ):
        """
        gen procedure when prompt_type=mixed
        """
        # select texts from one other intent
        other_dataset = {
            intent: self.select_texts(train_set, intent, exp_dict)
            for intent in other_intents
        }
        return self.generate(
            dataset={seed_intent: seed_texts},
            other_dataset=other_dataset,
            intent2desc=intent2desc,
        )

    def build_question_base(self, seed_set, intent2desc):
        question_base = "Consider the task of classifying between the following intents (along with some examples):\n"
        texts, intents = seed_set["text"], seed_set["intent_names"]
        for idx, intent in enumerate(intent2desc.keys()):
            _texts = [t for i, t in enumerate(texts) if intents[i] == intent]
            np.random.shuffle(_texts)
            examples = "\n".join(["- " + t for t in _texts])
            question_base += (
                f"{idx+1}. {intent}, which is about {intent2desc[intent]}. "
                + "Some examples of utterances include:\n"
                + f"{examples}\n"
            )
        return question_base

    def verify(self, question):
        valid_resp = False
        while not valid_resp:
            resp = openai_chat_complete(
                question,
                1,
                self.engine,
                0,
                top_p=1,
                max_tokens=1,
                logprobs=5,
            )
            if isinstance(resp[0], str):
                valid_resp = True
        # top_logprobs = resp[0].logprobs.top_logprobs[0]
        # pred = max(top_logprobs, key=top_logprobs.get).strip()
        return resp[0]

    def predict(self, question):
        valid_resp = False
        while not valid_resp:
            resp = openai_chat_complete(
                prompt=question, engine=self.engine, n=None, temp=0.0, top_p=1
            )
            if resp[0]:
                valid_resp = True
        # preds = [
        #     re.sub(r"^(\d+)\.", "", p).strip().lower() for p in resp[0].splitlines()
        # ]
        # return preds
        return resp[0].strip()

    def request_per_prompt(self, input_prompt, samples):
        print(input_prompt)
        final_resp = []
        while len(final_resp) < self.n_aug:
            resp = openai_chat_complete(
                input_prompt,
                # None for mixed as all examples are generated in a single request.
                self.n_aug
                if self.prompt_type
                not in ["mixed", "desc_w_mixing", "desc_no_mixing", "desc_w_examples"]
                else None,
                self.engine,
                self.temp,
                top_p=1,
            )
            valid_sent = True
            if self.prompt_type in [
                "mixed",
                "desc_no_mixing",
                "desc_w_mixing",
                "desc_w_examples",
            ]:
                resp = [r.replace('"', "").strip() for r in resp[0].splitlines()]
                for s in resp:
                    if "Generate a diverse set of" in s:
                        print(f"Found an invalid generation {s}")
                        valid_sent = False
                if not valid_sent:
                    continue
            _resp = []
            for s in resp:
                s = re.sub(r"^Example (\d+)\:", "", s).strip()  # remove "Example 1:"
                s = re.sub(r"^\d\.", "", s).strip()  # remove "1."
                s = re.sub(
                    r"\(\d+\%.+\)", "", s
                ).strip()  # remove (85% entity 15% location)
                if not s or s in (final_resp + samples.tolist()):
                    continue
                _resp.append(s)
            resp = _resp
            final_resp += list(set(resp))
        return final_resp[: self.n_aug]

    def generate(self, dataset, other_dataset=None, intent2desc=None):
        """
        other_dataset is used when prompt_type=mixed. it generates a prompt
        with examples from multiple intents
        """

        print(f"Engine: {self.engine.upper()} | Temp: {self.temp}")
        ret = {}
        for seed_intent, samples in dataset.items():
            # total samples in prompt
            prompt_size = min(len(samples), self.prompt_size)
            texts = np.random.choice(samples, size=prompt_size, replace=False)
            input_prompt = build_prompt(
                seed_intent,
                texts,
                self.prompt_type,
                self.n_aug,
                other_samples_dict=other_dataset,
                intent2desc=intent2desc,
            )
            if self.prompt_type in ["desc_w_mixing", "mixed"]:
                assert isinstance(input_prompt, list)
                final_resp = []
                # prompts for different alpha values
                for alpha, prompt in input_prompt:
                    resp = self.request_per_prompt(prompt, samples)
                    resp = np.random.choice(resp, size=1, replace=False).tolist()
                    if isinstance(resp, dict):
                        resp.update({"alpha": alpha})
                    else:
                        resp = {"text": resp, "alpha": alpha}
                    final_resp.append(resp)
            else:
                final_resp = self.request_per_prompt(input_prompt, samples)
            ret[seed_intent] = final_resp
        return ret


class EDAEngine(Engine):
    def __init__(self, hparams):
        self.haparams = hparams
        self.n_aug = hparams.get("generator_naug")
        self.alpha = hparams.get("alpha", 0.05)
        self.prompt_type = None

    def generate(self, dataset):
        ret = {}
        for seed_intent, samples in dataset.items():
            ret[seed_intent] = []
            for sample in samples:
                ret[seed_intent] += eda_utils.eda(
                    sentence=sample,
                    alpha_sr=self.alpha,
                    alpha_ri=self.alpha,
                    alpha_rs=self.alpha,
                    p_rd=self.alpha,
                    num_aug=self.n_aug,
                )
        return ret


if __name__ == "__main__":
    samples = [
        "what is the status of this flight?",
        "is the flight's name too long for flight number?",
        "what is flight 583 departing from john stow?is its time to go?",
        "im in the layover of flight 442",
        "what is going to be the impact of flight 1898 in san francisco?",
        "whats the departure time for flight BP483 from houston central?",
    ]
    hparams = dict(
        generator_naug=10,
        generator_prompttype="in_batch",
        generator_engine="text-davinci-003",
        generator_temp=1.0,
    )
    # engine = GPTNeoEngine(hparams)
    # engine = EDAEngine(hparams)
    # hparams = dict(generator_naug=10, generator_engine="ada")
    engine = GPT3Engine(hparams)
    generated = engine.generate({"flight_status": samples})
    for g in generated["flight_status"]:
        print(g)
    pass
