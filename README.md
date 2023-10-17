# PromptMix

This is repository for the EMNLP 2023 paper PromptMix: A Class Boundary Augmentation Method for Large Language Model Distillation.

```bibtex
@inproceedings{sahu-etal-2023-promptmix,
    title = "PromptMix: A Class Boundary Augmentation Method for Large Language Model Distillation.",
    author = "Sahu, Gaurav  and
      Vechtomova, Olga  and
      Bahdanau, Dzmitry and
      Laradji, Issam
      ",
    booktitle = "Empirical Methods in Natural Language Processing 2023 (EMNLP)",
    year = "2023"
}
```

## Running experiments

### 1. Clone the repo and setup up a virtualenv with python>=3.9

Below instructions are for conda/miniconda but you can also use virtualenv

```
git clone git@github.com:ServiceNow/PromptMix.git
cd PromptMix

conda create -n promptmix python=3.9
conda activate promptmix
pip install -r requirements.txt
```


### 2. Get the Datasets

Prepare the datasets by running the following command

```bash
python -m scripts.compile_ds -d subj
```

Set `-d` to "tc" for the twitter complaints dataset and "trec6" for the TREC6 dataset. To use the banking77 dataset, download the zip file [here](https://drive.google.com/file/d/1dPm68g7kAm30h8oLlV665IOoeiWqCaw-/view?usp=sharing) and unzip it into the `data` directory.

### 3. Train promptmix pipeline & Validate

Run the following command to run the promptmix pipeline (on SUBJ dataset)

```bash
# Run the experiment w/o relabeling (to get A1)
$(which python) -m main -e poc_naive -sb /path/to/save/directory -j 0 --python_binary $(which python)
# perform relabeling (to get A2)
$(which python) -m main -e poc_rlbl -sb /path/to/save/directory -j 0 --python_binary $(which python)
```

Run the following to get the NN+GPT3.5 results

```bash
$(which python) -m main -e gpt_nn_baseline -j 0 --python_binary $(which python)
```

Argument Descriptions:
```
-e  [Experiment group to run] 
-sb [Directory where the experiments are saved]
-j [whether or not to use toolkit]
```

You can modify [exp_configs/promptmix_exps.py](exp_configs/promptmix_exps.py) to change the hyperparameters/use different datasets.

### 4. Results

Open `results/active_nlp.ipynb` to visualize the results.
