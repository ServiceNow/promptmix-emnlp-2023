import argparse, os, exp_configs
from haven import haven_wizard as hw
from scripts import trainval, relabel, gpt_nn_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group_list",
        nargs="+",
        default="poc_naive",
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

    if args.exp_group_list == ["poc_naive"]:
        func = trainval.main
    elif args.exp_group_list == ["poc_rlbl"]:
        func = relabel.main
    elif args.exp_group_list == ["gpt_nn_baseline"]:
        func = gpt_nn_baseline.main
    else:
        func = None

    hw.run_wizard(
        func=func,
        exp_groups=exp_configs.EXP_GROUPS,
        job_config=job_config,
        python_binary_path=args.python_binary,
        # python_file_path=f"-m runners.{file_name}",
        use_threads=True,
        args=args,
        results_fname="results/active_nlp.ipynb",
    )
