import optuna
import yaml
import time
import subprocess
import numpy as np
import os
import argparse
from loguru import logger

def objective(trial):
    cfg = yaml.load(open(conf_name, 'r'), Loader=yaml.FullLoader) 

    cfg['num_ccp'] = trial.suggest_categorical("num_ccp", [1,2,3,4,5,6])
    name = f"{conf_name.split('/')[1]}-num_ccp{cfg['num_ccp']}"
    cfg_name = f"auto_configs/{name}.yml"
    
    with open(cfg_name, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
    cmd = f"python main.py --configs auto_configs/{name}.yml --run_name {name} --epochs {args.epochs} --flag {args.flag}"
    print(cmd)
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf8').stdout
    for line in pipe:
        print(line)
        # if "Report:" in line:
        #     line = line.strip("\n")
        #     info = line.split("Report:")[1].split(",")
        #     epoch = int(info[1])
        #     test_metric = float(info[0])
        #     trial.report(test_metric, epoch)
        #     if trial.should_prune():
        #         raise optuna.TrialPruned()
    with open(f"temp_r_{args.flag}") as f:
        lines = f.readlines()
        res = float(lines[-1].split(" ± ")[0])
    return res

def callback(study, trial):
    pass

# This run in 11 server for tmp.

parser = argparse.ArgumentParser(description='main')
parser.add_argument("--conf", type=str)
parser.add_argument("--n_trials", type=int, default=41)
parser.add_argument("--flag", type=str, default="relax") # find the unique study sqlite
parser.add_argument("--epochs", type=int, default=66) # find the unique study sqlite
args = parser.parse_args()
conf_name = args.conf
flag = args.flag
# 
study = optuna.create_study(study_name=f"{args.conf.split('/')[1]}_{args.n_trials}_{args.flag}",
                            direction="maximize",
                            storage=f"sqlite:///sqlite/{args.conf.split('/')[1]}_{args.n_trials}_{args.flag}.db",
                            load_if_exists=True,)
                            # pruner=optuna.pruners.MedianPruner())  # Create a new study.
study.optimize(objective, n_trials=args.n_trials)  # Invoke optimization of the objective function.
