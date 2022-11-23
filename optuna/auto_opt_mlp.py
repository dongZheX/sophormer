import optuna
import yaml
import time
import subprocess
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument("--conf", type=str, default="base")
args = parser.parse_args()
conf_name = "base_hiv_mlp"

def objective(trial):
    cfg = yaml.load(open(f"{conf_name}.yml"), Loader=yaml.FullLoader)
    cfg["num_mlp"] = trial.suggest_categorical('num_mlp',[1,2,3,4,5])

    name = f"{cfg['dataset']}-base-ccp-mlp_num_mlp{cfg['num_mlp']}"
    cfg_name = f"auto_configs/{name}.yml"

    with open(cfg_name, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    cmd = f"python main.py --configs auto_configs/{name}.yml --run_name {name} --epochs 66"
    print(cmd)
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf8').stdout
    for line in pipe:
        print(line)
    with open(f"temp_r") as f:
        lines = f.readlines()
        res = float(lines[-1].split(" ± ")[0])

    return res

def callback(study, trial):
    if study.best_trial == trial:
        result_str = ""
        result_str += "exp\t%s\n" % "HIV-CCP"
        for key, value in study.best_params.items():
            result_str += "param:%s\t%s\n" % (key, str(value))
        result_str += "Best Val-acc: %s\n" % study.best_trial.value
        with open("exps/final_result.txt","a+") as f:
            f.write(result_str+"\n")


study = optuna.create_study(direction="maximize")  # Create a new study.
study.optimize(objective, n_trials=10, callbacks=[callback])  # Invoke optimization of the objective function.
