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
conf_name = "base_freeze_hiv"
# 不过不行，就需要按照最优参数训练一个gnn
def objective(trial):
    cfg = yaml.load(open(f"{conf_name}.yml"), Loader=yaml.FullLoader)
    cfg["lr"] = trial.suggest_loguniform("lr", 1e-4, 1e-3)
    cfg["cheads"] = trial.suggest_categorical('cheads',[1,2])
    target = trial.suggest_categorical("target", [64, 32,16])
    cfg['reduction_list'] = f"{target * cfg['cheads']}"
    cfg['final_dropout'] = trial.suggest_loguniform("final_dropout", 0.1, 0.8)
    cfg['freeze_type'] = trial.suggest_categorical("freeze_type", ["all","gnn","trans"])
    cfg['fix_epochs'] = trial.suggest_categorical("fix_epochs", [1, 2, 3, 5, 10])
    name = f"{cfg['dataset']}-base-ccp-freeze-lr{cfg['lr']}-ch{cfg['cheads']}-rd{cfg['reduction_list']}-fdp{cfg['final_dropout']}-freezet_{cfg['freeze_type']}"
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
        result_str += "exp\t%s\n" % "HIV-CCP-FREEZE"
        for key, value in study.best_params.items():
            result_str += "param:%s\t%s\n" % (key, str(value))
        result_str += "Best Val-acc: %s\n" % study.best_trial.value
        with open("exps/final_result_freeze_hiv.txt","a+") as f:
            f.write(result_str+"\n")


study = optuna.create_study(direction="maximize")  # Create a new study.
study.optimize(objective, n_trials=100, callbacks=[callback])  # Invoke optimization of the objective function.
