import optuna
import yaml
import time
import subprocess
import numpy as np
import os
import argparse
from loguru import logger
# need check
def objective(trial):
    cfg = yaml.load(open(f"opt_configs/{conf_name if conf_name else model}.yml" ), Loader=yaml.FullLoader) 

    # some come search space
    cfg["lr"] = trial.suggest_loguniform("lr", 5e-4, 5e-3)

    # A common experience, momentum is good for ogbg-molhiv
    cfg['optimizer']= "momentum" if "hiv" in dataset else "adamw" 

    # need be optimized. Generally, plateau is worst
    cfg['scheduler']= trial.suggest_categorical('scheduler',['cosine', "onecycle"])

    # 128 is good generally, but also need be optimized
    dim_scope = [64, 32] if "mpn" in args.model else [128, 64]
    cfg['d_model'] = trial.suggest_categorical("d_model", [128, 64])

    # Default
    cfg['num_encoder_layers'] = trial.suggest_categorical('num_encoder_layers',[1,2,3,4])
    
    # Default
    cfg['gnn_num_layer'] = trial.suggest_categorical('gnn_num_layer',[2,3,4,5,6])

    # Fixed Settings
    cfg['gnn_virtual_node'] = True if "pcba" in dataset else False
    
    # need to be optimized. Generally, higher, better
    dim_scope = [64, 32, 128] if "IMDB" in args.dataset or "MUTAG" in args.dataset else [128, 256, 300, 512, 768]
    cfg['gnn_emb_dim'] = trial.suggest_categorical('gnn_emb_dim', dim_scope)

    # need to be optimizerd, Generally, better, 
    cfg['gnn_dropout'] = trial.suggest_loguniform("gnn_dropout", 0.1, 0.5)

    cfg['gnn_JK'] = trial.suggest_categorical('gnn_JK',['cat','last','sum'])

    # dataset need be indicated, ogbg-mol*, NCI, NCI109, MUTAG, IMDB-B, IMDB-M and so on
    cfg['dataset'] = dataset

    # 参数调整还得改 啊啊啊啊啊啊啊啊啊啊啊
    if model == "ccp":
        cfg['cheads'] = trial.suggest_categorical('cheads',[1,2,4])
        target = trial.suggest_categorical("target", [64, 32, 16])
        cfg['reduction_list'] = f"{target * cfg['cheads']}"
        cfg['cfeat_drop'] = trial.suggest_categorical('cfeat_drop',[0, 0.1, 0.2, 0.3])
        cfg['final_dropout'] = trial.suggest_loguniform("final_dropout", 0.2, 0.8)
        cfg['num_ccp'] = trial.suggest_categorical("num_ccp", [1,2,3])
        model_str = f"chd{cfg['cheads']}-rd{cfg['reduction_list']}-fdrop{cfg['cfeat_drop']}-fdrop{cfg['final_dropout']}"
    elif model == "mpn":
        cfg['fuse_type'] = trial.suggest_categorical('fuse_type', ['sum_fc', 'concat'])
        cfg['sf_type']= "GCP"
        cfg['sfheads']= trial.suggest_categorical('sfheads', [1,2,3,4])
        if "hiv" in dataset:
            cfg['remove_mean'] = False
        elif "NCI" in dataset:
            cfg['remove_mean'] = True
        else:
            cfg['remove_mean'] = trial.suggest_categorical('remove_mean', [True, False])
        model_str = f"ft{cfg['fuse_type']}-sfhd{cfg['sfheads']}-rm{cfg['remove_mean']}"
    elif model == "dropmpn":
        cfg['fuse_type'] = trial.suggest_categorical('fuse_type', ['sum_fc', 'concat'])
        cfg['sf_type']= "dropGCP"
        cfg['sfheads']= trial.suggest_categorical('sfheads', [1,2,3,4])
        if "hiv" in dataset:
            cfg['remove_mean'] = False
        elif "NCI" in dataset:
            cfg['remove_mean'] = True
        else:
            cfg['remove_mean'] = trial.suggest_categorical('remove_mean', [True, False])
        cfg['cfeat_drop'] = trial.suggest_categorical('cfeat_drop',[0, 0.1, 0.2, 0.3])
        cfg['final_dropout'] = trial.suggest_loguniform("final_dropout", 0.2, 0.8)
        model_str = f"ft{cfg['fuse_type']}-sfhd{cfg['sfheads']}-rm{cfg['remove_mean']}-fdrop{cfg['cfeat_drop']}-fdrop{cfg['final_dropout']}"
    elif model == "sfp":
        cfg['fuse_type'] = trial.suggest_categorical('fuse_type', ['sum_fc', 'concat'])
        cfg['sf_type']= "MGCrP"
        cfg['norm_type'] = "sqrt_d"
        cfg['svNum'] = 1
        cfg['iterNum'] = trial.suggest_categorical('iterNum', [2,3])
        if "hiv" in dataset:
            cfg['remove_mean'] = False
        elif "NCI" in dataset:
            cfg['remove_mean'] = True
        else:
            cfg['remove_mean'] = trial.suggest_categorical('remove_mean', [True, False])
        cfg['sfheads']= trial.suggest_categorical('sfheads', [1,2,3,4])
        cfg['n_alpha'] = trial.suggest_loguniform("n_alpha", 0.2, 0.8)
        cfg['regular'] = trial.suggest_loguniform("regular", 0.2, 0.8)
        cfg['cfeat_drop'] = trial.suggest_categorical('cfeat_drop',[0, 0.1, 0.2, 0.3])
        cfg['final_dropout'] = trial.suggest_loguniform("final_dropout", 0.2, 0.8)
        model_str = f"ft{cfg['fuse_type']}-sfhd{cfg['sfheads']}-rm{cfg['remove_mean']}-fdrop{cfg['cfeat_drop']}-fdrop{cfg['final_dropout']}-nt{cfg['norm_type']}-sN{cfg['svNum']}-iN{cfg['iterNum']}-na{cfg['n_alpha']}-rg{cfg['regular']}"
    name = f"l{cfg['num_encoder_layers']}-lr{cfg['lr']}-op{cfg['optimizer']}-dm{cfg['d_model']}-jk_{cfg['gnn_JK']}-sche_{cfg['scheduler']}-gl{cfg['gnn_num_layer']}-ghd{cfg['gnn_emb_dim']}-gdrop{cfg['gnn_dropout']}"
    if cfg['gnn_virtual_node']:
        name += '-vn'
    name = f"{cfg['dataset']}-{model}-" + model_str +"-"+ name
    cfg_name = f"auto_configs/{name}.yml"

    with open(cfg_name, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
    # PCBA epochs need be opt
    cmd = f"CUDA_VISIBLE_DEVICES={args.device} python main.py --configs auto_configs/{name}.yml --run_name {name} --epochs {args.epochs} --flag {args.flag}"
    print(cmd)
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf8').stdout
    for line in pipe:
        print(line)
        if "Report:" in line:
            line = line.strip("\n")
            info = line.split("Report:")[1].split(",")
            epoch = int(info[1])
            test_metric = float(info[0])
            trial.report(test_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if not os.path.exists(f"temp_r_{args.flag}"):
        raise NotImplementedError
    with open(f"temp_r_{args.flag}") as f:
        lines = f.readlines()
        res = float(lines[-1].split(" ± ")[0])
    return res

def callback(study, trial):
    pass

# This run in 11 server for tmp.

parser = argparse.ArgumentParser(description='main')
parser.add_argument("--conf", type=str)
parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
parser.add_argument("--model", type=str, choices=["ccp","mpn","sfp","dropmpn"],default="ccp")
parser.add_argument("--n_trials", type=int, default=41)
parser.add_argument("--flag", type=str, default="relax") # find the unique study sqlite
parser.add_argument("--epochs", type=int, default=66) # find the unique study sqlite
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
conf_name = args.conf
model = args.model
dataset = args.dataset
flag = args.flag
conf_name = args.conf
# 
study = optuna.create_study(study_name=f"{args.dataset}_{args.model}_{args.n_trials}_{args.flag}",
                            direction="maximize",
                            # storage=f'sqlite:///sqlite/{args.dataset}_{args.model}_{args.n_trials}_{args.flag}.db',
                            # load_if_exists=True,
                            pruner=optuna.pruners.MedianPruner())  # Create a new study.
study.optimize(objective, n_trials=args.n_trials)  # Invoke optimization of the objective function.
