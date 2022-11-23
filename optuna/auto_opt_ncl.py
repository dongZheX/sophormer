import optuna
import yaml
import time
import subprocess
import numpy as np
import os

conf_name = "base_ncl"

def objective(trial):
    cfg = yaml.load(open(f"{conf_name}.yml"), Loader=yaml.FullLoader)
    cfg["lr"] = trial.suggest_loguniform("lr", 5e-4, 5e-3)
    cfg["cheads"] = trial.suggest_categorical('cheads',[1,2])
    cfg['optimizer']= trial.suggest_categorical('optimizer',['momentum', "adamw"])
    cfg['scheduler']= trial.suggest_categorical('scheduler',['cosine', "onecycle"])
    cfg['d_model'] = trial.suggest_categorical("d_model", [300, 128, 256, 64])
    target = trial.suggest_categorical("target", [64, 32,16])
    cfg['reduction_list'] = f"{target * cfg['cheads']}"
    cfg['gnn_JK'] = trial.suggest_categorical('gnn_JK',['cat','last','sum'])
    cfg['use_weight'] = trial.suggest_categorical('use_weight',[True, False])
    cfg['use_bn'] = trial.suggest_categorical('use_bn',[True, False])
    cfg['last_bn'] = trial.suggest_categorical('last_bn',[True, False])
    cfg['cattn_drop'] = trial.suggest_categorical('cattn_drop',[0, 0.1, 0.2, 0.3])
    cfg['cfeat_drop'] = trial.suggest_categorical('cfeat_drop',[0, 0.1, 0.2, 0.3])
    cfg['final_dropout'] = trial.suggest_loguniform("final_dropout", 0.1, 0.8)
    cfg['num_encoder_layers'] = trial.suggest_categorical('num_encoder_layers',[1,2,3,4])
    cfg['gnn_num_layer'] = trial.suggest_categorical('gnn_num_layer',[3,4,5,6])
    cfg['gnn_virtual_node'] = trial.suggest_categorical('gnn_virtual_node',[True, False])
    cfg['gnn_emb_dim'] = trial.suggest_categorical('gnn_emb_dim',[128, 256, 300, 512, 768])
    cfg['gnn_dropout'] = trial.suggest_loguniform("gnn_dropout", 0.2, 0.8)
    name = f"{cfg['dataset']}-base-ccp-l{cfg['num_encoder_layers']}-lr{cfg['lr']}-ch{cfg['cheads']}-op{cfg['optimizer']}-dm{cfg['d_model']}-rd{cfg['reduction_list']}-jk_{cfg['gnn_JK']}-cad{cfg['cattn_drop']}-cfd{cfg['cfeat_drop']}-fdp{cfg['final_dropout']}-sche_{cfg['scheduler']}-gl{cfg['gnn_num_layer']}-ghd{cfg['gnn_emb_dim']}-gdrop{cfg['gnn_dropout']}"
    if cfg['use_weight']:
        name += "-uw"
    if cfg['use_bn']:
        name += "-ub"
    if cfg['last_bn']:
        name += '-lb'
    if cfg['gnn_virtual_node']:
        name += '-vn'
    cfg_name = f"auto_configs/{name}.yml"

    with open(cfg_name, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    cmd = f"CUDA_VISIBLE_DEVICES=2 python main.py --configs auto_configs/{name}.yml --run_name {name} --epochs 66"
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
        result_str += "exp\t%s\n" % "NCI1-CCP"
        for key, value in study.best_params.items():
            result_str += "param:%s\t%s\n" % (key, str(value))
        result_str += "Best Val-acc: %s\n" % study.best_trial.value
        with open("exps/final_result_NCI1.txt","a+") as f:
            f.write(result_str+"\n")


study = optuna.create_study(direction="maximize")  # Create a new study.
study.optimize(objective, n_trials=100, callbacks=[callback])  # Invoke optimization of the objective function.
