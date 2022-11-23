import optuna
import yaml
import time
import subprocess
import numpy as np
import os

conf_name = "base_hiv_sfp_norm"

def objective(trial):
    cfg = yaml.load(open(f"{conf_name}.yml"), Loader=yaml.FullLoader)
    cfg["lr"] = trial.suggest_loguniform("lr", 5e-4, 5e-3)
    cfg['scheduler']= trial.suggest_categorical('scheduler',['cosine', "onecycle"])
    cfg['d_model'] = trial.suggest_categorical("d_model", [128, 256, 64])
    cfg['gnn_JK'] = trial.suggest_categorical('gnn_JK',['cat','sum'])
    cfg['fuse_type'] = trial.suggest_categorical('fuse_type',["concat", "sum_fc"])
    cfg['remove_mean'] = trial.suggest_categorical('remove_mean',[True, False])
    cfg['sfheads'] = trial.suggest_categorical('sfheads',[1, 2, 4, 6])
    cfg['wr_dim'] = trial.suggest_categorical('wr_dim',[16, 24, 32, 64])
    cfg['num_encoder_layers'] = trial.suggest_categorical('num_encoder_layers',[1,2,3,4])
    cfg['gnn_num_layer'] = trial.suggest_categorical('gnn_num_layer',[3,4,5,6])
    cfg['norm_type'] = trial.suggest_categorical('norm_type',["svPN", "ePN", "LN", "sqrt_d","sqrt_t","None"])
    cfg['gnn_emb_dim'] = trial.suggest_categorical('gnn_emb_dim',[128, 256, 300, 512, 768])
    cfg['gnn_dropout'] = trial.suggest_loguniform("gnn_dropout", 0.2, 0.5)
    cfg['n_alpha'] = trial.suggest_loguniform("n_alpha", 0.2, 0.8)
    cfg['iterNum'] = trial.suggest_categorical('iterNum',[1,2,3])
    cfg['svNum'] = trial.suggest_categorical('svNum',[1,2,3])
    cfg['regular'] = trial.suggest_loguniform("regular", 0.2, 0.8)

    name = f"{cfg['dataset']}-base-sfp-norm-l{cfg['num_encoder_layers']}-lr{cfg['lr']}-sfh{cfg['sfheads']}-dm{cfg['d_model']}-jk_{cfg['gnn_JK']}-ft{cfg['fuse_type']}-rm{int(cfg['remove_mean'])}-wd{cfg['wr_dim']}-sche_{cfg['scheduler']}-gl{cfg['gnn_num_layer']}-ghd{cfg['gnn_emb_dim']}-gdrop{cfg['gnn_dropout']}-tn{cfg['norm_type']}-alp{cfg['n_alpha']}-iN{cfg['iterNum']}-sN{cfg['svNum']}-reg{cfg['regular']}"
    cfg_name = f"auto_configs/{name}.yml"

    with open(cfg_name, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    cmd = f"CUDA_VISIBLE_DEVICES=3 python main.py --configs auto_configs/{name}.yml --run_name {name} --epochs 66"
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
        result_str += "exp\t%s\n" % "HIV-SFP-NORM"
        for key, value in study.best_params.items():
            result_str += "param:%s\t%s\n" % (key, str(value))
        result_str += "Best Val-acc: %s\n" % study.best_trial.value
        with open("exps/final_result_sfp_hiv_norm.txt","a+") as f:
            f.write(result_str+"\n")


study = optuna.create_study(direction="maximize")  # Create a new study.
study.optimize(objective, n_trials=100, callbacks=[callback])  # Invoke optimization of the objective function.
