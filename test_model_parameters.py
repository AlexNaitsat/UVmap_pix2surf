"""
Main program
May the Force be with you.

This main file is used on slurm server without interactive check of config
"""
from torch.utils.data import DataLoader

from dataset import get_dataset
from logger import get_logger
from core.models import get_model
from core.trainer import Trainer
from config import get_cfg
import os
import glob
import numpy as np

def zip_results(cfg, zip_model=False):
    # archive these results wihtout model
    pwd= os.getcwd()
    os.chdir(pwd + "/log")
    os.system("cd log/")
    zip_log_cmd = "zip -r " + cfg['LOG_DIR'] + ".zip " + cfg['LOG_DIR'] + " -x \'" + "*model*" + "\'"
    os.system(zip_log_cmd)

    # myhost = os.uname()[1] #tried to download it to local client - doesn't work
    ## scp log results -- check how to do it here https://unix.stackexchange.com/questions/2857/copy-a-file-back-to-local-system-with-ssh
    if zip_model:
        model_file_paths= glob.glob("log/" + cfg['LOG_DIR'] + "/model/*_latest.model/")
        if model_file_paths:
            latest_model_file = os.path.split(model_file_paths[-1])
            zip_model_cmd = "zip -r " + cfg['LOG_DIR'] + "_" + latest_model_file + " " + model_file_paths[-1]
    os.chdir(pwd)




def run_batch_wrapper(cfg):
    # prepare dataset
    DatasetClass = get_dataset(cfg.DATASET)
    dataloader_dict = dict()
    for mode in cfg.MODES:
        phase_dataset = DatasetClass(cfg, mode=mode)
        debug_dataset = False
        if debug_dataset:
            first_item =phase_dataset[0]

        print('Load initial weights from ', cfg['MODEL_INIT_PATH'], '\n')
        #temporary disabled shuffle!!!
        dataloader_dict[mode] = DataLoader(phase_dataset, batch_size=cfg.BATCHSIZE,
                                           shuffle=False if mode in ['train'] else False,
                                           num_workers=cfg.DATALOADER_WORKERS, pin_memory=True,
                                           drop_last=True)

    # prepare models
    ModelClass = get_model(cfg.MODEL)
    model = ModelClass(cfg)

    # prepare logger
    LoggerClass = get_logger(cfg.LOGGER)
    logger = LoggerClass(cfg)

    # register dataset, models, logger to trainer
    trainer = Trainer(cfg, model, dataloader_dict, logger)

    # start training
    epoch_total = cfg.EPOCH_TOTAL + (cfg.RESUME_EPOCH_ID if cfg.RESUME else 0)
    while trainer.do_epoch() <= cfg.EPOCH_TOTAL:
        pass
    #zip log dir without models
    zip_results(cfg)



# preparer configuration
#cfg = get_cfg(interactive=False)
cfg = get_cfg(filename = "small/mv_car_xyz_uv_diff_small_dataset.yaml", interactive=False)
train_epoch_num = cfg['EPOCH_TOTAL']
for XYZ_UV_DIFF_ALPHA in np.linspace(0.05,0.25,4):
    for XYZ_UV_DIFF_COEF in np.linspace(0.05,0.5,5):
        print(f'--------------------------- training with params= (XYZ_UV_DIFF_ALPHA,XYZ_UV_DIFF_ALPHA) = ({XYZ_UV_DIFF_ALPHA:.2f},{XYZ_UV_DIFF_COEF:.2f}) ---------------------------')
        cfg['XYZ_UV_DIFF_ALPHA']  =XYZ_UV_DIFF_ALPHA
        cfg['XYZ_UV_DIFF_COEF'] = XYZ_UV_DIFF_COEF
        cfg['LOG_DIR'] = f"param_test/mv-car-xyz-uv-diff-small_dataset_ep{cfg['EPOCH_TOTAL']}_{XYZ_UV_DIFF_ALPHA:.2f}_{XYZ_UV_DIFF_COEF:.2f}"
        print(f" LOG_DIR =  {cfg['LOG_DIR']}\n")
        run_batch_wrapper(cfg)

cfg = get_cfg(filename = "small/render_mv_car_xyz_uv_diff_small_dataset.yaml", interactive=False)
for XYZ_UV_DIFF_ALPHA in np.linspace(0.05, 0.25, 4):
    for XYZ_UV_DIFF_COEF in np.linspace(0.05, 0.5, 5):
        cfg['XYZ_UV_DIFF_ALPHA']  =XYZ_UV_DIFF_ALPHA
        cfg['XYZ_UV_DIFF_COEF'] = XYZ_UV_DIFF_COEF
        print(f' -------------- rendering with params= (XYZ_UV_DIFF_ALPHA,XYZ_UV_DIFF_ALPHA) = ({XYZ_UV_DIFF_ALPHA:.2f},{XYZ_UV_DIFF_COEF:.2f}) --------------')
        cfg['LOG_DIR'] = f"param_test/render-mv-car-xyz-uv-diff-small_dataset_{XYZ_UV_DIFF_ALPHA:.2f}_{XYZ_UV_DIFF_COEF:.2f}"
        cfg['MODEL_INIT_PATH'] = f"log/param_test/mv-car-xyz-uv-diff-small_dataset_ep{train_epoch_num}_{XYZ_UV_DIFF_ALPHA:.2f}_{XYZ_UV_DIFF_COEF:.2f}/model/epoch_{train_epoch_num}_latest.model"
        print(f" LOG_DIR =  {cfg['LOG_DIR']}\n")
        print(f" MODEL_INIT_PATH =  {cfg['MODEL_INIT_PATH']}\n")
        run_batch_wrapper(cfg)

#writing command line arguments to open everything in a single tensorboard dashboard
f = open('tensorboard_cmd_args.txt', 'w')
f.write("\n#Tensorboard training metrics")
f.write("\ntensorboard  --logdir_spec  ")
is_first_speclogdir=True
for XYZ_UV_DIFF_ALPHA in np.linspace(0.05, 0.25, 4):
    for XYZ_UV_DIFF_COEF in np.linspace(0.05, 0.5, 5):
        if not is_first_speclogdir:
            f.write(",")#shoudl be no space between commas
        else:
            is_first_speclogdir = False
        speclogdir_arg = f"{XYZ_UV_DIFF_ALPHA:.2f}_{XYZ_UV_DIFF_COEF:.2f}:mv-car-xyz-uv-diff-small_dataset_ep{cfg['EPOCH_TOTAL']}_{XYZ_UV_DIFF_ALPHA:.2f}_{XYZ_UV_DIFF_COEF:.2f}"
        f.write(speclogdir_arg)
        print(speclogdir_arg)
f.close()

f = open('tensorboard_cmd_args.txt', 'a')
f.write("\n#Tensorboard renderings")
f.write("\ntensorboard  --logdir_spec  ")
is_first_speclogdir=True
for XYZ_UV_DIFF_ALPHA in np.linspace(0.05, 0.25, 4):
    for XYZ_UV_DIFF_COEF in np.linspace(0.05, 0.5, 5):
        if not is_first_speclogdir:
            f.write(",")
        else:
            is_first_speclogdir = False
        speclogdir_arg = f"{XYZ_UV_DIFF_ALPHA:.2f}_{XYZ_UV_DIFF_COEF:.2f}:render-mv-car-xyz-uv-diff-small_dataset_{XYZ_UV_DIFF_ALPHA:.2f}_{XYZ_UV_DIFF_COEF:.2f}"
        f.write(speclogdir_arg)
        print(speclogdir_arg)
f.close()