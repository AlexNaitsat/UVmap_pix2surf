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



# preparer configuration
cfg = get_cfg(interactive=False)

# prepare dataset
DatasetClass = get_dataset(cfg.DATASET)
dataloader_dict = dict()
for mode in cfg.MODES:
    phase_dataset = DatasetClass(cfg, mode=mode)
    dataloader_dict[mode] = DataLoader(phase_dataset, batch_size=cfg.BATCHSIZE,
                                       shuffle=cfg['SHUFFLE'] if mode in ['train'] else False,
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