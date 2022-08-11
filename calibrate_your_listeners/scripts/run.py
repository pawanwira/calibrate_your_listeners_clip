"""
source init_env.sh
python scripts/run.py --dryrun

or with interactive

WANDB_CONSOLE="wrap" python scripts/run_ou.py --dryrun
"""

import os
import wandb
import getpass
import random, torch, numpy
from omegaconf import OmegaConf
import pytorch_lightning as pl
import hydra
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from calibrate_your_listeners.src.systems import (
    listener_system,
    speaker_system,
    speaker_system_clip,
)

from calibrate_your_listeners import constants


NAME2SYSTEM = {
    'l0': listener_system.ListenerSystem,
    's1_clip': speaker_system_clip.SpeakerCLIPSystem,
    's1': speaker_system.SpeakerSystem,
}

torch.backends.cudnn.benchmark = True

@hydra.main(config_path="../config", config_name="l0")
def run(config):
    if config.wandb_params.dryrun:
        # print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'
    os.environ['WANDB_CONSOLE']='wrap'

    seed_everything(
        config.training_params.seed,
        use_cuda=config.training_params.cuda)

    group_name = config.wandb_params.group_name

    wandb.init(
        project=config.wandb_params.project,
        entity="pawanwira",
        group=group_name,
        name=config.wandb_params.exp_name,
        config=OmegaConf.to_container(config, resolve=[True|False]),
        tags=config.wandb_params.tags
    )

    CKPT_DIR = os.path.join(constants.ROOT_DIR, 'src/models/checkpoints', config.wandb_params.exp_name)
    print(f"CKPT AT {CKPT_DIR}")
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        CKPT_DIR,
        save_top_k=-1,
        every_n_epochs=config.training_params.checkpoint_steps,
    )

    SystemClass = NAME2SYSTEM[config.pl.system]
    system = SystemClass(config)
    # print(f"wandb run directory is {wandb.run.dir}")

    # profiler = pl.profiler.AdvancedProfiler(output_filename="profile")
    # early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True) # TODO: temp, remove

    trainer = pl.Trainer(
        gpus=1,
        checkpoint_callback=ckpt_callback,
        max_epochs=int(config.training_params.num_epochs),
        # max_epochs = 1, 
        min_epochs=int(config.training_params.num_epochs),
        check_val_every_n_epoch=1,
        # profiler=profile
        # callbacks = [early_stop_callback] # TODO: temp, remove
    )

    trainer.fit(system)
    if config.pl.system == 'l0':
        system.save()

def seed_everything(seed, use_cuda=True):
    # print(f"seed {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    run()



