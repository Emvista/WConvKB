import datetime
import logging
import os
import contextlib as cl
from hptuner import generate_grid
from openke.config import Trainer, Tester
from openke.data import TrainDataLoader, TestDataLoader
from openke.module.loss import MarginLoss
from openke.module.model import TransE
from openke.module.strategy import NegativeSampling

# Configuration =============================================================

dataset_dir = "../data/RezoJDM16K/"

# hyperparameters = {
#     # TrainDataLoader
#     "batch_size": [10_000],
#     "sampling_mode": ["normal"],
#     "bern_flag": [1],
#     "filter_flag": [1],
#     "neg_ent": [35],
#     "neg_rel": [0],
#     # Model
#     "dim": [50, 100],
#     "p_norm": [1, 2],
#     "norm_flag": [True],
#     # NegativeSampling
#     "margin": [1.0, 5.0],
#     # Trainer
#     "train_times": [3_000],
#     "alpha": [1e-4, 5e-4, 1e-3],
#     "opt_method": ["sgd"]
# }

hyperparameters = {
    # TrainDataLoader
    "batch_size": [10_000],
    "sampling_mode": ["normal"],
    "bern_flag": [1],
    "filter_flag": [1],
    "neg_ent": [35],
    "neg_rel": [0],
    # Model
    "dim": [50],
    "p_norm": [1],
    "norm_flag": [True],
    # NegativeSampling
    "margin": [1.0],
    # Trainer
    "train_times": [1],
    "alpha": [1e-4],
    "opt_method": ["sgd"]
}

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

today = datetime.date.today()

# entry point ===============================================================

# Make sure a few things are in place ---------------------------------------

# Do the job ----------------------------------------------------------------

grid = list(generate_grid(hyperparameters))

# dataloader for test
test_dataloader = TestDataLoader(dataset_dir, "link")

run_ptrn = "TransE_{date}_{hps}"

for run_num, hp in enumerate(grid):
    # preparation
    run_id = f"TransE_{today.year}{today.month}{today.day}_{run_num}"
    run_dir = os.path.join(dataset_dir, "TransE", run_id)
    os.makedirs(run_dir)
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=dataset_dir,
        batch_size=hp["batch_size"],
        threads=8,
        sampling_mode=hp["sampling_mode"],
        bern_flag=hp["bern_flag"],
        filter_flag=hp["filter_flag"],
        neg_ent=hp["neg_ent"],
        neg_rel=hp["neg_rel"]
    )
    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=hp["dim"],
        p_norm=hp["p_norm"],
        norm_flag=hp["norm_flag"]
    )
    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=hp["margin"]),
        batch_size=train_dataloader.get_batch_size()
    )
    # train the model
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=hp["train_times"],
        alpha=hp["alpha"],
        use_gpu=False
    )
    with open(os.path.join(run_dir, "train.out"), "w") as out, \
            open(os.path.join(run_dir, "train.err"), "w") as err, \
            cl.redirect_stdout(out), \
            cl.redirect_stderr(err):
        trainer.run()
        transe.save_checkpoint(os.path.join(run_dir, "transe_final.ckpt"))
        # test the model
        tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
        tester.run_link_prediction(type_constrain=False)
