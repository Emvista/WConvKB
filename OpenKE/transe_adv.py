import contextlib as cl
import ctypes
import io
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from itertools import product
from typing import Mapping, Iterable, Any

import numpy as np

from openke.module.loss import SigmoidLoss
from openke.config import Trainer, Tester
from openke.data import TrainDataLoader, TestDataLoader
from openke.module.model import TransE
from openke.module.strategy import NegativeSampling

# Configuration =============================================================

dataset_dir = "../data/RezoJDM16K/"

hyperparameters = {
    # TrainDataLoader
    "batch_size": [10_000, 100_000],
    "sampling_mode": ["cross"],
    "bern_flag": [0],
    "filter_flag": [1],
    "neg_ent": [0, 64],
    "neg_rel": [0],
    # Model
    "dim": [50, 100],
    "p_norm": [1],
    "norm_flag": [True],
    "transe_margin": [3.0, 5.0],
    # NegativeSampling
    "adv_temperature": [1],
    "regul_rate": [0.0],
    # Trainer
    "train_times": [3_000],
    "alpha": [2e-5, 5e-3],
    "opt_method": ["adam"]
}

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

today = datetime.today()

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')


# misc ======================================================================

def generate_grid(hyperparameter: Mapping[str, Iterable[Any]]) -> Iterable[Mapping[str, Iterable[Any]]]:
    """Generates an iterator over the cartesian product of hyperparameters values.

    :param hyperparameter: a mapping from hyperparameter name to possible values
    :return: an iterator over the cartesian product of hyperparameters values
    """
    # sort is used to make sure order is always the same
    sorted_keys = sorted(hyperparameter.keys())
    hpp = product(*[hyperparameter[k] for k in sorted_keys])
    for hp in hpp:
        grid_item = dict()
        for idx, key in enumerate(sorted_keys):
            grid_item[key] = hp[idx]
        yield grid_item


# The following code was taken from Eli Bendersky's website
#   https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/#id8
@cl.contextmanager
def redirect(source, destination):
    def _get_sys(source):
        if source == "stdout":
            return sys.stdout
        elif source == "stderr":
            return sys.stderr

    def _get_c_(source):
        if source == "stdout":
            return c_stdout
        elif source == "stderr":
            return c_stderr

    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_source_fd = _get_sys(source).fileno()

    def _redirect_to_fd(source, to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(_get_c_(source))
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        _get_sys(source).close()
        # Make original_source_fd point to the same file as to_fd
        os.dup2(to_fd, original_source_fd)
        # Create a new sys.stdout that points to the redirected fd
        if source == "stdout":
            sys.stdout = io.TextIOWrapper(os.fdopen(original_source_fd, 'wb'))
        elif source == "stderr":
            sys.stderr = io.TextIOWrapper(os.fdopen(original_source_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_fd
    saved_fd = os.dup(original_source_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_to_fd(source, tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_to_fd(source, saved_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        destination.write(tfile.read().decode())
    finally:
        tfile.close()
        os.close(saved_fd)


def save_as_txt(path, embs):
    with open(path, "w") as fp:
        for emb in list(embs):
            line = "\t".join(str(w) for w in list(emb))
            fp.write(f"{line}\t\n")


# entry point ===============================================================

grid = list(generate_grid(hyperparameters))

# dataloader for test
test_dataloader = TestDataLoader(dataset_dir, "link")

for run_num, hp in enumerate(grid):
    run_tic = datetime.now()
    # preparation
    run_id = f"TransEadv_{today.year}{today.month}{today.day}_{run_num}"
    run_dir = os.path.join(dataset_dir, "TransEadv", run_id)
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
        norm_flag=hp["norm_flag"],
        margin=hp["transe_margin"])
    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=SigmoidLoss(adv_temperature=hp["adv_temperature"]),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=hp["regul_rate"]
    )
    # train the model
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=hp["train_times"],
        alpha=hp["alpha"],
        opt_method=hp["opt_method"],
        use_gpu=True,
    )
    with open(os.path.join(run_dir, "train.out"), "w") as out, \
            open(os.path.join(run_dir, "train.err"), "w") as err, \
            redirect("stdout", out), \
            redirect("stderr", err):
        train_tic = datetime.now()
        trainer.run()
        train_tac = datetime.now()
    # save final state
    transe.save_checkpoint(os.path.join(run_dir, "transeadv_final.ckpt"))
    # test the model
    with open(os.path.join(run_dir, "test.out"), "w") as out, \
            open(os.path.join(run_dir, "test.err"), "w") as err, \
            redirect("stdout", out), \
            redirect("stderr", err):
        test_tic = datetime.now()
        tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
        tester.run_link_prediction(type_constrain=False)
        test_tac = datetime.now()
        run_tac = datetime.now()
    # save a few important things
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as fp:
        json.dump(hp, fp, indent=4)
    elapsed_time = {
        "train_time": str(train_tac - train_tic),
        "test_time": str(test_tac - test_tic),
        "total_time": str(run_tac - run_tic)
    }
    with open(os.path.join(run_dir, "time.json"), "w") as fp:
        json.dump(elapsed_time, fp, indent=4)
    # save embeddings
    ent_embeddings = transe.ent_embeddings.weight.detach().cpu().numpy()
    rel_embeddings = transe.rel_embeddings.weight.detach().cpu().numpy()
    np.save(os.path.join(run_dir, f"entity2vec{hp['dim']}.npy"), ent_embeddings)
    np.save(os.path.join(run_dir, f"relation2vec{hp['dim']}.npy"), rel_embeddings)
    save_as_txt(os.path.join(run_dir, f"relation2vec{hp['dim']}.init"), rel_embeddings)
    save_as_txt(os.path.join(run_dir, f"entity2vec{hp['dim']}.init"), ent_embeddings)
