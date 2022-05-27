import contextlib as cl
import ctypes
import io
import tempfile
import json
import logging
import os
import sys
from datetime import datetime
from itertools import product
from typing import Mapping, Iterable, Any

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
    "batch_size": [100_000],
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
    "alpha": [1e-4, 5e-5],
    "opt_method": ["sgd"]
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

    if source == "stdout":
        _source = sys.stdout
        c_source = c_stdout
    elif source == "stderr":
        _source = sys.stderr
        c_source = c_stderr
    else:
        raise ValueError(f"Can't handle source {source}, must be 'stdout' or 'stderr'")

    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_source_fd = sys.stdout.fileno()

    def _redirect_to_fd(_source, to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_source)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        _source.close()
        # Make original_source_fd point to the same file as to_fd
        os.dup2(to_fd, original_source_fd)
        # Create a new sys.stdout that points to the redirected fd
        _source = io.TextIOWrapper(os.fdopen(original_source_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_source_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_to_fd(_source, tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_to_fd(_source, saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        destination.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)



# entry point ===============================================================

grid = list(generate_grid(hyperparameters))

# dataloader for test
test_dataloader = TestDataLoader(dataset_dir, "link")

for run_num, hp in enumerate(grid):
    run_tic = datetime.now()
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
        train_tic = datetime.now()
        trainer.run()
        train_tac = datetime.now()
    # save final state
    transe.save_checkpoint(os.path.join(run_dir, "transe_final.ckpt"))
    # test the model
    with open(os.path.join(run_dir, "test.out"), "w") as out, \
            open(os.path.join(run_dir, "test.err"), "w") as err, \
            cl.redirect_stdout(out), \
            cl.redirect_stderr(err):
        test_tic = datetime.now()
        tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
        tester.run_link_prediction(type_constrain=False)
        test_tac = datetime.now()
        run_tac = datetime.now()
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as fp:
        json.dump(hp, fp, indent=4)
    elapsed_time = {
        "train_time": str(train_tac - train_tic),
        "test_time": str(test_tac - test_tic),
        "total_time": str(run_tac - run_tic)
    }
    with open(os.path.join(run_dir, "time.json"), "w") as fp:
        json.dump(elapsed_time, fp, indent=4)
