import sys
import os
import dill
import json
import argparse
import torch
from torch.utils import data
import numpy as np
import pandas as pd

sys.path.append("../../ScePT")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.ScePT import ScePT
import evaluation
from utils.comm import all_gather
from collections import defaultdict
from model.dataset import EnvironmentDataset, collate

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pdb

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def load_model(model_dir, env, device, ts=100):
    model_registrar = ModelRegistrar(model_dir, device)
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, "config.json"), "r") as config_json:
        hyperparams = json.load(config_json)

    ScePT = ScePT(model_registrar, hyperparams, None, device)

    ScePT.set_environment(env)
    ScePT.set_annealing_params()
    return ScePT, hyperparams


def main():
    with open("../processed/eth_test.pkl", "rb") as f:
        env = dill.load(f, encoding="latin1")

    ScePT, hyperparams = load_model(
        "models/eth_model-05_Aug_2021_12_20_01", env, device="cpu", ts=99
    )
    dists, _ = ScePT.predict(
        scene=env.scenes[0], timesteps=np.array([10]), ph=12, output_dists=True
    )
    pdb.set_trace()


if __name__ == "__main__":
    main()
