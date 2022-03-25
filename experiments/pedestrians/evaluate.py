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


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument(
    "--indexing_workers",
    help="number of processes to spawn for dataset indexing",
    type=int,
    default=1,
)
parser.add_argument(
    "--preprocess_workers",
    help="number of processes to spawn for preprocessing",
    type=int,
    default=0,
)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument(
    "--local_rank",
    help="local process number for distributed training",
    type=int,
    default=0,
)
parser.add_argument(
    "--eval_batch_size", help="evaluation batch size", type=int, default=256
)
args = parser.parse_args()


def load_model(model_dir, env, device, ts=100):
    model_registrar = ModelRegistrar(model_dir, device)
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, "config.json"), "r") as config_json:
        hyperparams = json.load(config_json)

    ScePT = ScePT(model_registrar, hyperparams, None, device)

    ScePT.set_environment(env)
    ScePT.set_annealing_params()
    return ScePT, hyperparams


def evaluate(rank, args):
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    with open(args.data, "rb") as f:
        env = dill.load(f, encoding="latin1")

    ScePT, hyperparams = load_model(
        args.model, env, device=args.device, ts=args.checkpoint
    )

    if torch.cuda.is_available():
        ScePT = DDP(
            ScePT, device_ids=[rank], output_device=rank, find_unused_parameters=True
        )
        ScePT_module = ScePT.module
    else:
        ScePT_module = ScePT

    if "override_attention_radius" in hyperparams:
        for attention_radius_override in hyperparams["override_attention_radius"]:
            node_type1, node_type2, attention_radius = attention_radius_override.split(
                " "
            )
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    # print("-- Preparing Node Graph")
    # for scene in tqdm(scenes):
    #     scene.calculate_scene_graph(env.attention_radius,
    #                                 hyperparams['edge_addition_filter'],
    #                                 hyperparams['edge_removal_filter'])

    eval_dataset = EnvironmentDataset(
        env,
        hyperparams["state"],
        hyperparams["pred_state"],
        scene_freq_mult=hyperparams["scene_freq_mult_eval"],
        node_freq_mult=hyperparams["node_freq_mult_eval"],
        hyperparams=hyperparams,
        min_history_timesteps=hyperparams["minimum_history_length"],
        min_future_timesteps=hyperparams["prediction_horizon"],
        return_robot=True,
        num_workers=args.indexing_workers,
        rank=rank,
    )
    eval_data_loader = dict()
    for node_type_data_set in eval_dataset:
        if len(node_type_data_set) == 0:
            continue

        eval_sampler = data.distributed.DistributedSampler(
            node_type_data_set, num_replicas=dist.get_world_size(), rank=rank
        )

        node_type_dataloader = data.DataLoader(
            node_type_data_set,
            collate_fn=collate,
            pin_memory=False if args.device == "cpu" else True,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.preprocess_workers,
            sampler=eval_sampler,
        )
        eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Rank {rank}: Loaded testing data from {args.data}")

    ph = hyperparams["prediction_horizon"]
    max_hl = hyperparams["maximum_history_length"]

    with torch.no_grad():
        for node_type, data_loader in eval_data_loader.items():
            error_dict = defaultdict(list)

            for batch in tqdm(
                data_loader,
                ncols=80,
                unit_scale=dist.get_world_size(),
                disable=(rank > 0),
                desc=str(node_type),
            ):
                errors = ScePT_module.predict_and_evaluate_batch(
                    batch, node_type, max_hl
                )

                for metric, values in errors.items():
                    error_dict[metric].append(values)

            for metric in error_dict:
                error_dict[metric] = torch.cat(error_dict[metric])
                if torch.cuda.is_available() and dist.get_world_size() > 1:
                    error_dict[metric] = np.concatenate(all_gather(error_dict[metric]))
                else:
                    error_dict[metric] = error_dict[metric].numpy()

                if rank == 0:
                    print(node_type, metric, np.mean(error_dict[metric]))

                    pd.DataFrame(
                        {
                            "value": error_dict[metric],
                            "metric": metric,
                            "type": "ml" if metric in ["ade", "fde"] else "full",
                            "node_type": str(node_type),
                        }
                    ).to_csv(
                        os.path.join(
                            args.output_path, args.output_tag + f"_{metric}.csv"
                        ),
                        index=False,
                    )


def spmd_main(local_rank):
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}, "
        + f"port = {os.environ['MASTER_PORT']} \n",
        end="",
    )

    evaluate(local_rank, args)


if __name__ == "__main__":
    spmd_main(args.local_rank)
