import torch
from torch import nn, optim
from torch.utils import data
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
from tqdm import tqdm
import visualization
import matplotlib.pyplot as plt
import matplotlib
from argument_parser import args
from model.ScePT import ScePT
from model.model_registrar import ModelRegistrar
from model.dataset import *
from torch.utils.tensorboard import SummaryWriter

# torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.comm import all_gather
from collections import defaultdict

matplotlib.use("Agg")
from functools import partial
from pathos.multiprocessing import ProcessPool as Pool
from torch.cuda.amp import autocast, GradScaler
import model.dynamics as dynamic_module

thismodule = sys.modules[__name__]


def train(rank, args):
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print("Config json not found!")
    with open(args.conf, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams["dynamic_edges"] = args.dynamic_edges

    hyperparams["batch_size"] = args.batch_size
    hyperparams["k_eval"] = args.k_eval
    hyperparams["offline_scene_graph"] = args.offline_scene_graph
    hyperparams["incl_robot_node"] = args.incl_robot_node

    hyperparams["edge_encoding"] = not args.no_edge_encoding
    hyperparams["use_map_encoding"] = args.map_encoding
    num_workers = args.preprocess_workers
    # hyperparams['augment'] = args.augment
    # hyperparams['override_attention_radius'] = args.override_attention_radius

    # Distributed LR Scaling
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    hyperparams["learning_rate"] *= dist.get_world_size()
    if args.train_data_dict == "nuScenes_train.pkl":
        nusc_path = "../experiments/nuScenes/v1.0-trainval_meta"
    elif args.train_data_dict == "nuScenes_mini_train.pkl":
        nusc_path = "../experiments/nuScenes/v1.0-mini"
    else:
        nusc_path = None

    processed_train_file = "processed_" + args.train_data_dict
    processed_eval_file = "processed_" + args.eval_data_dict

    if rank == 0:
        print("-----------------------")
        print("| TRAINING PARAMETERS |")
        print("-----------------------")
        print("| Batch Size: %d" % args.batch_size)
        print("| Eval Batch Size: %d" % args.eval_batch_size)
        print("| Device: %s" % args.device)
        print("| Learning Rate: %s" % hyperparams["learning_rate"])
        print("| Learning Rate Step Every: %s" % args.lr_step)
        print("| Offline Scene Graph Calculation: %s" % args.offline_scene_graph)
        print("| MHL: %s" % hyperparams["minimum_history_length"])
        print("| PH: %s" % hyperparams["prediction_horizon"])
        print("-----------------------")

    log_writer = None
    model_dir = None
    if not args.debug:
        # Create the log and model directory if they're not present.
        model_dir = os.path.join(
            args.log_dir,
            args.log_tag + time.strftime("%d_%b_%Y_%H_%M_%S", time.localtime()),
        )

        if rank == 0:
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

            # Save config to model directory
            with open(os.path.join(model_dir, "config.json"), "w") as conf_json:
                json.dump(hyperparams, conf_json)

            log_writer = SummaryWriter(log_dir=model_dir)

    # Load training and evaluation environments and scenes
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)

    with open(train_data_path, "rb") as f:
        # train_env = dill.load(f, encoding='latin1')
        train_env = dill.load(f)
    train_scenes = train_env.scenes

    if "default_con" in hyperparams["dynamic"]["PEDESTRIAN"]:
        dynamics = dict()
        default_con = dict()
        input_scale = dict()
        for nt in hyperparams["dynamic"]:
            model = getattr(dynamic_module, hyperparams["dynamic"][nt]["name"])
            input_scale[nt] = torch.tensor(hyperparams["dynamic"][nt]["limits"])
            dynamics[nt] = model(train_env.dt, input_scale[nt], "cpu", None, None, nt)
            model = getattr(thismodule, hyperparams["dynamic"][nt]["default_con"])
            default_con[nt] = model
    else:
        default_con = None
        dynamics = None
    if args.use_processed_data:
        with open(processed_train_file, "rb") as f:
            train_cliques = dill.load(f)
        f.close()
    else:

        # train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

        # Offline Calculate Training Scene Graphs
        # cliques = obtain_clique_from_scene( scene = train_env.scenes[0],
        #                                     adj_radius=hyperparams['adj_radius'],
        #                                     ht = hyperparams['maximum_history_length'],
        #                                     ft=hyperparams['prediction_horizon'],
        #                                     hyperparams=hyperparams,
        #                                     dynamics=dynamics,
        #                                     con=default_con,
        #                                     max_clique_size=hyperparams['max_clique_size'],
        #                                     nusc_path=nusc_path)
        with Pool(num_workers) as pool:
            scene_cliques = list(
                tqdm(
                    pool.imap(
                        partial(
                            obtain_clique_from_scene,
                            adj_radius=hyperparams["adj_radius"],
                            ht=hyperparams["maximum_history_length"],
                            ft=hyperparams["prediction_horizon"],
                            hyperparams=hyperparams,
                            dynamics=dynamics,
                            con=default_con,
                            max_clique_size=hyperparams["max_clique_size"],
                            nusc_path=nusc_path,
                        ),
                        train_env.scenes,
                    ),
                    desc=f"Processing Scenes ({num_workers} CPUs)",
                    total=len(train_env.scenes),
                    disable=(rank > 0),
                )
            )
        train_cliques = scene_cliques[0]
        for i in range(1, len(scene_cliques)):
            train_cliques = train_cliques + scene_cliques[i]

        with open(processed_train_file, "wb") as f:
            dill.dump(train_cliques, f)
        f.close()
    train_data = clique_dataset(train_cliques)
    train_data_loader = DataLoader(
        train_data,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=clique_collate,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        num_workers=6,
    )
    train_sampler = data.distributed.DistributedSampler(
        train_data, num_replicas=dist.get_world_size(), rank=rank
    )

    print(f"Rank {rank}: Loaded training data from {train_data_path}")

    eval_scenes = []
    eval_scenes_sample_probs = None
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, "rb") as f:
            eval_env = dill.load(f, encoding="latin1")
        eval_scenes = eval_env.scenes

        if args.use_processed_data:
            with open(processed_eval_file, "rb") as f:
                eval_cliques = dill.load(f)
        else:
            # eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None
            with Pool(num_workers) as pool:
                scene_cliques = list(
                    tqdm(
                        pool.imap(
                            partial(
                                obtain_clique_from_scene,
                                adj_radius=hyperparams["adj_radius"],
                                ht=hyperparams["maximum_history_length"],
                                ft=hyperparams["prediction_horizon"],
                                hyperparams=hyperparams,
                                dynamics=dynamics,
                                con=default_con,
                                max_clique_size=hyperparams["max_clique_size"],
                                nusc_path=nusc_path,
                            ),
                            eval_env.scenes,
                        ),
                        desc=f"Processing Scenes ({num_workers} CPUs)",
                        total=len(eval_env.scenes),
                        disable=(rank > 0),
                    )
                )
            eval_cliques = scene_cliques[0]
            for i in range(1, len(scene_cliques)):
                eval_cliques = eval_cliques + scene_cliques[i]

            with open(processed_eval_file, "wb") as f:
                dill.dump(eval_cliques, f)
            f.close()
    num_nodes = 0
    for scene in train_env.scenes:
        num_nodes += len(scene.nodes)
    for scene in eval_env.scenes:
        num_nodes += len(scene.nodes)
    print(num_nodes)

    eval_data = clique_dataset(eval_cliques)
    eval_data_loader = DataLoader(
        eval_data,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=clique_collate,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        num_workers=6,
    )
    eval_sampler = data.distributed.DistributedSampler(
        eval_data, num_replicas=dist.get_world_size(), rank=rank
    )

    model_registrar = ModelRegistrar(model_dir, args.device)
    # load_model_dir = os.path.join(args.log_dir,
    #                              args.log_tag + '27_Oct_2021_12_47_38')
    # model_registrar.model_dir = load_model_dir
    # model_registrar.load_models(iter_num=19)
    # model_registrar.model_dir = model_dir

    ScePT_model = ScePT(model_registrar, hyperparams, log_writer, args.device)
    ScePT_model.set_environment(train_env)
    ScePT_model.set_annealing_params()

    if torch.cuda.is_available():
        ScePT_model = DDP(
            ScePT_model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
        ScePT_module = ScePT_model.module
    else:
        ScePT_module = ScePT_model

    print(f"Rank {rank}: Created Training Model.")

    optimizer = optim.Adam(
        [
            {
                "params": model_registrar.get_all_but_name_match(
                    "map_encoder"
                ).parameters()
            },
            {
                "params": model_registrar.get_name_match("map_encoder").parameters(),
                "lr": 0.0008,
            },
        ],
        lr=hyperparams["learning_rate"],
    )
    if hyperparams["learning_rate_style"] == "const":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif hyperparams["learning_rate_style"] == "exp":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=hyperparams["learning_decay_rate"]
        )
    if args.lr_step is not None:
        step_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=0.1
        )

    #################################
    #           TRAINING            #
    #################################
    curr_iter = 0

    # scene = train_env.scenes[0]
    # timestep = scene.sample_timesteps(1, min_future_timesteps=hyperparams['prediction_horizon'],min_history_timesteps=hyperparams['maximum_history_length'])

    # results = ScePT_module.snapshot_predict(scene,timestep,hyperparams['prediction_horizon'],3,nusc_path)
    # clique_type,clique_first_timestep,clique_last_timestep,clique_state_history,clique_future_state,clique_state_pred,input_pred_res,clique_ref_traj,clique_pi_list,clique_node_size,clique_is_robot = results[0]
    # # Plot predicted timestep for random scene
    # if not clique_type is None:
    #     plt.clf()
    #     if scene.map is None:
    #         limits=None
    #     else:
    #         limits = [100,100]
    #     fig,ax = visualization.plot_trajectories_clique(clique_type,
    #                                                    clique_last_timestep,
    #                                                    clique_state_history,
    #                                                    clique_future_state,
    #                                                    clique_state_pred,
    #                                                    clique_ref_traj,
    #                                                    scene.map['VISUALIZATION'] if scene.map is not None else None,
    #                                                    clique_node_size,
    #                                                    clique_is_robot,
    #                                                    limits = limits)
    scaler = GradScaler()
    max_hl = hyperparams["maximum_history_length"]
    for epoch in range(1, args.train_epochs + 1):
        train_data.augment = args.augment
        train_sampler.set_epoch(epoch)

        pbar = tqdm(
            train_data_loader,
            ncols=80,
            unit_scale=dist.get_world_size(),
            position=0,
            leave=True,
            disable=(rank > 0),
        )
        for batch in pbar:
            ScePT_module.set_curr_iter(curr_iter)
            ScePT_module.step_annealers()
            optimizer.zero_grad()
            # with autocast():
            train_loss = ScePT_model(batch)
            pbar.set_description(
                f"Epoch {epoch}, gamma:{ScePT_module.model.gamma:.2f}, L: {train_loss.detach().item():.2f}"
            )
            if hyperparams["use_scaler"]:

                scaler.scale(train_loss).backward()

                scaler.unscale_(optimizer)
                if hyperparams["grad_clip"] is not None:
                    nn.utils.clip_grad_value_(
                        model_registrar.parameters(), hyperparams["grad_clip"]
                    )
                scaler.step(optimizer)

                scaler.update()
            else:

                train_loss.backward()

                # Clipping gradients.
                if hyperparams["grad_clip"] is not None:
                    nn.utils.clip_grad_value_(
                        model_registrar.parameters(), hyperparams["grad_clip"]
                    )
                optimizer.step()

            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()
            if rank == 0 and not args.debug:
                log_writer.add_scalar(
                    f"train/learning_rate", lr_scheduler.get_last_lr()[0], curr_iter
                )
                log_writer.add_scalar(
                    f"train/loss", train_loss.detach().item(), curr_iter
                )

            curr_iter += 1

        if args.lr_step is not None:
            step_scheduler.step()

        train_data.augment = False

        #################################
        #        VISUALIZATION          #
        #################################
        if rank == 0 and (
            args.vis_every is not None
            and not args.debug
            and epoch % args.vis_every == 0
            and epoch > 0
        ):
            max_hl = hyperparams["maximum_history_length"]

            with torch.no_grad():
                # Predict random timestep to plot for train data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(train_scenes, p=train_scenes_sample_probs)
                else:
                    scene = np.random.choice(train_scenes)

                try:
                    timestep = scene.sample_timesteps(
                        1,
                        min_future_timesteps=hyperparams["prediction_horizon"],
                        min_history_timesteps=hyperparams["maximum_history_length"],
                    )

                    results = ScePT_module.snapshot_predict(
                        scene, timestep, hyperparams["prediction_horizon"], 3, nusc_path
                    )
                    (
                        clique_type,
                        clique_first_timestep,
                        clique_last_timestep,
                        clique_state_history,
                        clique_future_state,
                        clique_state_pred,
                        input_pred_res,
                        clique_ref_traj,
                        clique_pi_list,
                        clique_node_size,
                        clique_is_robot,
                    ) = results[0]
                    # Plot predicted timestep for random scene
                    if not clique_type is None:
                        plt.clf()
                        if scene.map is None:
                            limits = None
                        else:
                            limits = [100, 100]
                        fig, ax = visualization.plot_trajectories_clique(
                            clique_type,
                            clique_last_timestep,
                            clique_state_history,
                            clique_future_state,
                            clique_state_pred,
                            clique_ref_traj,
                            scene.map["VISUALIZATION"]
                            if scene.map is not None
                            else None,
                            clique_node_size,
                            clique_is_robot,
                            limits=limits,
                        )

                        ax.set_title(f"{scene.name}-t: {timestep}")

                        log_writer.add_figure("train/prediction", fig, epoch)
                        del fig, ax
                except:
                    print("visualization failed")

                # Predict random timestep to plot for eval data set
                # if args.scene_freq_mult_viz:
                #     scene = np.random.choice(eval_scenes, p=eval_scenes_sample_probs)
                # else:
                #     scene = np.random.choice(eval_scenes)
                # timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                # predictions = ScePT_module.predict(scene,
                #                                         timestep,
                #                                         ph,
                #                                         num_samples=20,
                #                                         min_future_timesteps=ph,
                #                                         z_mode=False,
                #                                         full_dist=False)

                # # Plot predicted timestep for random scene
                # fig, ax = plt.subplots(figsize=(10, 10))
                # visualization.visualize_prediction(ax,
                #                                    predictions,
                #                                    scene.dt,
                #                                    max_hl=max_hl,
                #                                    ph=ph,
                #                                    map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                # ax.set_title(f"{scene.name}-t: {timestep}")
                # log_writer.add_figure('eval/prediction', fig, epoch)

                # # Predict random timestep to plot for eval data set
                # predictions = ScePT_module.predict(scene,
                #                                         timestep,
                #                                         ph,
                #                                         min_future_timesteps=ph,
                #                                         z_mode=True,
                #                                         gmm_mode=True,
                #                                         all_z_sep=True,
                #                                         full_dist=False)

                # # Plot predicted timestep for random scene
                # fig, ax = plt.subplots(figsize=(10, 10))
                # visualization.visualize_prediction(ax,
                #                                    predictions,
                #                                    scene.dt,
                #                                    max_hl=max_hl,
                #                                    ph=ph,
                #                                    map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                # ax.set_title(f"{scene.name}-t: {timestep}")
                # log_writer.add_figure('eval/prediction_all_z', fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if (
            args.eval_every is not None
            and not args.debug
            and epoch % args.eval_every == 0
            and epoch > 0
        ):
            max_hl = hyperparams["maximum_history_length"]
            ph = hyperparams["prediction_horizon"]
            with torch.no_grad():
                pbar = tqdm(
                    eval_data_loader,
                    ncols=80,
                    unit_scale=dist.get_world_size(),
                    position=0,
                    leave=True,
                    disable=(rank > 0),
                )
                for batch in pbar:
                    (
                        eval_loss,
                        ADE,
                        FDE,
                        ADE_count,
                        FDE_count,
                        _,
                        _,
                    ) = ScePT_module.eval_loss(batch)
                    pbar.set_description(f"Epoch {epoch}, L: {eval_loss:.2f}")

            if rank == 0 and not args.debug:
                log_writer.add_scalar(f"eval/loss", eval_loss, curr_iter)

        if rank == 0 and (
            args.save_every is not None
            and args.debug is False
            and epoch % args.save_every == 0
        ):
            model_registrar.save_models(epoch)

        # Waiting for process 0 to be done its evaluation and visualization.
        if torch.cuda.is_available():
            dist.barrier()


def spmd_main(local_rank):
    if torch.cuda.is_available():
        backend = "nccl"
        torch.backends.cudnn.benchmark = True
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}, "
        + f"port = {os.environ['MASTER_PORT']} \n",
        end="",
    )
    train(local_rank, args)


if __name__ == "__main__":
    spmd_main(args.local_rank)
