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
from argument_parser import args
from model.ScePT import ScePT
from model.components import *
from model.model_registrar import ModelRegistrar
from model.dataset import *
from torch.utils.tensorboard import SummaryWriter
from model.mgcvae_clique import MultimodalGenerativeCVAE_clique
from Planning import FTOCP
import pdb
from model.components import *
from model.model_utils import *
from model.dynamics import *
from torch.utils.data import Dataset, DataLoader
import time

# torch.autograd.set_detect_anomaly(True)

import torch.distributed as dist


from functools import partial
from pathos.multiprocessing import ProcessPool as Pool
import model.dynamics as dynamic_module

thismodule = sys.modules[__name__]


def make_video(rank):
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(data_path, "rb") as f:
        env = dill.load(f)
    model_dir = os.path.join(args.log_dir, args.log_tag + args.trained_model_dir)
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)

    hyperparams["use_map_encoding"] = args.map_encoding
    hyperparams["offline_scene_graph"] = args.offline_scene_graph
    hyperparams["incl_robot_node"] = args.incl_robot_node

    hyperparams["edge_encoding"] = not args.no_edge_encoding

    model_registrar = ModelRegistrar(model_dir, args.device)
    ScePT_model = ScePT(model_registrar, hyperparams, None, args.device)
    model_registrar.load_models(iter_num=args.iter_num)
    ScePT_model.set_environment(env)
    if (
        args.eval_data_dict == "nuScenes_train.pkl"
        or args.eval_data_dict == "nuScenes_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-trainval_meta"
    elif (
        args.eval_data_dict == "nuScenes_mini_train.pkl"
        or args.eval_data_dict == "nuScenes_mini_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-mini"
    else:
        nusc_path = None
    scene_idx = range(0, 30)
    if "default_con" in hyperparams["dynamic"]["VEHICLE"]:
        default_con = dict()
        input_scale = dict()
        dynamics = dict()
        for nt in hyperparams["dynamic"]:
            model = getattr(dynamic_module, hyperparams["dynamic"][nt]["name"])
            input_scale[nt] = torch.tensor(hyperparams["dynamic"][nt]["limits"])
            dynamics[nt] = model(env.dt, input_scale[nt], "cpu", None, None, nt)
            model = getattr(thismodule, hyperparams["dynamic"][nt]["default_con"])
            default_con[nt] = model
    else:
        dynamics = None
        default_con = None
    for k in scene_idx:
        scene = env.scenes[k]

        ft = hyperparams["prediction_horizon"]
        max_clique_size = hyperparams["max_clique_size"]
        results, _, _ = ScePT_model.replay_prediction(
            scene,
            None,
            ft,
            max_clique_size,
            dynamics,
            default_con,
            num_samples=3,
            nusc_path=nusc_path,
        )
        num_traj_show = 5
        if args.video_name is None:
            visualization.sim_clique_prediction(
                results,
                scene.map["VISUALIZATION"],
                env.dt,
                num_traj_show,
                limits=[100, 100],
            )
        else:
            visualization.sim_clique_prediction(
                results,
                scene.map["VISUALIZATION"],
                env.dt,
                num_traj_show,
                args.video_name + str(k) + ".mp4",
                limits=[100, 100],
            )


def sim_planning(rank):
    # nusc_25_Oct_2021_16_47_33
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(data_path, "rb") as f:
        env = dill.load(f)
    model_dir = os.path.join(args.log_dir, args.log_tag + args.trained_model_dir)
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)

    hyperparams["use_map_encoding"] = args.map_encoding
    hyperparams["offline_scene_graph"] = args.offline_scene_graph
    hyperparams["incl_robot_node"] = args.incl_robot_node

    hyperparams["edge_encoding"] = not args.no_edge_encoding

    model_registrar = ModelRegistrar(model_dir, args.device)
    ScePT_model = ScePT(model_registrar, hyperparams, None, args.device)
    model_registrar.load_models(iter_num=args.iter_num)
    ScePT_model.set_environment(env)
    if (
        args.eval_data_dict == "nuScenes_train.pkl"
        or args.eval_data_dict == "nuScenes_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-trainval_meta"
    elif (
        args.eval_data_dict == "nuScenes_mini_train.pkl"
        or args.eval_data_dict == "nuScenes_mini_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-mini"
    else:
        nusc_path = None

    if "default_con" in hyperparams["dynamic"]["VEHICLE"]:
        default_con = dict()
        input_scale = dict()
        dynamics = dict()
        for nt in hyperparams["dynamic"]:
            model = getattr(dynamic_module, hyperparams["dynamic"][nt]["name"])
            input_scale[nt] = torch.tensor(hyperparams["dynamic"][nt]["limits"])
            dynamics[nt] = model(env.dt, input_scale[nt], "cpu", None, None, nt)
            model = getattr(thismodule, hyperparams["dynamic"][nt]["default_con"])
            default_con[nt] = model
    else:
        dynamics = None
        default_con = None

    ft = hyperparams["prediction_horizon"]
    max_clique_size = hyperparams["max_clique_size"]
    M = 3
    # nuScenes_train
    # scene_indices = [3, 4, 9, 10, 16, 17]
    # nuScenes_val
    # scene_indices = [20,24]

    scene_indices = [24]
    for n in scene_indices:
        scene = env.scenes[n]
        print(n)
        results, extra_node_info, _ = ScePT_model.replay_prediction(
            scene,
            None,
            ft,
            max_clique_size,
            dynamics,
            default_con,
            num_samples=M,
            center_node=scene.robot,
            nusc_path=nusc_path,
        )
        num_traj_show = M
        planner = FTOCP(ft, M, scene.dt, scene.robot.width, scene.robot.length)
        plan_traj = list()
        robot = scene.robot
        traj_plan = [None] * len(results)
        start = time.time()
        for k in range(len(results)):
            plan_traj_t = dict()
            (
                clique_nodes,
                clique_state_history,
                clique_state_pred,
                clique_node_size,
                clique_pi_list,
                clique_lane,
            ) = results[k]

            for i in range(len(clique_nodes)):

                if robot in clique_nodes[i]:
                    nodes = [node for node in clique_nodes[i] if node != robot]
                    idx = clique_nodes[i].index(robot)
                    x0 = clique_state_history[i][idx][-1]
                    ego_lane = clique_lane[i][idx]

                    ypreds = (
                        clique_state_pred[i][0:idx] + clique_state_pred[i][idx + 1 :]
                    )

                    xref = obtain_ref(
                        ego_lane[:, [0, 1, 3]], x0[0:2], x0[2], ft, scene.dt
                    )
                    planner.buildandsolve(nodes, x0, xref, ypreds, clique_pi_list[i])
                    xplan = planner.xSol[1:].reshape((M, ft, 4))
                    xplan = np.concatenate(
                        (np.tile(x0, M).reshape(M, 1, -1), xplan), axis=1
                    )
                    traj_plan[k] = xplan
        end = time.time()
        print("planning takes ", end - start)

        if args.video_name is None:
            visualization.sim_clique_prediction(
                results,
                scene.map["VISUALIZATION"],
                env.dt,
                num_traj_show,
                limits=[80, 80],
                robot_plan=(scene.robot, traj_plan),
                extra_node_info=extra_node_info,
            )
        else:
            visualization.sim_clique_prediction(
                results,
                scene.map["VISUALIZATION"],
                env.dt,
                num_traj_show,
                args.video_name + "_" + str(n) + ".mp4",
                limits=[80, 80],
                robot_plan=(scene.robot, traj_plan),
                extra_node_info=extra_node_info,
            )


def simulate_prediction(rank):
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(data_path, "rb") as f:
        env = dill.load(f)
    with open(args.conf, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)

    hyperparams["use_map_encoding"] = args.map_encoding
    hyperparams["offline_scene_graph"] = args.offline_scene_graph
    hyperparams["incl_robot_node"] = args.incl_robot_node
    hyperparams["batch_size"] = args.batch_size
    hyperparams["edge_encoding"] = not args.no_edge_encoding

    model_dir = os.path.join(args.log_dir, args.log_tag + args.trained_model_dir)
    model_registrar = ModelRegistrar(model_dir, args.device)
    ScePT_model = ScePT(model_registrar, hyperparams, None, args.device)
    model_registrar.load_models(iter_num=args.iter_num)
    ScePT_model.set_environment(env)
    scene = env.scenes[1]
    results = ScePT_model.simulate_prediction(
        scene,
        15,
        hyperparams["maximum_history_length"],
        20,
        hyperparams["max_clique_size"],
    )
    num_traj_show = 1
    if scene.map is None:
        limits = [-10, 10, -10, 10]
    else:
        limits = [100, 100]
    if args.video_name is None:
        visualization.sim_clique_prediction(
            results, scene.map, env.dt, num_traj_show, limits=limits
        )
    else:
        visualization.sim_clique_prediction(
            results,
            scene.map,
            env.dt,
            num_traj_show,
            args.video_name + ".mp4",
            limits=limits,
        )


def eval_statistics(rank):
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    if args.preprocess_workers == 0:
        num_workers = 10
    else:
        num_workers = args.preprocess_workers
    data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(data_path, "rb") as f:
        env = dill.load(f)
    if (
        args.eval_data_dict == "nuScenes_train.pkl"
        or args.eval_data_dict == "nuScenes_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-trainval_meta"
    elif (
        args.eval_data_dict == "nuScenes_mini_train.pkl"
        or args.eval_data_dict == "nuScenes_mini_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-mini"
    else:
        nusc_path = None
    processed_eval_file = "processed_" + args.eval_data_dict

    model_dir = os.path.join(args.log_dir, args.log_tag + args.trained_model_dir)
    model_registrar = ModelRegistrar(model_dir, args.device)
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)
    hyperparams["use_map_encoding"] = args.map_encoding
    hyperparams["offline_scene_graph"] = args.offline_scene_graph
    hyperparams["incl_robot_node"] = args.incl_robot_node
    hyperparams["batch_size"] = args.batch_size
    hyperparams["edge_encoding"] = not args.no_edge_encoding
    ScePT_model = ScePT(model_registrar, hyperparams, None, args.device)

    model_registrar.load_models(iter_num=args.iter_num)
    if "default_con" in hyperparams["dynamic"]["PEDESTRIAN"]:
        dynamics = dict()
        default_con = dict()
        input_scale = dict()
        for nt in hyperparams["dynamic"]:
            model = getattr(dynamic_module, hyperparams["dynamic"][nt]["name"])
            input_scale[nt] = torch.tensor(hyperparams["dynamic"][nt]["limits"])
            dynamics[nt] = model(env.dt, input_scale[nt], "cpu", None, None, nt)
            model = getattr(thismodule, hyperparams["dynamic"][nt]["default_con"])
            default_con[nt] = model
    else:
        default_con = None
        dynamics = None
    ScePT_model.set_environment(env)
    if args.use_processed_data:
        with open(processed_eval_file, "rb") as f:
            eval_cliques = dill.load(f)
        f.close()
    else:
        scene_cliques = list()
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
                        env.scenes,
                    ),
                    desc=f"Processing Scenes ({num_workers} CPUs)",
                    total=len(env.scenes),
                    disable=(rank > 0),
                )
            )
        eval_cliques = scene_cliques[0]
        for i in range(1, len(scene_cliques)):
            eval_cliques = eval_cliques + scene_cliques[i]
    eval_data = clique_dataset(eval_cliques)
    eval_data_loader = DataLoader(
        eval_data,
        collate_fn=clique_collate,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
    )

    num_sample_list = [1, 2, 3, 5, 10]

    for num_samples in num_sample_list:
        total_ADE = {nt: 0 for nt in env.node_type_list}
        total_FDE = {
            nt: np.zeros(hyperparams["prediction_horizon"]) for nt in env.node_type_list
        }
        total_ADE_count = {nt: 0 for nt in env.node_type_list}
        total_FDE_count = {
            nt: np.zeros(hyperparams["prediction_horizon"]) for nt in env.node_type_list
        }
        total_coll_score = {nt: 0 for nt in env.node_type_list}
        total_nt_count = {nt: 0 for nt in env.node_type_list}
        nt = "PEDESTRIAN"
        with torch.no_grad():
            pbar = tqdm(eval_data_loader, ncols=80)
            # pbar = tqdm(eval_data_loader, ncols=80, unit_scale=dist.get_world_size(), disable=(rank > 0))
            for batch in pbar:
                (
                    eval_loss,
                    ADE,
                    FDE,
                    ADE_count,
                    FDE_count,
                    coll_score,
                    nt_count,
                ) = ScePT_model.eval_loss(batch, num_samples, criterion=1)
                for nt in env.node_type_list:
                    total_ADE[nt] += ADE[nt]
                    total_FDE[nt] += FDE[nt]
                    total_ADE_count[nt] += ADE_count[nt]
                    total_FDE_count[nt] += FDE_count[nt]
                    total_coll_score[nt] += coll_score[nt]
                    total_nt_count[nt] += nt_count[nt]
                pbar.set_description(
                    f"L: {eval_loss:.2f},ADE: {ADE[nt]:.2f},FDE: {FDE[nt][-1]:.2f}"
                )
        for nt in env.node_type_list:
            print(f"Average ADE: {total_ADE[nt]/total_ADE_count[nt]}")
            print(f"Average FDE: {total_FDE[nt]/total_FDE_count[nt]}")
            print(f"Average collision score: {total_coll_score[nt]/total_nt_count[nt]}")


def plot_snapshot(rank):
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(data_path, "rb") as f:
        env = dill.load(f)

    model_dir = os.path.join(args.log_dir, args.log_tag + args.trained_model_dir)
    model_registrar = ModelRegistrar(model_dir, args.device)
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)
    hyperparams["use_map_encoding"] = args.map_encoding
    hyperparams["offline_scene_graph"] = args.offline_scene_graph
    hyperparams["incl_robot_node"] = args.incl_robot_node

    # hyperparams['max_clique_size'] = 8

    hyperparams["edge_encoding"] = not args.no_edge_encoding
    ScePT_model = ScePT(model_registrar, hyperparams, None, args.device)
    model_registrar.load_models(iter_num=args.iter_num)
    model_registrar.model_dict["policy_net"].max_Nnode = hyperparams["max_clique_size"]
    ScePT_model.set_environment(env)
    if (
        args.eval_data_dict == "nuScenes_train.pkl"
        or args.eval_data_dict == "nuScenes_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-trainval_meta"
    elif (
        args.eval_data_dict == "nuScenes_mini_train.pkl"
        or args.eval_data_dict == "nuScenes_mini_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-mini"
    else:
        nusc_path = None
    scene = env.scenes[2]
    with open("example.pkl", "wb") as file:
        dill.dump(scene, file)

    ft = hyperparams["prediction_horizon"]

    max_clique_size = hyperparams["max_clique_size"]
    timesteps = [15]
    default_con = dict()
    input_scale = dict()
    dynamics = dict()
    for nt in hyperparams["dynamic"]:
        model = getattr(dynamic_module, hyperparams["dynamic"][nt]["name"])
        input_scale[nt] = torch.tensor(hyperparams["dynamic"][nt]["limits"])
        dynamics[nt] = model(env.dt, input_scale[nt], "cpu", None, None, nt)
        model = getattr(thismodule, hyperparams["dynamic"][nt]["default_con"])
        default_con[nt] = model
    batch = obtain_clique_from_scene(
        scene,
        hyperparams["adj_radius"],
        hyperparams["maximum_history_length"],
        ft,
        hyperparams,
        max_clique_size=hyperparams["max_clique_size"],
        dynamics=dynamics,
        con=default_con,
        time_steps=timesteps,
        nusc_path=nusc_path,
    )

    (
        clique_type,
        clique_state_history,
        clique_first_timestep,
        clique_last_timestep,
        clique_edges,
        clique_future_state,
        clique_map,
        clique_node_size,
        clique_is_robot,
        clique_lane,
        clique_lane_dev,
        clique_fut_lane_dev,
    ) = zip(*batch)
    clique_robot_traj = dict()
    for i in range(len(clique_type)):
        for j in range(len(clique_type[i])):
            if clique_is_robot[i][j]:
                clique_robot_traj[(i, j)] = clique_future_state[i][j]

    (
        clique_state_pred,
        clique_input_pred,
        clique_ref_traj,
        clique_pi_list,
    ) = ScePT_model.predict(
        clique_type,
        clique_state_history,
        clique_first_timestep,
        clique_edges,
        clique_map,
        clique_node_size,
        clique_is_robot,
        clique_lane,
        clique_lane_dev,
        ft,
        num_samples=5,
        clique_robot_traj=clique_robot_traj,
    )

    anim = False
    if anim == False:
        fig, ax = visualization.plot_trajectories_clique(
            clique_type,
            clique_last_timestep,
            clique_state_history,
            clique_future_state,
            clique_state_pred,
            clique_ref_traj,
            scene.map["VISUALIZATION"] if scene.map is not None else None,
            clique_node_size,
            clique_is_robot,
            limits=[100, 100],
            emphasized_nodes=[],
        )
        plt.show()
    else:
        visualization.animate_traj_pred_clique(
            scene.dt,
            clique_type,
            clique_last_timestep,
            clique_state_history,
            clique_future_state,
            clique_state_pred,
            clique_ref_traj,
            scene.map["VISUALIZATION"] if scene.map is not None else None,
            clique_node_size,
            clique_is_robot,
            limits=[100, 100],
            output="anim.mp4",
            interp_N=5,
        )


def plot_snapshot_conditioning(rank):
    # nusc_col_26_Oct_2021_17_24_42,mini_train
    # nusc_col_01_Nov_2021_default_con,val
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(data_path, "rb") as f:
        env = dill.load(f)

    model_dir = os.path.join(args.log_dir, args.log_tag + args.trained_model_dir)
    model_registrar = ModelRegistrar(model_dir, args.device)
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)
    hyperparams["use_map_encoding"] = args.map_encoding
    hyperparams["offline_scene_graph"] = args.offline_scene_graph
    hyperparams["incl_robot_node"] = args.incl_robot_node

    hyperparams["max_clique_size"] = 12

    hyperparams["edge_encoding"] = not args.no_edge_encoding
    ScePT_model = ScePT(model_registrar, hyperparams, None, args.device)
    model_registrar.load_models(iter_num=args.iter_num)
    model_registrar.model_dict["policy_net"].max_Nnode = hyperparams["max_clique_size"]
    ScePT_model.set_environment(env)
    if (
        args.eval_data_dict == "nuScenes_train.pkl"
        or args.eval_data_dict == "nuScenes_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-trainval_meta"
    elif (
        args.eval_data_dict == "nuScenes_mini_train.pkl"
        or args.eval_data_dict == "nuScenes_mini_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-mini"
    else:
        nusc_path = None

    # 5
    # 24
    scene = env.scenes[5]
    with open("example.pkl", "wb") as file:
        dill.dump(scene, file)

    ft = hyperparams["prediction_horizon"]

    max_clique_size = hyperparams["max_clique_size"]
    # 18
    # 14
    timesteps = [18]

    batch = obtain_clique_from_scene(
        scene,
        hyperparams["adj_radius"],
        hyperparams["maximum_history_length"],
        ft,
        hyperparams,
        max_clique_size=hyperparams["max_clique_size"],
        time_steps=timesteps,
        nusc_path=nusc_path,
    )

    (
        clique_type,
        clique_state_history,
        clique_first_timestep,
        clique_last_timestep,
        clique_edges,
        clique_future_state,
        clique_map,
        clique_node_size,
        clique_is_robot,
        clique_lane,
        clique_lane_dev,
        clique_fut_lane_dev,
    ) = zip(*batch)
    emph_nodes = [(0, 0), (0, 1), (0, 2), (0, 3)]
    # emph_nodes = [(1, 1), (1, 0), (1, 4)]
    clique_robot_traj = dict()
    for i in range(len(clique_type)):
        for j in range(len(clique_type[i])):
            if clique_is_robot[i][j]:
                clique_robot_traj[(i, j)] = clique_future_state[i][j]

    (
        clique_state_pred,
        clique_input_pred,
        clique_ref_traj,
        clique_pi_list,
    ) = ScePT_model.predict(
        clique_type,
        clique_state_history,
        clique_first_timestep,
        clique_edges,
        clique_map,
        clique_node_size,
        clique_is_robot,
        clique_lane,
        clique_lane_dev,
        ft,
        num_samples=1,
        clique_robot_traj=clique_robot_traj,
    )
    # fig, ax = visualization.plot_trajectories_clique(
    #         clique_type,
    #         clique_last_timestep,
    #         clique_state_history,
    #         clique_future_state,
    #         clique_state_pred,
    #         clique_ref_traj,
    #         scene.map["VISUALIZATION"] if scene.map is not None else None,
    #         clique_node_size,
    #         clique_is_robot,
    #         limits=[100, 100],
    #         emphasized_nodes=emph_nodes,
    #     )
    #     # emphasized_nodes=[(3, 2), (3, 3), (3, 0), (3, 1)]
    # plt.show()
    # pdb.set_trace()
    anim = False
    if anim == False:
        fig, ax = visualization.plot_trajectories_clique(
            clique_type,
            clique_last_timestep,
            clique_state_history,
            clique_future_state,
            clique_state_pred,
            clique_ref_traj,
            scene.map["VISUALIZATION"] if scene.map is not None else None,
            clique_node_size,
            clique_is_robot,
            limits=[60, 60],
            emphasized_nodes=emph_nodes,
        )
        plt.show()
    else:
        visualization.animate_traj_pred_clique(
            scene.dt,
            clique_type,
            clique_last_timestep,
            clique_state_history,
            clique_future_state,
            clique_state_pred,
            clique_ref_traj,
            scene.map["VISUALIZATION"] if scene.map is not None else None,
            clique_node_size,
            clique_is_robot,
            limits=[100, 100],
            output="cond.mp4",
            interp_N=5,
            emphasized_nodes=emph_nodes,
        )

    clique_robot_traj = dict()

    idx = [(0, 1), (0, 2)]
    # idx = [(1, 4)]
    for (i, j) in idx:
        clique_is_robot[i][j] = True
        x = np.array(clique_state_history[i][j][-1])
        traj = clique_future_state[i][j] * 0
        for t in range(0, traj.shape[0]):
            if x[2] > 5 * env.dt:
                u = [-5, 0]
                if (i, j) == (0, 2):
                    u = [-2, 0]
            else:
                u = [-x[2] / env.dt, 0]
            # u = [4, 0]

            dxdt = np.array([x[2] * np.cos(x[3]), x[2] * np.sin(x[3]), u[0], u[1]])
            x += dxdt * env.dt
            traj[t] = np.array(x)
        clique_robot_traj[(i, j)] = traj

    (
        clique_state_pred,
        clique_input_pred,
        clique_ref_traj,
        clique_pi_list,
    ) = ScePT_model.predict(
        clique_type,
        clique_state_history,
        clique_first_timestep,
        clique_edges,
        clique_map,
        clique_node_size,
        clique_is_robot,
        clique_lane,
        clique_lane_dev,
        ft,
        num_samples=4,
        clique_robot_traj=clique_robot_traj,
    )

    if anim == False:
        fig, ax = visualization.plot_trajectories_clique(
            clique_type,
            clique_last_timestep,
            clique_state_history,
            clique_future_state,
            clique_state_pred,
            clique_ref_traj,
            scene.map["VISUALIZATION"] if scene.map is not None else None,
            clique_node_size,
            clique_is_robot,
            limits=[60, 60],
            emphasized_nodes=emph_nodes,
        )

        plt.show()
    else:
        visualization.animate_traj_pred_clique(
            scene.dt,
            clique_type,
            clique_last_timestep,
            clique_state_history,
            clique_future_state,
            clique_state_pred,
            clique_ref_traj,
            scene.map["VISUALIZATION"] if scene.map is not None else None,
            clique_node_size,
            clique_is_robot,
            limits=[100, 100],
            output="cond1.mp4",
            interp_N=5,
            emphasized_nodes=emph_nodes,
        )


if __name__ == "__main__":
    
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"

    eval(args.eval_task)(args.local_rank)
    # plot_snapshot_conditioning(args.local_rank)
    # plot_snapshot(args.local_rank)
    # make_video(args.local_rank)
    # simulate_prediction(args.local_rank)
    # sim_planning(args.local_rank)
    # eval_statistics(args.local_rank)
