import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
import community as community_louvain
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from numpy.linalg import norm
from tqdm import tqdm
import pdb
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap
import sys, os
from model.dynamics import *
from model.model_utils import *
from functools import partial

container_abcs = collections.abc


def smooth_angle_kinks(theta0):
    theta = np.array(theta0)
    for i in range(1, theta.shape[0]):
        if theta[i] > theta[i - 1] + np.pi:
            theta[i] -= 2 * np.pi
        elif theta[i] < theta[i - 1] - np.pi:
            theta[i] += 2 * np.pi
    return theta


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


def collate(batch):
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, container_abcs.Sequence):
        if (
            len(elem) == 4
        ):  # We assume those are the maps, map points, headings and patch_size
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(
                scene_map,
                scene_pts=torch.Tensor(scene_pts),
                patch_size=patch_size[0],
                rotation=heading_angle,
            )
            return map
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return (
            dill.dumps(neighbor_dict)
            if torch.utils.data.get_worker_info()
            else neighbor_dict
        )
    return default_collate(batch)


def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    # TODO: We will have to make this more generic if robot_type != node_type
    # Make Robot State relative to node
    _, std = env.get_standardize_params(state[robot_type], node_type=robot_type)
    std[0:2] = env.attention_radius[(node_type, robot_type)]

    robot_traj_st = env.standardize(
        robot_traj, state[robot_type], node_type=robot_type, mean=node_traj, std=std
    )

    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)

    return robot_traj_st_t


def get_node_timestep_data(
    env,
    scene,
    t,
    node,
    state,
    pred_state,
    edge_types,
    max_ht,
    max_ft,
    hyperparams,
    scene_graph=None,
):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])

    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = np.array(x)[-1, 0:2]
    x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)
    if (
        list(pred_state[node.type].keys())[0] == "position"
    ):  # If we predict position we do it relative to current pos
        y_st = env.standardize(y, pred_state[node.type], node.type, mean=rel_state[0:2])
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    if hyperparams["edge_encoding"]:
        # Scene Graph
        scene_graph = (
            scene.get_scene_graph(
                t,
                env.attention_radius,
                hyperparams["edge_addition_filter"],
                hyperparams["edge_removal_filter"],
            )
            if scene_graph is None
            else scene_graph
        )

        neighbors_data_st = dict()
        neighbors_edge_value = dict()
        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams["dynamic_edges"] == "yes":
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(
                    scene_graph.get_edge_scaling(node), dtype=torch.float
                )
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(
                    np.array([t - max_ht, t]), state[connected_node.type], padding=0.0
                )

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(
                    state[connected_node.type], node_type=connected_node.type
                )
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(
                    neighbor_state_np,
                    state[connected_node.type],
                    node_type=connected_node.type,
                    mean=rel_state,
                    std=std,
                )

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)

    # Robot
    robot_traj_st_t = None
    if hyperparams['incl_robot_node']:
        timestep_range_r = np.array([t, t + max_ft])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=0.0)
        node_state = np.zeros_like(robot_traj[0])
        node_state[:x.shape[1]] = x[-1]
        robot_traj_st_t = get_relative_robot_traj(
            env, state, node_state, robot_traj, node.type, robot_type
        )

    # Map
    map_tuple = None
    if hyperparams["use_map_encoding"]:
        if node.type in hyperparams["map_encoder"]:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams["map_encoder"][node.type]
            if "heading_state_index" in me_hyp:
                heading_state_index = me_hyp["heading_state_index"]
                # We have to rotate the map in the opposit direction of the agent to match them
                if (
                    type(heading_state_index) is list
                ):  # infer from velocity or heading vector
                    heading_angle = (
                        -np.arctan2(
                            x[-1, heading_state_index[1]], x[-1, heading_state_index[0]]
                        )
                        * 180
                        / np.pi
                    )
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]

            patch_size = hyperparams["map_encoder"][node.type]["patch_size"]
            map_tuple = (scene_map, map_point, heading_angle, patch_size)

    return (
        first_history_index,
        x_t,
        y_t,
        x_st_t,
        y_st_t,
        neighbors_data_st,
        neighbors_edge_value,
        robot_traj_st_t,
        map_tuple,
    )


def get_timesteps_data(
    env,
    scene,
    t,
    node_type,
    state,
    pred_state,
    edge_types,
    min_ht,
    max_ht,
    min_ft,
    max_ft,
    hyperparams,
):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    nodes_per_ts = scene.present_nodes(
        t,
        type=node_type,
        min_history_timesteps=min_ht,
        min_future_timesteps=max_ft,
        return_robot=not hyperparams["incl_robot_node"],
    )
    batch = list()
    nodes = list()
    out_timesteps = list()
    for timestep in nodes_per_ts.keys():
        scene_graph = scene.get_scene_graph(
            timestep,
            env.attention_radius,
            hyperparams["edge_addition_filter"],
            hyperparams["edge_removal_filter"],
        )
        present_nodes = nodes_per_ts[timestep]
        for node in present_nodes:
            nodes.append(node)
            out_timesteps.append(timestep)
            batch.append(
                get_node_timestep_data(
                    env,
                    scene,
                    timestep,
                    node,
                    state,
                    pred_state,
                    edge_types,
                    max_ht,
                    max_ft,
                    hyperparams,
                    scene_graph=scene_graph,
                )
            )
    if len(out_timesteps) == 0:
        return None
    return collate(batch), nodes, out_timesteps


def break_graph(M, resol=1.0):
    if isinstance(M, np.ndarray):
        resol = resol * np.max(M)
        G = nx.Graph()
        for i in range(M.shape[0]):
            G.add_node(i)
        for i in range(M.shape[0]):
            for j in range(i + 1, M.shape[0]):
                if M[i, j] > 0:
                    G.add_edge(i, j, weight=M[i, j])
        partition = community_louvain.best_partition(G, resolution=resol)
    elif isinstance(M, networkx.classes.graph.Graph):
        G = M
        partition = community_louvain.best_partition(G, resolution=resol)

    while max(partition.values()) == 0 and resol >= 0.1:
        resol = resol * 0.9
        partition = community_louvain.best_partition(G, resolution=resol)
    return partition


def break_graph_recur(M, max_num):
    n_components, labels = connected_components(
        csgraph=csr_matrix(M), directed=False, return_labels=True
    )
    idx = 0

    while idx < n_components:
        subset = np.where(labels == idx)[0]
        if subset.shape[0] <= max_num:
            idx += 1
        else:
            partition = break_graph(M[np.ix_(subset, subset)])
            added_partition = 0
            for i in range(subset.shape[0]):
                if partition[i] > 0:
                    labels[subset[i]] = n_components + partition[i] - 1
                    added_partition = max(added_partition, partition[i])

            n_components += added_partition
            if added_partition == 0:
                idx += 1

    return n_components, labels


def generate_clique(node_types, node_state_history, adj_radius, max_clique_size):

    edges = list()
    adj_mat = np.zeros([len(node_types), len(node_types)])
    for i in range(0, len(node_types)):
        for j in range(0, len(node_types)):
            dis = norm(node_state_history[i][-1, 0:2] - node_state_history[j][-1, 0:2])

            radius = adj_radius[node_types[i]][node_types[j]]
            if j != i and dis < radius:
                adj_mat[i, j] = radius / dis
                adj_mat[j, i] = radius / dis
                edges.append((i, j))

        if max_clique_size is None:
            n_components, labels = connected_components(
                csgraph=csr_matrix(adj_mat), directed=False, return_labels=True
            )
        else:
            n_components, labels = break_graph_recur(adj_mat, max_clique_size)
    return n_components, labels, edges


def obtain_clique_from_scene(
    scene,
    adj_radius,
    ht,
    ft,
    hyperparams,
    dynamics=None,
    con=None,
    max_clique_size=None,
    time_steps=None,
    return_nodes=False,
    time_series=False,
    center_node=None,
    nusc_path=None,
):
    num_nodes = len(scene.nodes)
    T = scene.timesteps
    presence_table = np.zeros([num_nodes, T], dtype=np.bool)
    state = hyperparams["state"]
    if dynamics is not None and con is not None:
        use_deflt_traj = True
        safety_horizon = hyperparams["safety_horizon"]
    else:
        use_deflt_traj = False
    if nusc_path is not None:
        lane_info = hyperparams["lane_info"]
    else:
        lane_info = None
    if hasattr(scene, "nusc_map") and scene.nusc_map is not None:
        nusc_map = NuScenesMap(dataroot=nusc_path, map_name=scene.nusc_map)

    if center_node is not None:
        ego_idx = scene.nodes.index(center_node)
        extra_node_info = list()

    for i in range(0, num_nodes):
        presence_table[i][
            scene.nodes[i].first_timestep : scene.nodes[i].last_timestep
        ] = True

    if time_steps is None:
        time_steps = list(range(ht, T - ft))
    else:
        time_steps = [t for t in time_steps if t >= ht and t < T - ft]
    if time_series:
        result = [None] * len(time_steps)
    else:
        result = list()
    for t in time_steps:

        if time_series:
            result[time_steps.index(t)] = list()
        active_nodes = np.where(presence_table[:, t] == True)[0]
        active_nodes_state_history = list()
        active_nodes_future_state = list()
        active_nodes_lane_dev = list()
        active_nodes_fut_lane_dev = list()
        first_timestep = np.zeros(active_nodes.shape[0], dtype=np.int)
        last_timestep = ft * np.ones(active_nodes.shape[0], dtype=np.int)
        active_deflt_traj = list()
        for i in range(0, active_nodes.shape[0]):

            first_timestep[i] = max(
                0, ht - t + scene.nodes[active_nodes[i]].first_timestep
            )
            last_timestep[i] = min(ft, scene.nodes[active_nodes[i]].last_timestep - t)
            pred_end = min(scene.nodes[active_nodes[i]].last_timestep, t + ft)

            active_nodes_state_history.append(
                scene.nodes[active_nodes[i]].get(
                    np.array([t - ht, t]),
                    state[scene.nodes[active_nodes[i]].type],
                    padding=0.0,
                )
            )
            if lane_info is not None and scene.nodes[active_nodes[i]].type in lane_info:
                active_nodes_lane_dev.append(
                    scene.nodes[active_nodes[i]].get(
                        np.array([t - ht, t]),
                        lane_info[scene.nodes[active_nodes[i]].type],
                        padding=0.0,
                    )
                )
                active_nodes_fut_lane_dev.append(
                    scene.nodes[active_nodes[i]].get(
                        np.array([t + 1, t + ft]),
                        lane_info[scene.nodes[active_nodes[i]].type],
                        padding=0.0,
                    )
                )
            else:
                active_nodes_lane_dev.append(None)
                active_nodes_fut_lane_dev.append(None)
            active_nodes_future_state.append(
                scene.nodes[active_nodes[i]].get(
                    np.array([t + 1, t + ft]),
                    state[scene.nodes[active_nodes[i]].type],
                    padding=0.0,
                )
            )

            # if torch.isnan(active_nodes_future_state[-1]).any():
            #     pdb.set_trace()
            nt = scene.nodes[active_nodes[i]].type
            if use_deflt_traj:
                x_traj, _ = propagate_traj(
                    active_nodes_state_history[i][-1],
                    dynamics[nt],
                    con[nt],
                    scene.dt,
                    safety_horizon + 1,
                )
                active_deflt_traj.append(x_traj)
        edges = list()
        if center_node is None:

            adj_mat = np.zeros([active_nodes.shape[0], active_nodes.shape[0]])
            for i in range(0, active_nodes.shape[0]):
                for j in range(i + 1, active_nodes.shape[0]):
                    if use_deflt_traj:
                        dis = np.min(
                            np.linalg.norm(
                                active_deflt_traj[i][1:, :2]
                                - active_deflt_traj[j][1:, :2],
                                axis=-1,
                            )
                        )
                    else:
                        dis = norm(
                            active_nodes_state_history[i][-1, 0:2]
                            - active_nodes_state_history[j][-1, 0:2]
                        )

                    radius = adj_radius[scene.nodes[active_nodes[i]].type][
                        scene.nodes[active_nodes[j]].type
                    ]
                    if j != i and dis < radius:

                        adj_mat[i, j] = radius / dis
                        adj_mat[j, i] = radius / dis
                        edges.append((i, j))
                        edges.append((j, i))

            if max_clique_size is None:
                n_components, labels = connected_components(
                    csgraph=csr_matrix(adj_mat), directed=False, return_labels=True
                )
            else:
                n_components, labels = break_graph_recur(adj_mat, max_clique_size)

        else:
            labels = 10 * np.ones(len(active_nodes))

            if ego_idx not in active_nodes:
                n_components = 0
                extra_nodes = active_nodes
            else:
                nbs = list()
                nb_dis = list()
                ego_idx_in_active = np.where(active_nodes == ego_idx)[0][0]
                for i in range(0, active_nodes.shape[0]):
                    if i != ego_idx_in_active:
                        if use_deflt_traj:
                            dis = np.min(
                                np.linalg.norm(
                                    active_deflt_traj[i][1:, :2]
                                    - active_deflt_traj[ego_idx_in_active][1:, :2],
                                    axis=-1,
                                )
                            )
                        else:
                            dis = norm(
                                active_nodes_state_history[i][-1, 0:2]
                                - active_nodes_state_history[ego_idx_in_active][-1, 0:2]
                            )
                        radius = adj_radius[scene.nodes[active_nodes[i]].type][
                            center_node.type
                        ]
                        if dis < radius:
                            edges.append((ego_idx_in_active, i))
                            edges.append((i, ego_idx_in_active))
                            nbs.append(i)
                            nb_dis.append(radius / dis)
                if len(nbs) > max_clique_size - 1:
                    idx = np.argsort(-np.array(nb_dis))
                    nbs = np.array(nbs)[idx[: max_clique_size - 1]]
                labels[nbs] = 0
                labels[ego_idx_in_active] = 0
                n_components = 1
                extra_nodes = [
                    active_nodes[i]
                    for i in range(active_nodes.shape[0])
                    if labels[i] != 0
                ]
            extra_node_type = [scene.nodes[i].type for i in extra_nodes]

            extra_node_state = [
                active_nodes_state_history[i][-1]
                for i in range(len(active_nodes))
                if labels[i] != 0
            ]

            extra_node_size = [
                np.array(
                    [
                        scene.nodes[i].length
                        if scene.nodes[i].length is not None
                        else hyperparams["default_size"][scene.nodes[i].type][0],
                        scene.nodes[i].width
                        if scene.nodes[i].width is not None
                        else hyperparams["default_size"][scene.nodes[i].type][1],
                    ]
                )
                for i in extra_nodes
            ]

            extra_node_info.append((extra_node_type, extra_node_state, extra_node_size))

        # node_types = [scene.nodes[i].type for i in active_nodes]
        # n_components,labels,edges = generate_clique(node_types,active_nodes_state_history,adj_radius,max_clique_size)

        for k in range(0, n_components):

            indices = active_nodes[np.where(labels == k)[0]]

            clique_type = [scene.nodes[i].type for i in indices]
            clique_edges = [
                (i, j) for (i, j) in edges if labels[i] == k and labels[j] == k
            ]
            clique_state_history = [
                active_nodes_state_history[i]
                for i in range(0, labels.shape[0])
                if labels[i] == k
            ]
            clique_future_state = [
                active_nodes_future_state[i]
                for i in range(0, labels.shape[0])
                if labels[i] == k
            ]
            clique_lane_dev = [
                active_nodes_lane_dev[i]
                for i in range(0, labels.shape[0])
                if labels[i] == k
            ]
            clique_fut_lane_dev = [
                active_nodes_fut_lane_dev[i]
                for i in range(0, labels.shape[0])
                if labels[i] == k
            ]
            clique_first_timestep = first_timestep[labels == k]
            clique_last_timestep = last_timestep[labels == k]
            clique_lane = [None] * indices.shape[0]

            clique_node_size = [
                np.array(
                    [
                        scene.nodes[i].length
                        if scene.nodes[i].length is not None
                        else hyperparams["default_size"][scene.nodes[i].type][0],
                        scene.nodes[i].width
                        if scene.nodes[i].width is not None
                        else hyperparams["default_size"][scene.nodes[i].type][1],
                    ]
                )
                for i in indices
            ]

            clique_is_robot = [scene.nodes[i].is_robot for i in indices]

            if scene.map is None:
                clique_map = [None] * len(clique_type)
            else:
                clique_map = list()
                for i in range(len(clique_type)):
                    if (
                        "map_encoder" in hyperparams
                        and clique_type[i] in hyperparams["map_encoder"]
                    ):
                        if clique_type[i] == "VEHICLE":
                            x, y = (
                                clique_state_history[i][-1][0] + scene.x_min,
                                clique_state_history[i][-1][1] + scene.y_min,
                            )
                            x0, y0 = (
                                clique_state_history[i][-1][0],
                                clique_state_history[i][-1][1],
                            )
                            theta0 = clique_state_history[i][-1][3]
                            v0 = clique_state_history[i][-1][2]
                            closest_lane = nusc_map.get_closest_lane(x, y, radius=2)

                            if closest_lane != "":
                                lane_record = nusc_map.get_arcline_path(closest_lane)

                                interp_vel = max(abs(v0), 0.2)
                                poses = arcline_path_utils.discretize_lane(
                                    lane_record, resolution_meters=interp_vel * scene.dt
                                )
                                poses = np.array(poses)
                                poses[:, 0:2] -= np.array([scene.x_min, scene.y_min])

                                dis = norm(
                                    poses[:, 0:2] - clique_state_history[i][-1, 0:2],
                                    axis=1,
                                )
                                idx = np.argmin(dis)
                                if np.cos(theta0 - poses[idx, 2]) < 0.5:
                                    interp_vel = max(abs(v0), 0.2)
                                    s = (
                                        interp_vel
                                        * scene.dt
                                        * np.arange(ft).reshape([ft, 1])
                                    )
                                    clique_lane[i] = np.hstack(
                                        (
                                            s * np.cos(theta0) + x0,
                                            s * np.sin(theta0) + y0,
                                            interp_vel * np.ones([ft, 1]),
                                            theta0 * np.ones([ft, 1]),
                                        )
                                    )
                                else:
                                    if poses.shape[0] - idx >= ft:
                                        clique_lane[i] = np.hstack(
                                            (
                                                poses[idx : idx + ft, 0:2],
                                                interp_vel * np.ones([ft, 1]),
                                                poses[idx : idx + ft, 2:],
                                            )
                                        )
                                    else:
                                        len1 = poses.shape[0] - idx
                                        len2 = ft - len1
                                        lane = poses[idx:]
                                        prev_lane = closest_lane
                                        while len2 > 0:
                                            outlanes = nusc_map.get_outgoing_lane_ids(
                                                prev_lane
                                            )

                                            if len(outlanes) > 0:
                                                prev_lane = outlanes[0]
                                                try:
                                                    lane_record = (
                                                        nusc_map.get_arcline_path(
                                                            prev_lane
                                                        )
                                                    )
                                                    poses = arcline_path_utils.discretize_lane(
                                                        lane_record,
                                                        resolution_meters=interp_vel
                                                        * scene.dt,
                                                    )
                                                    poses = np.array(poses)
                                                    poses[:, 0:2] -= np.array(
                                                        [scene.x_min, scene.y_min]
                                                    )
                                                    lane = np.vstack(
                                                        (lane, poses[1 : 1 + len2])
                                                    )
                                                    len2 = max(
                                                        0, len2 - poses.shape[0] + 1
                                                    )
                                                except:
                                                    break
                                            else:
                                                break
                                        lane1 = np.hstack(
                                            (
                                                lane[:, 0:2],
                                                interp_vel
                                                * np.ones([lane.shape[0], 1]),
                                                lane[:, 2:],
                                            )
                                        )
                                        if len2 > 0:
                                            s = (
                                                interp_vel
                                                * scene.dt
                                                * np.arange(1, len2 + 1).reshape(
                                                    [len2, 1]
                                                )
                                            )
                                            theta = lane[-1, 2]
                                            lane2 = np.hstack(
                                                (
                                                    s * np.cos(theta) + lane1[-1, 0],
                                                    s * np.sin(theta) + lane1[-1, 1],
                                                    interp_vel * np.ones([len2, 1]),
                                                    theta * np.ones([len2, 1]),
                                                )
                                            )
                                            clique_lane[i] = np.vstack((lane1, lane2))
                                        else:
                                            clique_lane[i] = lane1
                            else:
                                interp_vel = max(abs(v0), 0.2)
                                s = (
                                    interp_vel
                                    * scene.dt
                                    * np.arange(ft).reshape([ft, 1])
                                )
                                clique_lane[i] = np.hstack(
                                    (
                                        s * np.cos(theta0) + x0,
                                        s * np.sin(theta0) + y0,
                                        interp_vel * np.ones([ft, 1]),
                                        theta0 * np.ones([ft, 1]),
                                    )
                                )

                            clique_lane[i][:, 3] = smooth_angle_kinks(
                                clique_lane[i][:, 3]
                            )
                        else:
                            clique_lane[i] = np.repeat(
                                clique_state_history[i][-1], ft, dim=0
                            )

                        patch_size = hyperparams["map_encoder"][clique_type[i]][
                            "patch_size"
                        ]
                        heading = clique_state_history[i][
                            -1,
                            hyperparams["map_encoder"][clique_type[i]][
                                "heading_state_index"
                            ],
                        ]
                        clique_map.append(
                            (
                                scene.map[clique_type[i]],
                                clique_state_history[i][-1, :2],
                                heading,
                                patch_size,
                            )
                        )
                    else:
                        clique_map.append(None)

            if time_series:
                if return_nodes:
                    clique_nodes = [scene.nodes[i] for i in indices]
                    result[time_steps.index(t)].append(
                        (
                            clique_nodes,
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
                        )
                    )
                else:
                    result[time_steps.index(t)].append(
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
                        )
                    )
            else:
                if return_nodes:
                    clique_nodes = [scene.nodes[i] for i in indices]
                    result.append(
                        (
                            clique_nodes,
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
                        )
                    )
                else:
                    result.append(
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
                        )
                    )

    if "nusc_map" in locals():
        del nusc_map
    if center_node is None:
        return result
    else:
        return result, extra_node_info


def get_IRL_data_from_scene(
    scene,
    node_types,
    dis_threshold,
    ht,
    hyperparams,
    dynamics,
    con,
    col_fun,
    time_steps=None,
    return_nodes=False,
    time_series=False,
    nusc_path=None,
):
    num_nodes = len(scene.nodes)
    safety_horizon = hyperparams["safety_horizon"]
    T = scene.timesteps
    presence_table = np.zeros([num_nodes, T], dtype=np.bool)
    state = hyperparams["state"]
    lane_info = hyperparams["lane_info"]
    if scene.nusc_map is not None:
        nusc_map = NuScenesMap(dataroot=nusc_path, map_name=scene.nusc_map)

    for i in range(0, num_nodes):
        presence_table[i][
            scene.nodes[i].first_timestep : scene.nodes[i].last_timestep
        ] = True
    if time_steps is None:
        time_steps = list(range(ht, T))
    else:
        time_steps = [t for t in time_steps if t >= ht and t < T]
    if time_series:
        result = [None] * len(time_steps)
    else:
        result = {nt: list() for nt in node_types}

    for t in time_steps:
        if time_series:
            result[time_steps.index(t)] = {nt: list() for nt in node_types}
        active_idx = np.where((presence_table[:, t - 1 : t + 2] == True).all(axis=1))[0]
        active_nodes_type = [scene.nodes[i].type for i in active_idx]
        active_nodes_state_history = list()
        active_nodes_input = list()
        active_nodes_lane_dev = list()
        first_timestep = np.zeros(active_idx.shape[0], dtype=np.int)
        active_deflt_traj = list()
        active_nodes_size = list()
        active_nodes_lane = list()
        active_nodes_map = list()
        for i in range(0, active_idx.shape[0]):
            nt = active_nodes_type[i]
            first_timestep[i] = max(
                0, ht - t + scene.nodes[active_idx[i]].first_timestep
            )
            active_nodes_size.append(
                np.array(
                    [
                        scene.nodes[active_idx[i]].length
                        if scene.nodes[active_idx[i]].length is not None
                        else hyperparams["default_size"][nt][0],
                        scene.nodes[active_idx[i]].width
                        if scene.nodes[active_idx[i]].width is not None
                        else hyperparams["default_size"][nt][1],
                    ]
                )
            )

            active_nodes_state_history.append(
                scene.nodes[active_idx[i]].get(
                    np.array([t - ht, t + 1]), state[nt], padding=0.0
                )
            )

            if nt in lane_info:
                active_nodes_lane_dev.append(
                    scene.nodes[active_idx[i]].get(
                        np.array([t - ht, t]), lane_info[nt], padding=0.0
                    )
                )
            else:
                active_nodes_lane_dev.append(None)

            if scene.nodes[active_idx[i]].first_timestep <= t - 1:
                u = dynamics[nt].inverse_dyn(
                    active_nodes_state_history[i][-2],
                    active_nodes_state_history[i][-1],
                    scene.dt,
                )
                u = np.clip(
                    u, dynamics[nt].dyn_limits[:, 0], dynamics[nt].dyn_limits[:, 1]
                )
                active_nodes_input.append(u)
                # if nt == "VEHICLE" and (
                #     (
                #         abs(active_nodes_input[-1][0]) > 6
                #         or abs(active_nodes_input[-1][1]) > 0.5
                #     )
                # ):
                #     pdb.set_trace()
            else:
                # pdb.set_trace()
                active_nodes_input.append(
                    np.zeros_like(
                        dynamics[nt].inverse_dyn(
                            active_nodes_state_history[i][-1],
                            active_nodes_state_history[i][-1],
                            scene.dt,
                        )
                    )
                )

            lane_horizon = hyperparams["lane_horizon"]
            if "map_encoder" in hyperparams and nt in hyperparams["map_encoder"]:
                if nt == "VEHICLE":
                    x, y = (
                        active_nodes_state_history[i][-1][0] + scene.x_min,
                        active_nodes_state_history[i][-1][1] + scene.y_min,
                    )
                    x0, y0 = (
                        active_nodes_state_history[i][-1][0],
                        active_nodes_state_history[i][-1][1],
                    )
                    v0 = active_nodes_state_history[i][-1][2]
                    closest_lane = nusc_map.get_closest_lane(x, y, radius=2)

                    if closest_lane != "":
                        lane_record = nusc_map.get_arcline_path(closest_lane)

                        interp_vel = max(0.2, v0)
                        poses = arcline_path_utils.discretize_lane(
                            lane_record, resolution_meters=interp_vel * scene.dt
                        )
                        poses = np.array(poses)
                        poses[:, 0:2] -= np.array([scene.x_min, scene.y_min])
                        dis = norm(
                            poses[:, 0:2] - active_nodes_state_history[i][-1, 0:2],
                            axis=1,
                        )
                        idx = np.argmin(dis)
                        if poses.shape[0] - idx >= lane_horizon:
                            active_nodes_lane.append(
                                np.hstack(
                                    (
                                        poses[idx : idx + lane_horizon, 0:2],
                                        interp_vel * np.ones([lane_horizon, 1]),
                                        poses[idx : idx + lane_horizon, 2:],
                                    )
                                )
                            )
                        else:
                            len1 = poses.shape[0] - idx
                            len2 = lane_horizon - len1
                            lane = poses[idx:]
                            prev_lane = closest_lane
                            while len2 > 0:
                                outlanes = nusc_map.get_outgoing_lane_ids(prev_lane)

                                if len(outlanes) > 0:
                                    prev_lane = outlanes[0]
                                    try:
                                        lane_record = nusc_map.get_arcline_path(
                                            prev_lane
                                        )
                                        poses = arcline_path_utils.discretize_lane(
                                            lane_record,
                                            resolution_meters=interp_vel * scene.dt,
                                        )
                                        poses = np.array(poses)
                                        poses[:, 0:2] -= np.array(
                                            [scene.x_min, scene.y_min]
                                        )
                                        lane = np.vstack((lane, poses[1 : 1 + len2]))
                                        len2 = max(0, len2 - poses.shape[0] + 1)
                                    except:
                                        break
                                else:
                                    break
                            lane1 = np.hstack(
                                (
                                    lane[:, 0:2],
                                    interp_vel * np.ones([lane.shape[0], 1]),
                                    lane[:, 2:],
                                )
                            )
                            if len2 > 0:
                                s = (
                                    interp_vel
                                    * scene.dt
                                    * np.arange(1, len2 + 1).reshape([len2, 1])
                                )
                                theta = lane[-1, 2]
                                lane2 = np.hstack(
                                    (
                                        s * np.cos(theta) + lane1[-1, 0],
                                        s * np.sin(theta) + lane1[-1, 1],
                                        interp_vel * np.ones([len2, 1]),
                                        theta * np.ones([len2, 1]),
                                    )
                                )
                                active_nodes_lane.append(np.vstack((lane1, lane2)))
                            else:
                                active_nodes_lane.append(lane1)
                    else:
                        interp_vel = max(v0, 0.2)
                        s = (
                            interp_vel
                            * scene.dt
                            * np.arange(lane_horizon).reshape([lane_horizon, 1])
                        )
                        theta = active_nodes_state_history[i][-1][3]
                        active_nodes_lane.append(
                            np.hstack(
                                (
                                    s * np.cos(theta) + x0,
                                    s * np.sin(theta) + y0,
                                    interp_vel * np.ones([lane_horizon, 1]),
                                    theta * np.ones([lane_horizon, 1]),
                                )
                            )
                        )
                    active_nodes_lane[i][:, 3] = smooth_angle_kinks(
                        active_nodes_lane[i][:, 3]
                    )
                else:
                    active_nodes_lane.append(
                        np.repeat(
                            active_nodes_state_history[i][-1], lane_horizon, dim=0
                        )
                    )

                patch_size = hyperparams["map_encoder"][nt]["patch_size"]
                heading = active_nodes_state_history[i][
                    -1, hyperparams["map_encoder"][nt]["heading_state_index"]
                ]
                active_nodes_map.append(
                    (
                        scene.map[nt],
                        active_nodes_state_history[i][-1, :2],
                        heading,
                        patch_size,
                    )
                )
            else:
                active_nodes_map.append(None)
                active_nodes_lane.append(None)

            if con[nt] == VEH_LK_control:
                con1 = partial(
                    VEH_LK_control,
                    line=active_nodes_lane[i][:, [0, 1, 3]],
                    Ky=hyperparams["Ky"],
                    Kpsi=hyperparams["Kpsi"],
                )
                x_traj, _ = propagate_traj(
                    active_nodes_state_history[i][-1],
                    dynamics[nt],
                    con1,
                    scene.dt,
                    safety_horizon + 1,
                )
            else:
                x_traj, _ = propagate_traj(
                    active_nodes_state_history[i][-1],
                    dynamics[nt],
                    con[nt],
                    scene.dt,
                    safety_horizon + 1,
                )
            active_deflt_traj.append(x_traj[1:])

        adj_mat = np.zeros([active_idx.shape[0], active_idx.shape[0]])
        for i in range(active_idx.shape[0]):
            adj_mat[i, i] = np.inf
            for j in range(i + 1, active_idx.shape[0]):
                et = (active_nodes_type[i], active_nodes_type[j])
                dis = safety_measure(
                    active_deflt_traj[i],
                    active_deflt_traj[j],
                    active_nodes_size[i],
                    active_nodes_size[j],
                    col_fun[et],
                )
                adj_mat[i, j] = dis
                adj_mat[j, i] = dis
        active_nodes = [scene.nodes[i] for i in active_idx]
        for i in range(active_idx.shape[0]):
            if (
                active_nodes_type[i] == "VEHICLE"
                and abs(active_nodes_state_history[i][-1, 2]) > 0.2
            ):
                nb = np.where(adj_mat[i] < dis_threshold)[0]
                nb_history = [active_nodes_state_history[j][0:-1] for j in nb]
                nb_node_size = [active_nodes_size[j] for j in nb]
                nb_types = [active_nodes_type[j] for j in nb]
                nb_nodes = [active_nodes[j] for j in nb]
                nb_first_time = [first_timestep[j] for j in nb]
                nb_deflt_traj = [active_deflt_traj[j][1:] for j in nb]
                if time_series:
                    if return_nodes:
                        result[time_steps.index(t)][active_nodes_type[i]].append(
                            (
                                active_nodes[i],
                                nb_nodes,
                                active_nodes_state_history[i][0:-1],
                                active_nodes_input[i],
                                first_timestep[i],
                                nb_history,
                                nb_first_time,
                                active_nodes_map[i],
                                active_nodes_size[i],
                                nb_node_size,
                                active_nodes_lane[i],
                                active_nodes_lane_dev[i],
                                nb_deflt_traj,
                            )
                        )
                    else:
                        result[time_steps.index(t)][active_nodes_type[i]].append(
                            (
                                nb_types,
                                active_nodes_state_history[i][0:-1],
                                active_nodes_input[i],
                                first_timestep[i],
                                nb_history,
                                nb_first_time,
                                active_nodes_map[i],
                                active_nodes_size[i],
                                nb_node_size,
                                active_nodes_lane[i],
                                active_nodes_lane_dev[i],
                                nb_deflt_traj,
                            )
                        )
                else:
                    if return_nodes:
                        result[active_nodes_type[i]].append(
                            (
                                active_nodes[i],
                                nb_nodes,
                                active_nodes_state_history[i][0:-1],
                                active_nodes_input[i],
                                first_timestep[i],
                                nb_history,
                                nb_first_time,
                                active_nodes_map[i],
                                active_nodes_size[i],
                                nb_node_size,
                                active_nodes_lane[i],
                                active_nodes_lane_dev[i],
                                nb_deflt_traj,
                            )
                        )
                    else:
                        result[active_nodes_type[i]].append(
                            (
                                nb_types,
                                active_nodes_state_history[i][0:-1],
                                active_nodes_input[i],
                                first_timestep[i],
                                nb_history,
                                nb_first_time,
                                active_nodes_map[i],
                                active_nodes_size[i],
                                nb_node_size,
                                active_nodes_lane[i],
                                active_nodes_lane_dev[i],
                                nb_deflt_traj,
                            )
                        )

    if "nusc_map" in locals():
        del nusc_map
    if time_series:
        return result, time_steps
    else:
        return result


def clique_collate(data):
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
    ) = zip(*data)
    bs = len(clique_type)
    scene_maps = list()
    scene_pts = list()
    heading = list()
    idx = list()
    patch_size = list()
    processed_map = [None] * bs
    for i in range(bs):
        processed_map[i] = [None] * len(clique_map[i])
        for j in range(len(clique_map[i])):
            if clique_map[i][j] is not None:
                scene_maps.append(clique_map[i][j][0])
                scene_pts.append(clique_map[i][j][1])
                heading.append(clique_map[i][j][2])
                patch_size.append(clique_map[i][j][3])
                idx.append((i, j))

    if len(scene_maps) > 0:
        maps = scene_maps[0].get_cropped_maps_from_scene_map_batch(
            scene_maps,
            scene_pts=torch.Tensor(scene_pts),
            patch_size=patch_size[0],
            rotation=heading,
        )

        for n in range(len(idx)):
            i, j = idx[n]
            processed_map[i][j] = maps[n]
    return (
        clique_type,
        clique_state_history,
        clique_first_timestep,
        clique_last_timestep,
        clique_edges,
        clique_future_state,
        processed_map,
        clique_node_size,
        clique_is_robot,
        clique_lane,
        clique_lane_dev,
        clique_fut_lane_dev,
    )


def IRL_collate(data):
    (
        nb_types,
        state_history,
        node_input,
        first_timestep,
        nb_history,
        nb_first_time,
        node_map,
        node_size,
        nb_node_size,
        lane,
        lane_dev,
        nb_deflt_traj,
    ) = zip(*data)
    bs = len(nb_types)
    scene_maps = list()
    scene_pts = list()
    heading = list()
    patch_size = list()
    idx = list()
    processed_map = [None] * bs
    for i in range(bs):
        if node_map[i] is not None:
            scene_maps.append(node_map[i][0])
            scene_pts.append(node_map[i][1])
            heading.append(node_map[i][2])
            patch_size.append(node_map[i][3])
            idx.append(i)

    if len(scene_maps) > 0:
        maps = scene_maps[0].get_cropped_maps_from_scene_map_batch(
            scene_maps,
            scene_pts=torch.Tensor(scene_pts),
            patch_size=patch_size[0],
            rotation=heading,
        )

        for n in range(len(idx)):
            processed_map[idx[n]] = maps[n]
    return (
        nb_types,
        state_history,
        node_input,
        first_timestep,
        nb_history,
        nb_first_time,
        processed_map,
        node_size,
        nb_node_size,
        lane,
        lane_dev,
        nb_deflt_traj,
    )
