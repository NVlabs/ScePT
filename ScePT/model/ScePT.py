import torch
from torch import nn
import numpy as np
from model.mgcvae_clique import MultimodalGenerativeCVAE_clique
from model.dataset import (
    get_timesteps_data,
    restore,
    obtain_clique_from_scene,
    generate_clique,
)
import time


class ScePT(nn.Module):
    def __init__(self, model_registrar, hyperparams, log_writer, device):
        super(ScePT, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = nn.ModuleDict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams["minimum_history_length"]
        self.max_ht = self.hyperparams["maximum_history_length"]
        self.ph = self.hyperparams["prediction_horizon"]
        self.state = self.hyperparams["state"]
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[state_type].values()
                    ]
                )
            )
        self.pred_state = self.hyperparams["pred_state"]

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()
        self.model = MultimodalGenerativeCVAE_clique(
            env=env,
            node_types=env.NodeType,
            model_registrar=self.model_registrar,
            hyperparams=self.hyperparams,
            device=self.device,
            edge_types=edge_types,
            log_writer=self.log_writer,
        )

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        self.model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        self.model.set_annealing_params()

    def step_annealers(self):
        self.model.step_annealers()

    def forward(self, batch):
        return self.train_loss(batch)

    def train_loss(self, batch):
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
        ) = batch

        loss = self.model(
            clique_type=clique_type,
            clique_state_history=clique_state_history,
            clique_first_timestep=clique_first_timestep,
            clique_last_timestep=clique_last_timestep,
            clique_edges=clique_edges,
            clique_future_state=clique_future_state,
            clique_map=clique_map,
            clique_node_size=clique_node_size,
            clique_is_robot=clique_is_robot,
            clique_lane=clique_lane,
            clique_lane_dev=clique_lane_dev,
            clique_fut_lane_dev=clique_fut_lane_dev,
        )

        return loss

    def eval_loss(self, batch, num_samples=None, criterion=0):
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
        ) = batch

        (
            loss,
            ADE,
            FDE,
            ADE_count,
            FDE_count,
            coll_score,
            nt_count,
        ) = self.model.eval_loss(
            clique_type=clique_type,
            clique_state_history=clique_state_history,
            clique_first_timestep=clique_first_timestep,
            clique_last_timestep=clique_last_timestep,
            clique_edges=clique_edges,
            clique_future_state=clique_future_state,
            clique_map=clique_map,
            clique_node_size=clique_node_size,
            clique_is_robot=clique_is_robot,
            clique_lane=clique_lane,
            clique_lane_dev=clique_lane_dev,
            clique_fut_lane_dev=clique_fut_lane_dev,
            num_samples=num_samples,
            criterion=criterion,
        )

        ADE_np = {nt: ADE[nt].cpu().detach().numpy() for nt in self.model.node_types}
        FDE_np = {nt: FDE[nt].cpu().detach().numpy() for nt in self.model.node_types}
        FDE_count_np = {
            nt: FDE_count[nt].cpu().detach().numpy() for nt in self.model.node_types
        }
        return (
            loss.cpu().detach().numpy(),
            ADE_np,
            FDE_np,
            ADE_count,
            FDE_count_np,
            coll_score,
            nt_count,
        )

    def snapshot_predict(self, scene, timesteps, ft, num_samples=1, nusc_path=None):

        batches = obtain_clique_from_scene(
            scene,
            self.hyperparams["adj_radius"],
            self.hyperparams["maximum_history_length"],
            ft,
            self.hyperparams,
            max_clique_size=self.hyperparams["max_clique_size"],
            time_steps=timesteps,
            time_series=True,
            nusc_path=nusc_path,
        )
        if len(batches) == 0:
            return None, None, None, None, None, None, None, None, None
        results = list()
        for batch in batches:
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
            else:
                processed_map = None

            (
                clique_state_pred,
                clique_input_pred,
                clique_ref_traj,
                clique_pi_list,
            ) = self.model.predict(
                clique_type,
                clique_state_history,
                clique_first_timestep,
                clique_edges,
                processed_map,
                clique_node_size,
                clique_is_robot,
                clique_lane,
                clique_lane_dev,
                None,
                ft,
                num_samples,
            )

            results.append(
                (
                    clique_type,
                    clique_first_timestep,
                    clique_last_timestep,
                    clique_state_history,
                    clique_future_state,
                    clique_state_pred,
                    clique_input_pred,
                    clique_ref_traj,
                    clique_pi_list,
                    clique_node_size,
                    clique_is_robot,
                )
            )
        return results

    def predict(
        self,
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
        clique_robot_traj=None,
    ):

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
        else:
            processed_map = None

        if clique_robot_traj is None:
            (
                clique_state_pred,
                clique_input_pred,
                clique_ref_traj,
                clique_pi_list,
            ) = self.model.predict(
                clique_type,
                clique_state_history,
                clique_first_timestep,
                clique_edges,
                processed_map,
                clique_node_size,
                clique_is_robot,
                clique_lane,
                clique_lane_dev,
                clique_robot_traj,
                ft,
                num_samples,
            )
        else:
            (
                clique_state_pred,
                clique_input_pred,
                clique_ref_traj,
                clique_pi_list,
            ) = self.model.predict(
                clique_type,
                clique_state_history,
                clique_first_timestep,
                clique_edges,
                processed_map,
                clique_node_size,
                clique_is_robot,
                clique_lane,
                clique_lane_dev,
                clique_robot_traj,
                ft,
                num_samples,
                incl_robot_future=True,
            )

        return clique_state_pred, clique_input_pred, clique_ref_traj, clique_pi_list

    def replay_prediction(
        self,
        scene,
        timesteps,
        ft,
        max_clique_size,
        dynamics,
        con,
        num_samples=1,
        center_node=None,
        calc_safety_measure=False,
        nusc_path=None,
    ):
        start = time.time()
        print("start data processing")
        if center_node is None:
            batches = obtain_clique_from_scene(
                scene,
                self.hyperparams["adj_radius"],
                self.hyperparams["maximum_history_length"],
                ft,
                self.hyperparams,
                max_clique_size=max_clique_size,
                dynamics=dynamics,
                con=con,
                time_steps=timesteps,
                return_nodes=True,
                time_series=True,
                center_node=None,
                nusc_path=nusc_path,
            )
            extra_node_info = None
        else:
            batches, extra_node_info = obtain_clique_from_scene(
                scene,
                self.hyperparams["adj_radius"],
                self.hyperparams["maximum_history_length"],
                ft,
                self.hyperparams,
                max_clique_size=max_clique_size,
                dynamics=dynamics,
                con=con,
                time_steps=timesteps,
                return_nodes=True,
                time_series=True,
                center_node=center_node,
                nusc_path=nusc_path,
            )
        end = time.time()
        print("data preparation:", end - start)
        results = list()
        safety_measure = list()
        print("start prediction")
        start = time.time()
        for t in range(len(batches)):

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
            ) = zip(*batches[t])

            clique_type = [None] * len(clique_nodes)
            for i in range(len(clique_nodes)):

                clique_type[i] = [node.type for node in clique_nodes[i]]

            (
                clique_state_pred,
                clique_input_pred,
                clique_ref_traj,
                clique_pi_list,
            ) = self.predict(
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
                num_samples=num_samples,
                clique_robot_traj=None,
            )

            results.append(
                (
                    clique_nodes,
                    clique_state_history,
                    clique_state_pred,
                    clique_node_size,
                    clique_pi_list,
                    clique_lane,
                )
            )

            if calc_safety_measure:
                safety_measure.append(
                    self.model.safety_measure(
                        clique_nodes,
                        clique_state_pred,
                        clique_node_size,
                        clique_pi_list,
                        gamma=0.5,
                    )
                )
        end = time.time()
        print("prediction takes", end - start)
        return results, extra_node_info, safety_measure

    def simulate_prediction(self, scene, start_t, ht, horizon, max_clique_size):
        scene_nodes = list()
        node_types = list()
        node_states = list()
        node_first_timestep = list()
        scene_node_size = list()
        adj_radius = self.hyperparams["adj_radius"]
        for node in scene.nodes:
            if node.first_timestep <= start_t and node.last_timestep >= start_t:
                scene_nodes.append(node)
                node_types.append(node.type)
                node_states.append(
                    node.get(
                        np.array([start_t - ht, start_t]),
                        self.hyperparams["state"][node.type],
                        padding=0.0,
                    )
                )
                node_first_timestep.append(max(0, ht - start_t + node.first_timestep))
                scene_node_size.append(
                    np.array(
                        [
                            node.length
                            if node.length is not None
                            else self.hyperparams["default_size"][node.type][0],
                            node.width
                            if node.width is not None
                            else self.hyperparams["default_size"][node.type][1],
                        ]
                    )
                )
        N = len(node_states)

        if scene.map is None:
            maps = None
        else:
            scene_node_maps = [None] * N
            map_node_idx = list()
            scene_map = list()
            for i in range(N):
                if node_types[i] in self.hyperparams["map_encoder"]:
                    map_node_idx.append(i)
                    scene_map.append(scene.map[node_types[i]])

            patch_size = self.hyperparams["map_encoder"][node_types[map_node_idx[0]]][
                "patch_size"
            ]

        node_idx = dict()
        results = list()
        for t in range(start_t, start_t + horizon):
            if scene.map is not None:

                scene_pts = [node_states[i][-1, 0:2] for i in map_node_idx]
                heading = [
                    node_states[i][
                        -1,
                        self.hyperparams["map_encoder"][node_types[i]][
                            "heading_state_index"
                        ],
                    ]
                    for i in map_node_idx
                ]
                maps = scene_map[0].get_cropped_maps_from_scene_map_batch(
                    scene_map,
                    scene_pts=torch.Tensor(scene_pts),
                    patch_size=patch_size,
                    rotation=heading,
                )
                for i in range(len(map_node_idx)):
                    scene_node_maps[map_node_idx[i]] = maps[i]

            n_components, labels, edges = generate_clique(
                node_types, node_states, adj_radius, max_clique_size
            )

            clique_type = list()
            clique_nodes = list()
            clique_state_history = list()
            clique_first_timestep = list()
            clique_edges = list()
            processed_map = list()
            clique_node_size = list()
            clique_is_robot = list()
            if maps is None:
                processed_map = None
            else:
                processed_map = list()
            for k in range(0, n_components):
                indices = np.where(labels == k)[0]
                for j in range(indices.shape[0]):
                    node_idx[indices[j]] = (k, j)
                clique_type.append([node_types[i] for i in indices])
                clique_nodes.append([scene_nodes[i] for i in indices])
                clique_edges.append(
                    [(i, j) for (i, j) in edges if labels[i] == k and labels[j] == k]
                )
                clique_state_history.append(
                    [np.array(node_states[i][-ht - 1 :]) for i in indices]
                )
                clique_first_timestep.append([node_first_timestep[i] for i in indices])
                clique_node_size.append([scene_node_size[i] for i in indices])
                clique_is_robot.append([False] * indices.shape[0])
                if maps is not None:
                    processed_map.append([scene_node_maps[i] for i in indices])

            (
                clique_state_pred,
                clique_input_pred,
                clique_ref_traj,
                clique_pi_list,
            ) = self.model.predict(
                clique_type,
                clique_state_history,
                clique_first_timestep,
                clique_edges,
                processed_map,
                clique_node_size,
                clique_is_robot,
                None,
                self.ph,
                1,
            )
            for i in range(N):
                (k, j) = node_idx[i]
                node_states[i] = np.vstack(
                    (node_states[i], clique_state_pred[k][j][0][0])
                )

            results.append(
                (
                    clique_nodes,
                    clique_state_history,
                    clique_state_pred,
                    clique_ref_traj,
                    clique_node_size,
                )
            )
        return results
