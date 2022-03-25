from os import DirEntry
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.components import *
from model.model_utils import *
import model.dynamics as dynamic_module
import model.components as model_components
from environment.scene_graph import DirectedEdge
import pdb
import itertools
import sys
from utils import CVaR, CVaR_weight
from functools import partial

thismodule = sys.modules[__name__]


class MultimodalGenerativeCVAE_clique(nn.Module):
    def __init__(
        self,
        env,
        node_types,
        model_registrar,
        hyperparams,
        device,
        edge_types,
        log_writer=None,
    ):
        """
        node_types: list of node_types ('VEHICLE', 'PEDESTRIAN'...)
        edge_types: list of edge types
        """
        super(MultimodalGenerativeCVAE_clique, self).__init__()
        self.hyperparams = hyperparams
        self.env = env
        self.node_types = node_types
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types = edge_types
        self.curr_iter = 0
        self.z_dim = hyperparams["K"]
        self.edge_pre_enc_net = dict()
        self.node_modules = nn.ModuleDict()

        self.min_hl = self.hyperparams["minimum_history_length"]
        self.max_hl = self.hyperparams["maximum_history_length"]
        self.ph = self.hyperparams["prediction_horizon"]
        self.state = self.hyperparams["state"]
        self.pred_state = self.hyperparams["pred_state"]
        self.state_length = dict()
        self.input_length = dict()
        self.input_scale = dict()
        self.dynamic = dict()
        self.rel_state_fun = dict()
        self.collision_fun = dict()

        # relative state function, generate translation invariant relative state between nodes
        for nt in self.node_types:
            self.rel_state_fun[nt] = getattr(
                model_components, hyperparams["rel_state_fun"][nt]
            )
        # collision function between different types of nodes, return distance or collision penalty.
        for et in self.edge_types:
            params = hyperparams["collision_fun"][et[0]][et[1]]
            func = getattr(thismodule, params["func"])
            self.collision_fun[et] = partial(func, device=device, alpha=params["alpha"])
        self.max_Nnode = self.hyperparams["max_clique_size"]
        for node_type in self.node_types:
            self.state_length[node_type] = hyperparams["dynamic"][node_type][
                "state_dim"
            ]
            self.input_length[node_type] = hyperparams["dynamic"][node_type][
                "input_dim"
            ]
            self.input_scale[node_type] = torch.tensor(
                hyperparams["dynamic"][node_type]["limits"]
            )
            # dynamic model of each type of nodes
            model = getattr(dynamic_module, hyperparams["dynamic"][node_type]["name"])

            self.dynamic[node_type] = model(
                env.dt,
                self.input_scale[node_type],
                device,
                model_registrar,
                None,
                node_type,
            )
        if self.hyperparams["incl_robot_node"]:
            self.robot_state_length = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[env.robot_type].values()
                    ]
                )
            )
        self.pred_state_length = dict()
        for node_type in self.node_types:
            self.pred_state_length[node_type] = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.pred_state[node_type].values()
                    ]
                )
            )

        self.create_graphical_model(self.edge_types)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):

        #####################
        #   Edge Encoders   #
        #####################
        edge_encoder_input_size = self.hyperparams["edge_encoding_dim"]
        for edge_type in self.edge_types:

            model = getattr(
                model_components,
                self.hyperparams["edge_pre_enc_net"][edge_type[0]][edge_type[1]],
            )
            self.add_submodule(
                str(edge_type[0]) + "->" + str(edge_type[1]) + "/edge_pre_encoding",
                model_if_absent=model(edge_encoder_input_size, device=self.device),
            )
            self.edge_pre_enc_net[edge_type] = self.node_modules[
                str(edge_type[0]) + "->" + str(edge_type[1]) + "/edge_pre_encoding"
            ]

            self.add_submodule(
                str(edge_type[0]) + "->" + str(edge_type[1]) + "/edge_encoder",
                model_if_absent=nn.LSTM(
                    input_size=edge_encoder_input_size,
                    hidden_size=self.hyperparams["enc_rnn_dim_edge"],
                    batch_first=False,
                ),
            )

        ############################
        #   Node History Encoder   #
        ############################
        for node_type in self.node_types:
            model = getattr(
                model_components,
                self.hyperparams["node_pre_encode_net"][node_type]["module"],
            )
            enc_dim = self.hyperparams["node_pre_encode_net"][node_type]["enc_dim"]
            if (
                self.hyperparams["use_map_encoding"]
                and node_type in self.hyperparams["map_encoder"]
                and self.hyperparams["use_lane_info"]
            ):
                use_lane_info = True
            else:
                use_lane_info = False
            self.add_submodule(
                node_type + "/node_pre_encoder",
                model_if_absent=model(
                    enc_dim, self.device, use_lane_info=use_lane_info
                ),
            )

            self.add_submodule(
                node_type + "/node_history_encoder",
                model_if_absent=nn.LSTM(
                    input_size=enc_dim,
                    hidden_size=self.hyperparams["enc_rnn_dim_history"],
                    batch_first=False,
                ),
            )

        ###########################
        #   Node Future Encoder   #
        ###########################
        for node_type in self.node_types:
            self.add_submodule(
                node_type + "/node_future_encoder",
                model_if_absent=nn.LSTM(
                    input_size=self.hyperparams["node_pre_encode_net"][node_type][
                        "enc_dim"
                    ],
                    hidden_size=self.hyperparams["enc_rnn_dim_future"],
                    bidirectional=True,
                    batch_first=False,
                ),
            )

            # These are related to how you initialize states for the node future encoder.

            self.add_submodule(
                node_type + "/node_future_encoder/initial_h",
                model_if_absent=nn.Linear(
                    self.hyperparams["node_pre_encode_net"][node_type]["enc_dim"],
                    self.hyperparams["enc_rnn_dim_future"],
                ),
            )
            self.add_submodule(
                node_type + "/node_future_encoder/initial_c",
                model_if_absent=nn.Linear(
                    self.hyperparams["node_pre_encode_net"][node_type]["enc_dim"],
                    self.hyperparams["enc_rnn_dim_future"],
                ),
            )

        ###################
        #   Map Encoder   #
        ###################
        if self.hyperparams["use_map_encoding"]:
            for node_type in self.node_types:
                if node_type in self.hyperparams["map_encoder"]:
                    me_params = self.hyperparams["map_encoder"][node_type]
                    self.add_submodule(
                        node_type + "/map_encoder",
                        model_if_absent=CNNMapEncoder(
                            me_params["map_channels"],
                            me_params["hidden_channels"],
                            me_params["output_size"],
                            me_params["masks"],
                            me_params["strides"],
                            me_params["patch_size"],
                        ),
                    )

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################

        z_size = self.hyperparams["K"]

        ################
        #### p_z_x #####
        ################
        state_enc_dim = dict()
        for nt in self.node_types:
            if (
                self.hyperparams["use_map_encoding"]
                and nt in self.hyperparams["map_encoder"]
            ):
                state_enc_dim[nt] = (
                    self.hyperparams["enc_rnn_dim_history"]
                    + self.hyperparams["map_encoder"][nt]["output_size"]
                )
            else:
                state_enc_dim[nt] = self.hyperparams["enc_rnn_dim_history"]

        edge_encoding_dim = {
            et: self.hyperparams["enc_rnn_dim_edge"] for et in self.edge_types
        }
        # Gibbs distribution for joint latent distribution
        self.add_submodule(
            "p_z_x",
            model_if_absent=clique_gibbs_distr(
                state_enc_dim=state_enc_dim,
                edge_encoding_dim=edge_encoding_dim,
                z_dim=z_size,
                edge_types=self.edge_types,
                node_types=self.node_types,
                hyperparams=self.hyperparams,
                device=self.device,
                node_hidden_dim=[64, 64],
                edge_hidden_dim=[64, 64],
            ),
        )

        ################
        #### q_z_xy ####
        ################
        # Gibbs distribution for joint latent distribution
        for node_type in self.node_types:
            state_enc_dim[node_type] += self.hyperparams["enc_rnn_dim_future"] * 4
        self.add_submodule(
            "q_z_xy",
            model_if_absent=clique_gibbs_distr(
                state_enc_dim=state_enc_dim,
                edge_encoding_dim=edge_encoding_dim,
                z_dim=z_size,
                edge_types=self.edge_types,
                node_types=self.node_types,
                hyperparams=self.hyperparams,
                device=self.device,
                node_hidden_dim=[64, 64],
                edge_hidden_dim=[64, 64],
            ),
        )

        #################################
        ##  policy network as decoder  ##
        #################################

        if self.hyperparams["use_map_encoding"]:
            map_enc_dim = dict()
            for nt in self.node_types:
                if nt in self.hyperparams["map_encoder"]:
                    map_enc_dim[nt] = self.hyperparams["map_encoder"][nt]["output_size"]
        else:
            map_enc_dim = None

        self.add_submodule(
            "policy_net",
            model_if_absent=clique_guided_policy_net(
                device=self.device,
                node_types=self.node_types,
                edge_types=self.edge_types,
                input_dim=self.input_length,
                state_dim=self.state_length,
                z_dim=z_size,
                rel_state_fun=self.rel_state_fun,
                collision_fun=self.collision_fun,
                dyn_net=self.dynamic,
                edge_encoding_net=self.edge_pre_enc_net,
                edge_enc_dim=self.hyperparams["edge_encoding_dim"],
                map_enc_dim=map_enc_dim,
                history_enc_dim=self.hyperparams["enc_rnn_dim_history"],
                obs_lstm_hidden_dim=self.hyperparams["policy_obs_LSTM_hidden_dim"],
                guide_RNN_hidden_dim=self.hyperparams["dec_rnn_dim"],
                FC_hidden_dim=self.hyperparams["policy_FC_hidden_dim"],
                input_scale=self.input_scale,
                max_Nnode=self.max_Nnode,
                dt=self.env.dt,
                hyperparams=self.hyperparams,
            ),
        )

    def create_graphical_model(self, edge_types):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        self.create_node_models()

        for name, module in self.node_modules.items():
            module.to(self.device)

    def create_new_scheduler(
        self, name, annealer, annealer_kws, creation_condition=True
    ):
        value_scheduler = None
        rsetattr(self, name + "_scheduler", value_scheduler)
        if creation_condition:
            annealer_kws["device"] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + "_annealer", value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer(
                [rgetattr(self, name)], {"lr": value_annealer(0).clone().detach()}
            )
            rsetattr(self, name + "_optimizer", dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer, value_annealer)
            rsetattr(self, name + "_scheduler", value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(
            name="kl_weight",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": self.hyperparams["kl_weight_start"],
                "finish": self.hyperparams["kl_weight"],
                "center_step": self.hyperparams["kl_crossover"],
                "steps_lo_to_hi": self.hyperparams["kl_crossover"]
                / self.hyperparams["kl_sigmoid_divisor"],
            },
        )

        self.create_new_scheduler(
            name="gamma",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": self.hyperparams["gamma_init"],
                "finish": self.hyperparams["gamma_end"],
                "center_step": self.hyperparams["gamma_crossover"],
                "steps_lo_to_hi": self.hyperparams["gamma_crossover"]
                / self.hyperparams["gamma_sigmoid_divisor"],
            },
        )
        self.create_new_scheduler(
            name="collision_weight",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": self.hyperparams["col_weight_start"],
                "finish": self.hyperparams["col_weight"],
                "center_step": self.hyperparams["col_crossover"],
                "steps_lo_to_hi": self.hyperparams["col_crossover"]
                / self.hyperparams["col_sigmoid_divisor"],
            },
        )

        self.create_new_scheduler(
            name="ref_match_weight",
            annealer=exp_anneal,
            annealer_kws={
                "start": self.hyperparams["ref_match_weight_init"],
                "finish": self.hyperparams["ref_match_weight_final"],
                "rate": self.hyperparams["ref_match_weight_decay_rate"],
            },
        )

    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + "_scheduler") is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + "_scheduler").step()

                # Then we set the annealed vars' value.
                rsetattr(
                    self,
                    annealed_var,
                    rgetattr(self, annealed_var + "_optimizer").param_groups[0]["lr"],
                )

        self.summarize_annealers()

    def summarize_annealers(self):
        if self.log_writer is not None:
            for annealed_var in self.annealed_vars:
                if rgetattr(self, annealed_var) is not None:
                    self.log_writer.add_scalar(
                        "%s" % (annealed_var.replace(".", "/")),
                        rgetattr(self, annealed_var),
                        self.curr_iter,
                    )

    def obtain_encoded_tensors(
        self,
        mode,
        batch_state_history,
        batch_state_history_st,
        batch_state_future,
        batch_state_future_st,
        batch_edge,
        batch_first_timestep,
        batch_last_timestep,
        batch_edge_first_timestep,
        batch_map,
        batch_node_size,
        batch_lane_dev,
        batch_fut_lane_dev,
        indices,
    ):

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(
            mode, batch_state_history_st, batch_lane_dev, batch_first_timestep
        )

        ##################
        # Encode Future #
        ##################
        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            node_future_encoded = self.encode_node_future(
                mode,
                batch_state_history_st,
                batch_lane_dev,
                batch_state_future_st,
                batch_fut_lane_dev,
                batch_last_timestep,
            )

        else:
            node_future_encoded = None

        ##############################
        # Encode Node Edges per Type #
        ##############################
        batch_edge_hist_enc = self.encode_edge(
            mode, batch_edge, batch_edge_first_timestep, batch_node_size, indices
        )

        ################
        # Map Encoding #
        ################
        encoded_map = dict()
        if self.hyperparams["use_map_encoding"]:

            for nt in self.node_types:
                if nt in self.hyperparams["map_encoder"]:
                    encoded_map[nt] = self.node_modules[nt + "/map_encoder"](
                        batch_map[nt] * 2.0 - 1.0, (mode == ModeKeys.TRAIN)
                    )
                    do = self.hyperparams["map_encoder"][nt]["dropout"]
                    encoded_map[nt] = F.dropout(
                        encoded_map[nt], do, training=(mode == ModeKeys.TRAIN)
                    )
                    if encoded_map[nt].isnan().any():
                        pdb.set_trace()

        return (
            node_history_encoded,
            node_future_encoded,
            batch_edge_hist_enc,
            encoded_map,
        )

    def batch_clique(
        self,
        mode,
        clique_type,
        clique_state_history,
        clique_first_timestep,
        clique_last_timestep,
        clique_edges,
        clique_map,
        clique_node_size,
        clique_is_robot,
        clique_lane,
        clique_lane_dev,
        clique_future_state,
        clique_fut_lane_dev,
    ):
        """
        put clique data into batches by node type
        node_index: node index in batch -> (index of clique, index of node in the clique)
        edge_index: edge index in batch -> (index of clique, index of node 1 in the clique, index of node 2 in the clique)
        node_inverse_index: (index of clique, index of node in the clique) -> (node type, node index in the batch)
        batch_node_to_edge_index: edge index in batch -> (node 1 index in batch, node 2 index in batch)
        edge_to_node_index: edge index in batch -> (node 1 index in batch, index of node 2 among all neighbors of node 1)
        batch_edge_idx1: list of indices of node 1 for all edges of a particular edge type
        batch_edge_idx2: list of indices of node 2 for all edges of a particular edge type

        """

        bs = len(clique_type)
        ht = self.hyperparams["maximum_history_length"]
        ft = self.hyperparams["prediction_horizon"]
        node_count = {node_type: 0 for node_type in self.node_types}
        edge_count = {edge_type: 0 for edge_type in self.edge_types}
        node_index = {node_type: {} for node_type in self.node_types}
        node_inverse_index = dict()
        edge_index = {edge_type: {} for edge_type in self.edge_types}
        batch_node_to_edge_index = {edge_type: {} for edge_type in self.edge_types}
        edge_to_node_index = {edge_type: {} for edge_type in self.edge_types}
        Nnodes = torch.zeros(bs, dtype=torch.int)
        batch_lane = dict()
        batch_lane_dev = dict()
        batch_fut_lane_dev = dict()
        batch_lane_st = dict()

        clique_is_robot = list(clique_is_robot)

        for i in range(bs):
            Nnodes[i] = len(clique_type[i])

            for j in range(Nnodes[i]):
                ntj = clique_type[i][j]

                node_index[ntj][node_count[ntj]] = (i, j)
                node_inverse_index[(i, j)] = (ntj, node_count[ntj])

                nb_count = 0
                for k in range(Nnodes[i]):
                    if k != j:
                        et = (ntj, clique_type[i][k])
                        edge_index[et][edge_count[et]] = (i, j, k)
                        edge_to_node_index[et][edge_count[et]] = (
                            node_count[ntj],
                            nb_count,
                        )
                        nb_count += 1
                        edge_count[et] += 1

                node_count[ntj] += 1
        batch_map = dict()
        for nt in self.node_types:
            if nt in self.hyperparams["map_encoder"]:
                patch_size = self.hyperparams["map_encoder"][nt]["patch_size"]
                num_channels = self.hyperparams["map_encoder"][nt]["map_channels"]
                batch_map[nt] = torch.zeros(
                    [
                        len(node_index[nt]),
                        num_channels,
                        patch_size[0] + patch_size[2],
                        patch_size[1] + patch_size[3],
                    ]
                ).to(self.device)
                batch_lane[nt] = torch.zeros(
                    [ft, len(node_index[nt]), self.state_length[nt]]
                ).to(self.device)
                batch_lane_st[nt] = torch.zeros(
                    [ft, len(node_index[nt]), self.state_length[nt]]
                ).to(self.device)
                batch_lane_dev[nt] = torch.zeros([ht + 1, len(node_index[nt]), 2]).to(
                    self.device
                )
                batch_fut_lane_dev[nt] = torch.zeros([ft, len(node_index[nt]), 2]).to(
                    self.device
                )
        for et in self.edge_types:
            for idx, (i, j, k) in edge_index[et].items():
                _, idxj = node_inverse_index[(i, j)]
                _, idxk = node_inverse_index[(i, k)]
                batch_node_to_edge_index[et][idx] = (idxj, idxk)

        batch_first_timestep = {
            nt: torch.zeros(len(node_index[nt]), dtype=torch.int)
            for nt in self.node_types
        }
        batch_state_history = {
            nt: torch.zeros([ht + 1, len(node_index[nt]), self.state_length[nt]]).to(
                self.device
            )
            for nt in self.node_types
        }
        batch_state_history_st = {
            nt: torch.zeros([ht + 1, len(node_index[nt]), self.state_length[nt]]).to(
                self.device
            )
            for nt in self.node_types
        }
        batch_node_size = {
            nt: torch.zeros([len(node_index[nt]), 2]).to(self.device)
            for nt in self.node_types
        }
        batch_is_robot = {
            nt: torch.zeros(len(node_index[nt]), dtype=torch.bool).to(self.device)
            for nt in self.node_types
        }
        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:

            batch_state_future = {
                nt: torch.zeros([ft, len(node_index[nt]), self.state_length[nt]]).to(
                    self.device
                )
                for nt in self.node_types
            }
            batch_state_future_st = {
                nt: torch.zeros([ft, len(node_index[nt]), self.state_length[nt]]).to(
                    self.device
                )
                for nt in self.node_types
            }
            batch_last_timestep = {
                nt: torch.zeros(len(node_index[nt]), dtype=torch.int)
                for nt in self.node_types
            }

            for nt in self.node_types:
                if (
                    nt in self.hyperparams["map_encoder"]
                    and self.hyperparams["use_map_encoding"]
                ):
                    for idx, (i, j) in node_index[nt].items():
                        batch_state_history[nt][:, idx] = torch.tensor(
                            clique_state_history[i][j]
                        ).to(self.device)
                        first_timestep = clique_first_timestep[i][j]
                        x0 = batch_state_history[nt][-1, idx]
                        batch_state_history_st[nt][
                            first_timestep:, idx
                        ] = self.rel_state_fun[nt](
                            batch_state_history[nt][first_timestep:, idx],
                            x0.repeat(ht + 1 - first_timestep, 1),
                        )

                        # batch_lane_dev[nt][:,idx] = torch.tensor(np.clip(clique_lane_dev[i][j],[-3.0,-12],[3.0,12])).to(self.device)
                        batch_lane_dev[nt][:, idx] = torch.tensor(
                            clique_lane_dev[i][j]
                        ).to(self.device)

                        # batch_fut_lane_dev[nt][:,idx] = torch.tensor(np.clip(clique_fut_lane_dev[i][j],[-3.0,-12],[3.0,12])).to(self.device)
                        batch_fut_lane_dev[nt][:, idx] = torch.tensor(
                            clique_fut_lane_dev[i][j]
                        ).to(self.device)
                        batch_lane[nt][:, idx] = torch.tensor(clique_lane[i][j]).to(
                            self.device
                        )
                        batch_lane_st[nt][:, idx] = self.rel_state_fun[nt](
                            batch_lane[nt][:, idx], x0.repeat(ft, 1)
                        )

                        batch_node_size[nt][idx] = torch.tensor(
                            clique_node_size[i][j]
                        ).to(self.device)

                        last_timestep = clique_last_timestep[i][j]
                        batch_state_future[nt][:, idx] = torch.tensor(
                            clique_future_state[i][j]
                        ).to(self.device)
                        batch_state_future_st[nt][
                            :last_timestep, idx
                        ] = self.rel_state_fun[nt](
                            batch_state_future[nt][:last_timestep, idx],
                            x0.repeat(last_timestep, 1),
                        )
                        batch_first_timestep[nt][idx] = first_timestep
                        batch_last_timestep[nt][idx] = last_timestep
                        batch_map[nt][idx] = clique_map[i][j].clone().to(self.device)
                        batch_is_robot[nt][idx] = clique_is_robot[i][j]
                else:
                    for idx, (i, j) in node_index[nt].items():
                        batch_state_history[nt][:, idx] = torch.tensor(
                            clique_state_history[i][j]
                        ).to(self.device)
                        first_timestep = clique_first_timestep[i][j]
                        x0 = batch_state_history[nt][-1, idx]
                        batch_state_history_st[nt][
                            first_timestep:, idx
                        ] = self.rel_state_fun[nt](
                            batch_state_history[nt][first_timestep:, idx],
                            x0.repeat(ht + 1 - first_timestep, 1),
                        )
                        batch_node_size[nt][idx] = torch.tensor(
                            clique_node_size[i][j]
                        ).to(self.device)
                        last_timestep = clique_last_timestep[i][j]
                        batch_state_future[nt][:, idx] = torch.tensor(
                            clique_future_state[i][j]
                        ).to(self.device)
                        batch_state_future_st[nt][
                            :last_timestep, idx
                        ] = self.rel_state_fun[nt](
                            batch_state_future[nt][:last_timestep, idx],
                            x0.repeat(last_timestep, 1),
                        )
                        batch_first_timestep[nt][idx] = first_timestep
                        batch_last_timestep[nt][idx] = last_timestep
                        batch_is_robot[nt][idx] = clique_is_robot[i][j]

        else:
            batch_state_future = None
            batch_state_future_st = None
            batch_fut_lane_dev = None
            batch_last_timestep = None

            for nt in self.node_types:
                if (
                    nt in self.hyperparams["map_encoder"]
                    and self.hyperparams["use_map_encoding"]
                ):
                    for idx, (i, j) in node_index[nt].items():
                        batch_state_history[nt][:, idx] = torch.tensor(
                            clique_state_history[i][j]
                        ).to(self.device)
                        first_timestep = clique_first_timestep[i][j]
                        x0 = batch_state_history[nt][-1, idx]
                        batch_state_history_st[nt][
                            first_timestep:, idx
                        ] = self.rel_state_fun[nt](
                            batch_state_history[nt][first_timestep:, idx],
                            x0.repeat(ht + 1 - first_timestep, 1),
                        )
                        # batch_lane_dev[nt][:,idx] = torch.tensor(np.clip(clique_lane_dev[i][j],[-3.0,-12],[3.0,12])).to(self.device)
                        batch_lane_dev[nt][:, idx] = torch.tensor(
                            clique_lane_dev[i][j]
                        ).to(self.device)
                        batch_lane[nt][:, idx] = torch.tensor(clique_lane[i][j]).to(
                            self.device
                        )
                        batch_lane_st[nt][:, idx] = self.rel_state_fun[nt](
                            batch_lane[nt][:, idx], x0.repeat(ft, 1)
                        )

                        batch_node_size[nt][idx] = torch.tensor(
                            clique_node_size[i][j]
                        ).to(self.device)
                        batch_first_timestep[nt][idx] = first_timestep
                        batch_map[nt][idx] = clique_map[i][j].clone().to(self.device)
                        batch_is_robot[nt][idx] = clique_is_robot[i][j]
                else:
                    for idx, (i, j) in node_index[nt].items():
                        batch_state_history[nt][:, idx] = torch.tensor(
                            clique_state_history[i][j]
                        ).to(self.device)
                        first_timestep = clique_first_timestep[i][j]
                        x0 = batch_state_history[nt][-1, idx]
                        batch_state_history_st[nt][
                            first_timestep:, idx
                        ] = self.rel_state_fun[nt](
                            batch_state_history[nt][first_timestep:, idx],
                            x0.repeat(ht + 1 - first_timestep, 1),
                        )
                        batch_node_size[nt][idx] = torch.tensor(
                            clique_node_size[i][j]
                        ).to(self.device)
                        batch_first_timestep[nt][idx] = first_timestep
                        batch_is_robot[nt][idx] = clique_is_robot[i][j]

        batch_edge_idx1 = {et: list() for et in self.edge_types}
        batch_edge_idx2 = {et: list() for et in self.edge_types}
        for et in self.edge_types:
            for idx, (idxj, idxk) in batch_node_to_edge_index[et].items():
                batch_edge_idx1[et].append(idxj)
                batch_edge_idx2[et].append(idxk)

        batch_edge = dict()
        batch_edge_first_timestep = dict()

        for edge_type in self.edge_types:
            batch_edge[edge_type] = torch.zeros(
                [
                    ht + 1,
                    len(edge_index[edge_type]),
                    self.state_length[edge_type[0]] + self.state_length[edge_type[1]],
                ]
            ).to(self.device)
            batch_edge_first_timestep[edge_type] = torch.zeros(
                len(edge_index[edge_type]), dtype=torch.int
            ).to(self.device)
            dim1 = self.state_length[edge_type[0]]
            dim2 = self.state_length[edge_type[1]]
            for idx, (i, j, k) in edge_index[edge_type].items():
                batch_edge[edge_type][:, idx, 0:dim1] = torch.tensor(
                    clique_state_history[i][j]
                ).to(self.device)
                batch_edge[edge_type][:, idx, dim1 : dim1 + dim2] = torch.tensor(
                    clique_state_history[i][k]
                ).to(self.device)
                batch_edge_first_timestep[edge_type][idx] = max(
                    clique_first_timestep[i][j], clique_first_timestep[i][k]
                )

        return (
            batch_state_history,
            batch_state_history_st,
            batch_state_future,
            batch_state_future_st,
            batch_edge,
            batch_first_timestep,
            batch_last_timestep,
            batch_edge_first_timestep,
            batch_map,
            batch_node_size,
            batch_is_robot,
            batch_lane,
            batch_lane_st,
            batch_lane_dev,
            batch_fut_lane_dev,
            (
                node_index,
                edge_index,
                node_inverse_index,
                batch_node_to_edge_index,
                edge_to_node_index,
                batch_edge_idx1,
                batch_edge_idx2,
            ),
        )

    def encode_node_history(
        self, mode, batch_state_history_st, batch_lane_dev, batch_first_timestep
    ):
        """
        Encodes the nodes history.

        """
        batch_node_hist_enc = dict()

        for node_type in self.node_types:
            if batch_state_history_st[node_type].nelement() > 0:
                if (
                    self.hyperparams["use_map_encoding"]
                    and node_type in self.hyperparams["map_encoder"]
                    and self.hyperparams["use_lane_info"]
                ):
                    pre_encoded_vec = self.node_modules[
                        node_type + "/node_pre_encoder"
                    ](
                        torch.cat(
                            (
                                batch_state_history_st[node_type],
                                batch_lane_dev[node_type],
                            ),
                            dim=-1,
                        )
                    )
                else:
                    pre_encoded_vec = self.node_modules[
                        node_type + "/node_pre_encoder"
                    ](batch_state_history_st[node_type])

                batch_node_hist_enc[node_type], _ = run_lstm_on_variable_length_seqs(
                    self.node_modules[node_type + "/node_history_encoder"],
                    original_seqs=pre_encoded_vec,
                    lower_indices=batch_first_timestep[node_type],
                    batch_first=False,
                )

                batch_node_hist_enc[node_type] = F.dropout(
                    batch_node_hist_enc[node_type],
                    p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                    training=(mode == ModeKeys.TRAIN),
                )  # [bs, max_time, enc_rnn_dim]
            else:
                batch_node_hist_enc[node_type] = torch.zeros(
                    [
                        batch_state_history_st[node_type].shape[0],
                        0,
                        self.hyperparams["enc_rnn_dim_history"],
                    ]
                ).to(self.device)

        return batch_node_hist_enc

    def encode_edge(
        self, mode, batch_edge, batch_edge_first_timestep, batch_node_size, indices
    ):
        """
        Encode edges history
        """
        (
            _,
            _,
            _,
            _,
            _,
            batch_edge_idx1,
            batch_edge_idx2,
        ) = indices
        batch_edge_hist_enc = dict()
        for et in self.edge_types:
            if batch_edge[et].nelement() == 0:
                batch_edge_hist_enc[et] = None
            else:
                pre_enc_net = self.node_modules[
                    str(et[0]) + "->" + str(et[1]) + "/edge_pre_encoding"
                ]

                dim1 = self.state_length[et[0]]
                dim2 = self.state_length[et[1]]
                ht = self.hyperparams["maximum_history_length"]

                edge_pre_encode = pre_enc_net(
                    batch_edge[et][..., 0:dim1],
                    batch_edge[et][..., dim1 : dim1 + dim2],
                    batch_node_size[et[0]][batch_edge_idx1[et]].repeat(ht + 1, 1, 1),
                    batch_node_size[et[1]][batch_edge_idx2[et]].repeat(ht + 1, 1, 1),
                )
                batch_edge_hist_enc[et], _ = run_lstm_on_variable_length_seqs(
                    self.node_modules[str(et[0]) + "->" + str(et[1]) + "/edge_encoder"],
                    original_seqs=edge_pre_encode,
                    lower_indices=batch_edge_first_timestep[et],
                    batch_first=False,
                )

                batch_edge_hist_enc[et] = F.dropout(
                    batch_edge_hist_enc[et],
                    p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                    training=(mode == ModeKeys.TRAIN),
                )
        return batch_edge_hist_enc

    def encode_node_future(
        self,
        mode,
        batch_state_history_st,
        batch_lane_dev,
        batch_state_future_st,
        batch_fut_lane_dev,
        batch_last_timestep,
    ) -> dict:
        """
        Encodes the node future (during training) using a bi-directional LSTM
        """
        encoded_future = dict()
        for node_type in self.node_types:
            if (
                self.hyperparams["use_map_encoding"]
                and node_type in self.hyperparams["map_encoder"]
                and self.hyperparams["use_lane_info"]
            ):
                encoded_history_vec = self.node_modules[
                    node_type + "/node_pre_encoder"
                ](
                    torch.cat(
                        (
                            batch_state_history_st[node_type][-1],
                            batch_lane_dev[node_type][-1],
                        ),
                        dim=-1,
                    )
                )
            else:
                encoded_history_vec = self.node_modules[
                    node_type + "/node_pre_encoder"
                ](batch_state_history_st[node_type][-1])

            initial_h_model = self.node_modules[
                node_type + "/node_future_encoder/initial_h"
            ]
            initial_c_model = self.node_modules[
                node_type + "/node_future_encoder/initial_c"
            ]

            # Here we're initializing the forward hidden states,
            # but zeroing the backward ones.
            initial_h = initial_h_model(encoded_history_vec)
            initial_h = torch.stack(
                [initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0
            )
            initial_c = initial_c_model(encoded_history_vec)
            initial_c = torch.stack(
                [initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0
            )

            initial_state = (initial_h, initial_c)
            if (
                self.hyperparams["use_map_encoding"]
                and node_type in self.hyperparams["map_encoder"]
                and self.hyperparams["use_lane_info"]
            ):
                encoded_future_vec = self.node_modules[node_type + "/node_pre_encoder"](
                    torch.cat(
                        (
                            batch_state_future_st[node_type],
                            batch_fut_lane_dev[node_type],
                        ),
                        dim=-1,
                    )
                )
            else:
                encoded_future_vec = self.node_modules[node_type + "/node_pre_encoder"](
                    batch_state_future_st[node_type]
                )

            if encoded_future_vec.nelement() > 0:
                _, state = run_lstm_on_variable_length_seqs(
                    self.node_modules[node_type + "/node_future_encoder"],
                    original_seqs=encoded_future_vec,
                    upper_indices=batch_last_timestep[node_type],
                    batch_first=False,
                )
                state = unpack_RNN_state(state)
                encoded_future[node_type] = F.dropout(
                    state,
                    p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                    training=(mode == ModeKeys.TRAIN),
                )
            else:
                encoded_future[node_type] = None

        return encoded_future

    def encoder(
        self,
        mode,
        clique_type,
        node_history_encoded,
        batch_edge_hist_enc,
        encoded_map,
        clique_is_robot,
        node_future_encoded,
        indices,
        num_samples=None,
    ):
        """
        Encoder of the CVAE.

        """
        (
            node_index,
            edge_index,
            node_inverse_index,
            batch_node_to_edge_index,
            edge_to_node_index,
            batch_edge_idx1,
            batch_edge_idx2,
        ) = indices
        Nclique = len(clique_type)
        Nnodes = {
            node_type: node_history_encoded[node_type].shape[1]
            for node_type in self.node_types
        }
        # since the Gibbs distribution is on cliques, we need to put the encoded features into cliques
        clique_state_enc = [None] * Nclique
        clique_edge_enc = [None] * Nclique
        # joint latent cardinality for each node
        z_num = {
            node_type: torch.zeros(Nnodes[node_type], dtype=torch.int).to(self.device)
            for node_type in self.node_types
        }
        nsample_greedy = self.hyperparams["max_greedy_sample"]
        nsample_random = self.hyperparams["max_random_sample"]
        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            kl_obj = torch.tensor(0.0, dtype=torch.float).to(self.device)
        else:
            kl_obj = None
        if mode == ModeKeys.PREDICT:
            if num_samples is None:
                num_samples = self.hyperparams["pred_num_samples"]

            # probability of modes
            pi_list = {
                nt: torch.zeros([Nnodes[nt], num_samples]).to(self.device)
                for nt in self.node_types
            }
            # joint latent variable list
            z_list = {
                nt: torch.zeros([Nnodes[nt], num_samples], dtype=torch.long).to(
                    self.device
                )
                for nt in self.node_types
            }
            for i in range(Nclique):
                clique_state_enc[i] = list()
                clique_edge_enc[i] = dict()

            for (i, j), (nt, idx) in node_inverse_index.items():
                if (
                    self.hyperparams["use_map_encoding"]
                    and nt in self.hyperparams["map_encoder"]
                ):
                    clique_state_enc[i].append(
                        torch.cat(
                            (node_history_encoded[nt][-1, idx], encoded_map[nt][idx]),
                            dim=-1,
                        )
                    )
                else:
                    clique_state_enc[i].append(node_history_encoded[nt][-1, idx])
        else:
            clique_state_future_enc = [None] * Nclique
            if mode == ModeKeys.TRAIN:
                z_list = {
                    nt: torch.zeros(
                        [Nnodes[nt], nsample_greedy + nsample_random], dtype=torch.long
                    ).to(self.device)
                    for nt in self.node_types
                }
                pi_list = {
                    nt: torch.zeros([Nnodes[nt], nsample_greedy + nsample_random]).to(
                        self.device
                    )
                    for nt in self.node_types
                }
            elif mode == ModeKeys.EVAL:
                if num_samples is None:
                    num_samples = self.hyperparams["eval_num_samples"]
                z_list = {
                    nt: torch.zeros([Nnodes[nt], num_samples], dtype=torch.long).to(
                        self.device
                    )
                    for nt in self.node_types
                }
                pi_list = {
                    nt: torch.zeros([Nnodes[nt], num_samples]).to(self.device)
                    for nt in self.node_types
                }
            # prepare encoded tensors for Gibbs distr
            for i in range(Nclique):
                clique_state_enc[i] = list()
                clique_state_future_enc[i] = list()
                clique_edge_enc[i] = dict()

            for (i, j), (nt, idx) in node_inverse_index.items():
                if (
                    self.hyperparams["use_map_encoding"]
                    and nt in self.hyperparams["map_encoder"]
                ):
                    clique_state_enc[i].append(
                        torch.cat(
                            (node_history_encoded[nt][-1, idx], encoded_map[nt][idx]),
                            dim=-1,
                        )
                    )
                else:
                    clique_state_enc[i].append(node_history_encoded[nt][-1, idx])

                clique_state_future_enc[i].append(node_future_encoded[nt][idx])

        for et in self.edge_types:
            for idx, (i, j, k) in edge_index[et].items():
                clique_edge_enc[i][(j, k)] = batch_edge_hist_enc[et][-1, idx]

        clique_zlist = dict()
        clique_zlist[0] = torch.tensor(0, dtype=torch.long)
        clique_zlist[1] = (
            torch.arange(self.z_dim, dtype=torch.long).view(-1, 1).to(self.device)
        )
        for n in range(2, self.max_Nnode + 1):
            zzlist = list(itertools.product(range(0, self.z_dim), repeat=n))
            clique_zlist[n] = (
                torch.tensor(zzlist, dtype=torch.long).view(-1, n).to(self.device)
            )

        for i in range(Nclique):
            Nnode_i = len(clique_type[i])

            if self.hyperparams["incl_robot_node"] and any(clique_is_robot[i]):
                log_pis_p = self.node_modules["p_z_x"](
                    clique_type[i],
                    clique_state_enc[i],
                    clique_edge_enc[i],
                    clique_is_robot[i],
                ).flatten()
            else:
                log_pis_p = self.node_modules["p_z_x"](
                    clique_type[i], clique_state_enc[i], clique_edge_enc[i]
                ).flatten()

            log_pis_p = torch.clamp(log_pis_p, min=self.hyperparams["log_pi_clamp"])
            # z_sel: greedily selected latent
            # z_close: modes with large pi but passed due to lack of diversity score, may be selected if there is no latent with better diversity score.
            z_sel = np.zeros(log_pis_p.shape[0], dtype=np.bool)
            z_close = np.zeros(log_pis_p.shape[0], dtype=np.bool)
            if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
                state_fut_enc = list()
                for j in range(len(clique_state_enc[i])):
                    state_fut_enc.append(
                        torch.cat(
                            (clique_state_enc[i][j], clique_state_future_enc[i][j]),
                            dim=-1,
                        )
                    )

                if self.hyperparams["incl_robot_node"] and any(clique_is_robot[i]):
                    log_pis_q = self.node_modules["q_z_xy"](
                        clique_type[i],
                        state_fut_enc,
                        clique_edge_enc[i],
                        clique_is_robot[i],
                    ).flatten()
                else:
                    log_pis_q = self.node_modules["q_z_xy"](
                        clique_type[i], state_fut_enc, clique_edge_enc[i]
                    ).flatten()
                log_pis_q = torch.clamp(log_pis_q, min=self.hyperparams["log_pi_clamp"])
                kl_clique = torch.clamp(discrete_KL_div(log_pis_p, log_pis_q), min=0.0)
                if kl_clique.isnan() or kl_clique.isinf():
                    pdb.set_trace()
                if not kl_clique.isnan() and not kl_clique.isinf():
                    kl_obj += kl_clique
                pis = torch.exp(log_pis_q)
            else:
                pis = torch.exp(log_pis_p)
            if mode == ModeKeys.PREDICT:
                if log_pis_p.shape[0] > num_samples:
                    idx = (
                        torch.argsort(log_pis_p, descending=True).detach().cpu().numpy()
                    )
                    z_idx = list()
                    close_z_idx = list()
                    k = 0
                    if self.hyperparams["incl_robot_node"]:
                        N_non = Nnode_i - sum(clique_is_robot[i])
                    else:
                        N_non = Nnode_i
                    while len(z_idx) < num_samples and k < idx.shape[0]:
                        if z_close[idx[k]] == False:
                            z_idx.append(idx[k])
                            z_sel[idx[k]] = True
                            z_close = (
                                z_close
                                | (
                                    torch.sum(
                                        clique_zlist[N_non]
                                        != clique_zlist[N_non][idx[k]],
                                        dim=-1,
                                    )
                                    <= 1
                                )
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        else:
                            close_z_idx.append(idx[k])
                        k += 1
                    if len(z_idx) == num_samples:
                        z_idx = np.array(z_idx)
                    else:
                        z_idx = np.array(
                            z_idx + close_z_idx[: (num_samples - len(z_idx))]
                        )

                    # z_idx = idx[0:num_samples]

                else:
                    z_idx = np.arange(log_pis_p.shape[0])
            elif mode == ModeKeys.TRAIN:
                if log_pis_p.shape[0] > nsample_greedy + nsample_random:
                    idx = (
                        torch.argsort(log_pis_p, descending=True).detach().cpu().numpy()
                    )
                    # z_idx_greedy = idx[0:nsample_greedy]
                    # z_idx_rand = np.random.choice(idx[nsample_greedy:],nsample_random)
                    z_idx_greedy = list()
                    close_z_idx = list()
                    k = 0
                    if self.hyperparams["incl_robot_node"]:
                        N_non = Nnode_i - sum(clique_is_robot[i])
                    else:
                        N_non = Nnode_i
                    # greedy algorithm to select diverse latent
                    while len(z_idx_greedy) < nsample_greedy and k < idx.shape[0]:
                        if z_close[idx[k]] == False:
                            z_idx_greedy.append(idx[k])
                            z_sel[idx[k]] = True
                            z_close = (
                                z_close
                                | (
                                    torch.sum(
                                        clique_zlist[N_non]
                                        != clique_zlist[N_non][idx[k]],
                                        dim=-1,
                                    )
                                    <= 1
                                )
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        else:
                            close_z_idx.append(idx[k])
                        k += 1
                    if len(z_idx_greedy) == nsample_greedy:
                        z_idx_greedy = np.array(z_idx_greedy)
                    else:
                        z_sel[
                            close_z_idx[: (nsample_greedy - len(z_idx_greedy))]
                        ] = True
                        z_idx_greedy = np.array(
                            z_idx_greedy
                            + close_z_idx[: (nsample_greedy - len(z_idx_greedy))]
                        )

                    remain_idx = np.where(z_sel == False)[0]
                    z_idx_rand = np.random.choice(remain_idx, nsample_random)

                    z_idx = np.append(z_idx_greedy, z_idx_rand)
                else:
                    z_idx = np.arange(log_pis_p.shape[0])
            elif mode == ModeKeys.EVAL:
                if num_samples is None:
                    num_samples = self.hyperparams["eval_num_samples"]
                if log_pis_p.shape[0] > num_samples:
                    idx = (
                        torch.argsort(log_pis_p, descending=True).detach().cpu().numpy()
                    )
                    z_idx = list()
                    close_z_idx = list()
                    k = 0
                    if self.hyperparams["incl_robot_node"]:
                        N_non = Nnode_i - sum(clique_is_robot[i])
                    else:
                        N_non = Nnode_i
                    while len(z_idx) < num_samples and k < idx.shape[0]:
                        if z_close[idx[k]] == False:
                            z_idx.append(idx[k])
                            z_sel[idx[k]] = True

                            z_close = (
                                z_close
                                | (
                                    torch.sum(
                                        clique_zlist[N_non]
                                        != clique_zlist[N_non][idx[k]],
                                        dim=-1,
                                    )
                                    <= 1
                                )
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        else:
                            close_z_idx.append(idx[k])
                        k += 1
                    if len(z_idx) == num_samples:
                        z_idx = np.array(z_idx)
                    else:
                        z_idx = np.array(
                            z_idx + close_z_idx[: (num_samples - len(z_idx))]
                        )
                else:
                    z_idx = np.arange(log_pis_p.shape[0])

            if self.hyperparams["incl_robot_node"] and any(clique_is_robot[i]):
                clique_z = clique_zlist[Nnode_i - sum(clique_is_robot[i])]
                idx = 0
                for j in range(Nnode_i):
                    nt, batch_idx = node_inverse_index[(i, j)]
                    if clique_is_robot[i][j]:
                        z_list[nt][batch_idx][0 : z_idx.shape[0]] = 0
                    else:
                        z_list[nt][batch_idx][0 : z_idx.shape[0]] = clique_z[z_idx, idx]
                        idx += 1
                    pi_list[nt][batch_idx][0 : z_idx.shape[0]] = torch.clone(
                        pis[z_idx] / torch.sum(pis[z_idx])
                    )
                    z_num[nt][batch_idx] = z_idx.shape[0]
            else:
                for j in range(Nnode_i):
                    nt, batch_idx = node_inverse_index[(i, j)]

                    z_list[nt][batch_idx][0 : z_idx.shape[0]] = clique_zlist[Nnode_i][
                        z_idx, j
                    ]

                    pi_list[nt][batch_idx][0 : z_idx.shape[0]] = torch.clone(
                        pis[z_idx] / torch.sum(pis[z_idx])
                    )
                    z_num[nt][batch_idx] = z_idx.shape[0]

        if mode == ModeKeys.TRAIN:
            if self.log_writer is not None:
                self.log_writer.add_scalar("%s" % ("kl"), kl_obj, self.curr_iter)
        else:
            kl_obj = None

        return z_list, pi_list, z_num, kl_obj

    def decoder(
        self,
        mode,
        batch_state_history,
        batch_state_history_st,
        node_history_encoded,
        encoded_map,
        batch_node_size,
        batch_lane_st,
        robot_traj,
        indices,
        z_list,
        z_num,
        ft,
    ):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.

        """
        state_pred_res = list()
        state_pred_res_st = list()
        input_pred_res = list()
        ref_traj = list()
        max_z_num = 0

        for nt in self.node_types:
            if z_num[nt].shape[0] > 0:
                max_z_num = max(int(torch.max(z_num[nt])), max_z_num)
        model = self.node_modules["policy_net"]
        # turn the latent to one-hot
        for i in range(max_z_num):
            batch_z = {
                nt: to_one_hot(z_list[nt][:, i], self.z_dim).to(self.device)
                for nt in self.node_types
            }

            (
                batch_state_pred,
                batch_state_pred_st,
                batch_input_pred,
                traj_des,
                tracking_error,
                collision_cost,
            ) = model(
                batch_state_history=batch_state_history,
                batch_state_history_st=batch_state_history_st,
                node_history_encoded=node_history_encoded,
                encoded_map=encoded_map,
                batch_node_size=batch_node_size,
                batch_lane_st=batch_lane_st,
                indices=indices,
                ft=ft,
                batch_z=batch_z,
                robot_traj=robot_traj,
            )
            state_pred_res.append(batch_state_pred)
            state_pred_res_st.append(batch_state_pred_st)
            input_pred_res.append(batch_input_pred)
            ref_traj.append(traj_des)

        return (
            state_pred_res,
            state_pred_res_st,
            input_pred_res,
            ref_traj,
            tracking_error,
            collision_cost,
        )

    def forward(self, **kwargs):
        return self.train_loss(**kwargs)

    def calc_traj_matching(
        self,
        mode,
        state_pred_res_st,
        batch_state_future_st,
        ref_traj,
        batch_last_timestep,
        pi_list,
        z_num,
    ):
        loss = torch.tensor(0.0).to(self.device)
        mask = dict()
        for nt in self.node_types:
            mask[nt] = torch.zeros_like(batch_state_future_st[nt]).to(self.device)
            for i in range(batch_state_future_st[nt].shape[1]):
                mask[nt][: batch_last_timestep[nt][i], i] = 1

            N_z = int(torch.max(z_num[nt]))
            Nnode = batch_state_future_st[nt].shape[1]
            loss_mat = torch.zeros_like(pi_list[nt], dtype=torch.float).to(self.device)
            if mode == ModeKeys.TRAIN:
                for i in range(N_z):
                    loss_mat[:, i] += torch.linalg.norm(
                        mask[nt][..., 0:2]
                        * (
                            state_pred_res_st[i][nt][..., 0:2]
                            - batch_state_future_st[nt][..., 0:2]
                        ),
                        dim=[0, 2],
                    )
                    loss_mat[:, i] += self.ref_match_weight * torch.linalg.norm(
                        mask[nt][..., 0:2]
                        * (ref_traj[i][nt] - batch_state_future_st[nt][..., 0:2]),
                        dim=[0, 2],
                    )
                gamma = self.gamma
                # implement CVaR cost (gamma here meant alpha in the paper)
                if gamma < 0.7:
                    for j in range(loss_mat.shape[0]):
                        pi_altered = CVaR_weight(
                            loss_mat[j],
                            pi_list[nt][j],
                            gamma,
                            sign=1,
                            end_idx=z_num[nt][j],
                        )
                        loss += (torch.sum(loss_mat[j] * pi_altered)) / Nnode
                else:
                    for j in range(loss_mat.shape[0]):
                        pi_altered = CVaR_weight(
                            loss_mat[j],
                            pi_list[nt][j],
                            gamma,
                            sign=1,
                            end_idx=z_num[nt][j],
                        )
                        max_idx = torch.argmax(pi_list[nt][j])
                        new_loss = (
                            torch.sum(
                                loss_mat[j][0:max_idx].detach() * pi_altered[0:max_idx]
                            )
                            + loss_mat[j][max_idx] * pi_altered[max_idx]
                            + torch.sum(
                                loss_mat[j][max_idx + 1 :].detach()
                                * pi_altered[max_idx + 1 :]
                            )
                        )
                        loss += new_loss / Nnode

            else:
                for i in range(N_z):
                    loss_mat[:, i] += torch.linalg.norm(
                        mask[nt][..., 0:2]
                        * (
                            state_pred_res_st[i][nt][..., 0:2]
                            - batch_state_future_st[nt][..., 0:2]
                        ),
                        dim=[0, 2],
                    )
                loss += torch.sum(loss_mat * pi_list[nt]) / Nnode

            # loss+=torch.sum(loss_mat*pi_list[nt])/Nnode
        return loss

    def traj_collision_count(
        self, batch_state_pred, batch_node_size, indices, pi_list, mode
    ):

        (
            node_index,
            edge_index,
            node_inverse_index,
            batch_node_to_edge_index,
            edge_to_node_index,
            batch_edge_idx1,
            batch_edge_idx2,
        ) = indices
        ft = self.hyperparams["prediction_horizon"]
        coll_score = {nt: torch.zeros(ft).to(self.device) for nt in self.node_types}

        for et in self.edge_types:
            edge_node_size1 = torch.unsqueeze(
                batch_node_size[et[0]][batch_edge_idx1[et]], dim=0
            ).repeat(ft, 1, 1)
            edge_node_size2 = torch.unsqueeze(
                batch_node_size[et[1]][batch_edge_idx2[et]], dim=0
            ).repeat(ft, 1, 1)
            pis = pi_list[et[0]][batch_edge_idx1[et]]

            for i in range(len(batch_state_pred)):

                traj1 = batch_state_pred[i][et[0]][:, batch_edge_idx1[et]]
                traj2 = batch_state_pred[i][et[1]][:, batch_edge_idx2[et]]
                # mode 0: calculate collision loss (same as training)
                # mode 1: calculate collision rate (for evaluation)
                if traj1.nelement() > 0:
                    if mode == 0:
                        coll_score[et[0]] += torch.sum(
                            self.collision_fun[et](
                                traj1, traj2, edge_node_size1, edge_node_size2
                            )
                            * pis[:, i],
                            dim=-1,
                        )
                    elif mode == 1:

                        edge_col = (
                            self.collision_fun[et](
                                traj1[0:ft],
                                traj2[0:ft],
                                edge_node_size1,
                                edge_node_size2,
                                return_dis=True,
                            )
                            < 0
                        )
                        node_col = torch.zeros(
                            ft,
                            (len(node_index[et[0]])),
                        ).to(self.device)
                        for idx, (j, k) in edge_to_node_index[et].items():
                            node_col[:, j] = torch.logical_or(
                                node_col[:, j], edge_col[:, idx]
                            )
                        # for t in range(1,ft):
                        #     node_col[t] = torch.logical_or(node_col[t], node_col[t-1])
                        coll_score[et[0]] += torch.sum(
                            node_col * pi_list[et[0]][:, i], dim=-1
                        )

        return coll_score

    def eval_traj_matching(
        self,
        state_pred_res_st,
        batch_state_future_st,
        ref_traj,
        batch_first_timestep,
        batch_last_timestep,
        z_num,
        criterion=0,
    ):
        # calcualte ADE/FDE for evaluation
        ft = self.hyperparams["prediction_horizon"]
        ht = self.hyperparams["maximum_history_length"]
        ADE = {nt: torch.tensor(0.0).to(self.device) for nt in self.node_types}
        FDE = {nt: torch.zeros(ft).to(self.device) for nt in self.node_types}
        FDE_count = {nt: torch.zeros([ft]).to(self.device) for nt in self.node_types}
        Nnode = dict()
        ADE_count = {nt: 0 for nt in self.node_types}
        mask = dict()
        for nt in self.node_types:
            mask[nt] = torch.zeros_like(batch_state_future_st[nt]).to(self.device)
            for i in range(batch_state_future_st[nt].shape[1]):
                mask[nt][: batch_last_timestep[nt][i], i] = 1

            N_z = int(torch.max(z_num[nt]))
            Nnode[nt] = batch_state_future_st[nt].shape[1]
            ADE_mat = torch.zeros([Nnode[nt], N_z], dtype=torch.float).to(self.device)
            FDE_mat = torch.zeros([Nnode[nt], ft, N_z], dtype=torch.float).to(
                self.device
            )
            for i in range(N_z):
                ADE_mat[:, i] += (
                    torch.linalg.norm(
                        mask[nt][..., 0:2]
                        * (
                            state_pred_res_st[i][nt][..., 0:2]
                            - batch_state_future_st[nt][..., 0:2]
                        ),
                        dim=[0, 2],
                    )
                    / batch_last_timestep[nt].to(self.device)
                )
                for j in range(Nnode[nt]):
                    FDE_mat[
                        j, : batch_last_timestep[nt][j].long(), i
                    ] = torch.linalg.norm(
                        (
                            state_pred_res_st[i][nt][
                                : batch_last_timestep[nt][j].long(), j, 0:2
                            ]
                            - batch_state_future_st[nt][
                                : batch_last_timestep[nt][j].long(), j, 0:2
                            ]
                        ),
                        dim=-1,
                    )

            for i in range(Nnode[nt]):
                if (
                    (criterion == 0)
                    or (
                        criterion == 1
                        and batch_first_timestep[nt][i] == 0
                        and batch_last_timestep[nt][i] == ft
                    )
                    or (criterion == 2 and batch_first_timestep[nt][i] < ht / 2)
                ):

                    ADE_count[nt] += 1
                    FDE_count[nt][: batch_last_timestep[nt][i].long()] += 1
                    ADE[nt] += torch.min(ADE_mat[i])
                    FDE[nt] += torch.min(FDE_mat[i], dim=-1)[0]

            # loss+=torch.sum(loss_mat*pi_list[nt])/Nnode

        return ADE, FDE, ADE_count, FDE_count

    def train_loss(
        self,
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
    ) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :return: Scalar tensor -> nll loss
        """
        ft = clique_future_state[0][0].shape[0]
        mode = ModeKeys.TRAIN

        (
            batch_state_history,
            batch_state_history_st,
            batch_state_future,
            batch_state_future_st,
            batch_edge,
            batch_first_timestep,
            batch_last_timestep,
            batch_edge_first_timestep,
            batch_map,
            batch_node_size,
            batch_is_robot,
            batch_lane,
            batch_lane_st,
            batch_lane_dev,
            batch_fut_lane_dev,
            indices,
        ) = self.batch_clique(
            mode,
            clique_type,
            clique_state_history,
            clique_first_timestep,
            clique_last_timestep,
            clique_edges,
            clique_map,
            clique_node_size,
            clique_is_robot,
            clique_lane,
            clique_lane_dev,
            clique_future_state,
            clique_fut_lane_dev,
        )

        (
            node_history_encoded,
            node_future_encoded,
            batch_edge_hist_enc,
            encoded_map,
        ) = self.obtain_encoded_tensors(
            mode=mode,
            batch_state_history=batch_state_history,
            batch_state_history_st=batch_state_history_st,
            batch_state_future=batch_state_future,
            batch_state_future_st=batch_state_future_st,
            batch_edge=batch_edge,
            batch_first_timestep=batch_first_timestep,
            batch_last_timestep=batch_last_timestep,
            batch_edge_first_timestep=batch_edge_first_timestep,
            batch_map=batch_map,
            batch_node_size=batch_node_size,
            batch_lane_dev=batch_lane_dev,
            batch_fut_lane_dev=batch_fut_lane_dev,
            indices=indices,
        )
        z_list, pi_list, z_num, kl = self.encoder(
            mode,
            clique_type,
            node_history_encoded,
            batch_edge_hist_enc,
            encoded_map,
            clique_is_robot,
            node_future_encoded,
            indices,
        )
        if self.hyperparams["incl_robot_node"]:
            robot_traj = dict()
            for nt in self.node_types:
                robot_traj[nt] = dict()
                idx = torch.where(batch_is_robot[nt])[0]
                for k in range(idx.shape[0]):
                    robot_traj[nt][int(idx[k])] = (
                        batch_state_future[nt][:, int(idx[k])],
                        batch_state_future_st[nt][:, int(idx[k])],
                    )
        else:
            robot_traj = None
        (
            state_pred_res,
            state_pred_res_st,
            input_pred_res,
            ref_traj,
            tracking_error,
            collision_cost,
        ) = self.decoder(
            mode,
            batch_state_history,
            batch_state_history_st,
            node_history_encoded,
            encoded_map,
            batch_node_size,
            batch_lane_st,
            robot_traj,
            indices,
            z_list,
            z_num,
            ft,
        )

        matching_loss = self.calc_traj_matching(
            mode,
            state_pred_res_st,
            batch_state_future_st,
            ref_traj,
            batch_last_timestep,
            pi_list,
            z_num,
        )

        loss = (
            matching_loss
            + self.kl_weight * kl
            + tracking_error
            + self.collision_weight * collision_cost
        )

        if self.log_writer is not None:
            # self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_q'),
            #                            mutual_inf_q,
            #                            self.curr_iter)
            # self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_p'),
            #                            mutual_inf_p,
            #                            self.curr_iter)
            # self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'log_likelihood'),
            #                            log_likelihood,
            #                            self.curr_iter)
            self.log_writer.add_scalar("%s" % ("loss"), loss, self.curr_iter)
            # if self.hyperparams['log_histograms']:
            #     self.latent.summarize_for_tensorboard(self.log_writer, str(self.node_type), self.curr_iter)
        if loss.isnan() or loss.isinf():
            pdb.set_trace()
        return loss

    def eval_loss(
        self,
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
        num_samples=None,
        criterion=0,
    ) -> torch.Tensor:
        """
        Calculates the evaluation loss for a batch.


        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL
        (
            batch_state_history,
            batch_state_history_st,
            batch_state_future,
            batch_state_future_st,
            batch_edge,
            batch_first_timestep,
            batch_last_timestep,
            batch_edge_first_timestep,
            batch_map,
            batch_node_size,
            batch_is_robot,
            batch_lane,
            batch_lane_st,
            batch_lane_dev,
            batch_fut_lane_dev,
            indices,
        ) = self.batch_clique(
            mode,
            clique_type,
            clique_state_history,
            clique_first_timestep,
            clique_last_timestep,
            clique_edges,
            clique_map,
            clique_node_size,
            clique_is_robot,
            clique_lane,
            clique_lane_dev,
            clique_future_state,
            clique_fut_lane_dev,
        )

        (
            node_history_encoded,
            node_future_encoded,
            batch_edge_hist_enc,
            encoded_map,
        ) = self.obtain_encoded_tensors(
            mode=mode,
            batch_state_history=batch_state_history,
            batch_state_history_st=batch_state_history_st,
            batch_state_future=batch_state_future,
            batch_state_future_st=batch_state_future_st,
            batch_edge=batch_edge,
            batch_first_timestep=batch_first_timestep,
            batch_last_timestep=batch_last_timestep,
            batch_edge_first_timestep=batch_edge_first_timestep,
            batch_map=batch_map,
            batch_node_size=batch_node_size,
            batch_lane_dev=batch_lane_dev,
            batch_fut_lane_dev=batch_fut_lane_dev,
            indices=indices,
        )
        z_list, pi_list, z_num, kl = self.encoder(
            mode,
            clique_type,
            node_history_encoded,
            batch_edge_hist_enc,
            encoded_map,
            clique_is_robot,
            node_future_encoded,
            indices,
            num_samples=num_samples,
        )

        if self.hyperparams["incl_robot_node"]:
            robot_traj = dict()
            for nt in self.node_types:
                robot_traj[nt] = dict()
                idx = torch.where(batch_is_robot[nt])[0]
                for k in range(idx.shape[0]):
                    robot_traj[nt][idx[k]] = (
                        batch_state_future[nt][:, idx[k]],
                        batch_state_future_st[nt][:, idx[k]],
                    )
        else:
            robot_traj = None
        ft = clique_future_state[0][0].shape[0]
        (
            state_pred_res,
            state_pred_res_st,
            input_pred_res,
            ref_traj,
            tracking_error,
            collision_cost,
        ) = self.decoder(
            mode,
            batch_state_history,
            batch_state_history_st,
            node_history_encoded,
            encoded_map,
            batch_node_size,
            batch_lane_st,
            robot_traj,
            indices,
            z_list,
            z_num,
            ft,
        )

        matching_loss = self.calc_traj_matching(
            mode,
            state_pred_res_st,
            batch_state_future_st,
            ref_traj,
            batch_last_timestep,
            pi_list,
            z_num,
        )

        ADE, FDE, ADE_count, FDE_count = self.eval_traj_matching(
            state_pred_res_st,
            batch_state_future_st,
            ref_traj,
            batch_first_timestep,
            batch_last_timestep,
            z_num,
            criterion,
        )
        coll_score = self.traj_collision_count(
            state_pred_res, batch_node_size, indices, pi_list, 1
        )

        nt_count = {nt: batch_state_history[nt].shape[1] for nt in self.node_types}

        return matching_loss, ADE, FDE, ADE_count, FDE_count, coll_score, nt_count

    def clique_batch_pred(
        self,
        clique_type,
        batch_state_history,
        state_pred_res,
        input_pred_res,
        ref_traj,
        indices,
        z_num,
        pi_list,
    ):
        Nclique = len(clique_type)
        clique_state_pred = [None] * Nclique
        clique_input_pred = [None] * Nclique
        clique_ref_traj = [None] * Nclique
        clique_pi_list = [None] * Nclique

        (
            node_index,
            edge_index,
            node_inverse_index,
            batch_node_to_edge_index,
            edge_to_node_index,
            batch_edge_idx1,
            batch_edge_idx2,
        ) = indices
        for i in range(Nclique):
            Nnode_i = len(clique_type[i])
            clique_state_pred[i] = [None] * Nnode_i
            clique_input_pred[i] = [None] * Nnode_i
            clique_ref_traj[i] = [None] * Nnode_i

            (nt, idx) = node_inverse_index[(i, 0)]
            clique_pi_list[i] = (
                pi_list[nt][idx, 0 : z_num[nt][idx]].detach().cpu().numpy()
            )
            for j in range(Nnode_i):
                (nt, idx) = node_inverse_index[(i, j)]
                clique_state_pred[i][j] = [None] * z_num[nt][idx]
                clique_input_pred[i][j] = [None] * z_num[nt][idx]
                clique_ref_traj[i][j] = [None] * z_num[nt][idx]
                for k in range(z_num[nt][idx]):
                    clique_state_pred[i][j][k] = (
                        state_pred_res[k][nt][:, idx].cpu().detach().numpy()
                    )
                    clique_input_pred[i][j][k] = (
                        input_pred_res[k][nt][:, idx].cpu().detach().numpy()
                    )

                    if not ref_traj[k][nt] is None:
                        if clique_type[i][j].name == "VEHICLE":
                            traj = ref_traj[k][nt][:, idx].cpu().detach().numpy()
                            x0 = batch_state_history[nt][-1, idx].cpu().detach().numpy()
                            theta = x0[3]
                            M = np.array(
                                [
                                    [np.cos(theta), np.sin(theta)],
                                    [-np.sin(theta), np.cos(theta)],
                                ]
                            )
                            clique_ref_traj[i][j][k] = traj @ M + x0[0:2]
                        elif clique_type[i][j].name == "PEDESTRIAN":
                            clique_ref_traj[i][j][k] = (
                                (
                                    ref_traj[k][nt][:, idx]
                                    + batch_state_history[nt][-1, idx, 0:2]
                                )
                                .cpu()
                                .detach()
                                .numpy()
                            )

        return clique_state_pred, clique_input_pred, clique_ref_traj, clique_pi_list

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
        clique_robot_traj,
        ft,
        num_samples=None,
        incl_robot_future=False,
    ):

        mode = ModeKeys.PREDICT

        (
            batch_state_history,
            batch_state_history_st,
            batch_state_future,
            batch_state_future_st,
            batch_edge,
            batch_first_timestep,
            batch_last_timestep,
            batch_edge_first_timestep,
            batch_map,
            batch_node_size,
            batch_is_robot,
            batch_lane,
            batch_lane_st,
            batch_lane_dev,
            batch_fut_lane_dev,
            indices,
        ) = self.batch_clique(
            mode,
            clique_type,
            clique_state_history,
            clique_first_timestep,
            None,
            clique_edges,
            clique_map,
            clique_node_size,
            clique_is_robot,
            clique_lane,
            clique_lane_dev,
            None,
            None,
        )
        (
            node_index,
            edge_index,
            node_inverse_index,
            batch_node_to_edge_index,
            edge_to_node_index,
            batch_edge_idx1,
            batch_edge_idx2,
        ) = indices
        (
            node_history_encoded,
            node_future_encoded,
            batch_edge_hist_enc,
            encoded_map,
        ) = self.obtain_encoded_tensors(
            mode=mode,
            batch_state_history=batch_state_history,
            batch_state_history_st=batch_state_history_st,
            batch_state_future=batch_state_future,
            batch_state_future_st=batch_state_future_st,
            batch_edge=batch_edge,
            batch_first_timestep=batch_first_timestep,
            batch_last_timestep=batch_last_timestep,
            batch_edge_first_timestep=batch_edge_first_timestep,
            batch_map=batch_map,
            batch_node_size=batch_node_size,
            batch_lane_dev=batch_lane_dev,
            batch_fut_lane_dev=batch_fut_lane_dev,
            indices=indices,
        )

        z_list, pi_list, z_num, kl = self.encoder(
            mode,
            clique_type,
            node_history_encoded,
            batch_edge_hist_enc,
            encoded_map,
            clique_is_robot,
            None,
            indices,
            num_samples,
        )
        if incl_robot_future and not clique_robot_traj is None:
            robot_traj = {nt: dict() for nt in self.node_types}

            for (i, j), traj_raw in clique_robot_traj.items():
                traj = torch.tensor(traj_raw).to(self.device)
                x0 = torch.tensor(clique_state_history[i][j][-1]).to(self.device)
                traj_st = self.rel_state_fun[clique_type[i][j]](
                    traj, x0.repeat(traj.shape[0], 1)
                )
                (nt, idx) = node_inverse_index[(i, j)]
                robot_traj[nt][idx] = (traj, traj_st)
        else:
            robot_traj = None

        (
            state_pred_res,
            state_pred_res_st,
            input_pred_res,
            ref_traj,
            tracking_error,
            collision_cost,
        ) = self.decoder(
            mode,
            batch_state_history,
            batch_state_history_st,
            node_history_encoded,
            encoded_map,
            batch_node_size,
            batch_lane_st,
            robot_traj,
            indices,
            z_list,
            z_num,
            ft,
        )

        (
            clique_state_pred,
            clique_input_pred,
            clique_ref_traj,
            clique_pi_list,
        ) = self.clique_batch_pred(
            clique_type,
            batch_state_history,
            state_pred_res,
            input_pred_res,
            ref_traj,
            indices,
            z_num,
            pi_list,
        )
        return clique_state_pred, clique_input_pred, clique_ref_traj, clique_pi_list
