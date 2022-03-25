import os
from torch.utils import data
import numpy as np
import pdb


try:
    from math import prod
except ImportError:
    from functools import reduce  # Required in Python 3
    import operator
    def prod(iterable):
        return reduce(operator.mul, iterable, 1)

from .preprocessing import get_node_timestep_data
from tqdm import tqdm
from environment import EnvironmentMetadata
from functools import partial
from pathos.multiprocessing import ProcessPool as Pool

class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self._augment = False
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDataset(env, node_type, state, pred_state, node_freq_mult,
                                                           scene_freq_mult, hyperparams, **kwargs))

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


def parallel_process_scene(scene, env_metadata, node_type, 
                           state, pred_state, edge_types, 
                           max_ht, max_ft, 
                           node_freq_mult, scene_freq_mult, 
                           hyperparams, augment, kwargs):
    results = list()
    indexing_info = list()

    tsteps = np.arange(0, scene.timesteps)
    present_node_dict = scene.present_nodes(tsteps, type=node_type, **kwargs)

    for t, nodes in present_node_dict.items():
        for node in nodes:
            if augment:
                scene_aug = scene.augment()
                node_aug = scene.get_node_by_id(node.id)

                scene_data = get_node_timestep_data(env_metadata, scene_aug, t, node_aug, state, pred_state, 
                                                    edge_types, max_ht, max_ft, hyperparams)
            else:
                scene_data = get_node_timestep_data(env_metadata, scene, t, node, state, pred_state, 
                                                    edge_types, max_ht, max_ft, hyperparams)

            results += [(
                scene_data, 
                (scene, t, node)
            )]

            indexing_info += [(
                scene.frequency_multiplier if scene_freq_mult else 1, 
                node.frequency_multiplier if node_freq_mult else 1
            )]
    return (results, indexing_info)


class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.env_metadata = EnvironmentMetadata(env)
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]
        self.index, self.data, self.data_origin = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = self.index.shape[0]

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        num_cpus = kwargs['num_workers']
        del kwargs['num_workers']

        rank = kwargs['rank']
        del kwargs['rank']

        # parallel_process_scene(self.env.scenes[0], 
        #                        env_metadata=self.env_metadata, 
        #                        node_type=self.node_type,
        #                        state=self.state,
        #                        pred_state=self.pred_state,
        #                        edge_types=self.edge_types,
        #                        max_ht=self.max_ht,
        #                        max_ft=self.max_ft,
        #                        node_freq_mult=node_freq_mult,
        #                        scene_freq_mult=scene_freq_mult,
        #                        hyperparams=self.hyperparams,
        #                        augment=self.augment,
        #                        kwargs=kwargs)
        with Pool(num_cpus) as pool:
            indexed_scenes = list(
                tqdm(
                    pool.imap(
                        partial(parallel_process_scene,
                                env_metadata=self.env_metadata,
                                node_type=self.node_type,
                                state=self.state,
                                pred_state=self.pred_state,
                                edge_types=self.edge_types,
                                max_ht=self.max_ht,
                                max_ft=self.max_ft,
                                node_freq_mult=node_freq_mult,
                                scene_freq_mult=scene_freq_mult,
                                hyperparams=self.hyperparams,
                                augment=self.augment,
                                kwargs=kwargs),
                        self.env.scenes
                    ),
                    desc=f'Indexing {self.node_type}s ({num_cpus} CPUs)',
                    total=len(self.env.scenes),
                    disable=(rank > 0)
                )
            )
        results = list()
        indexing_info = list()
        for res in indexed_scenes:
            results.extend(res[0])
            indexing_info.extend(res[1])

        index = list()
        for i, counts in enumerate(indexing_info):
            total = prod(counts)

            index += [i]*total

        data, data_origin = zip(*results)

        return np.asarray(index, dtype=int), list(data), list(data_origin)

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.data[self.index[i]]

class clique_dataset(data.Dataset):
    def __init__(self, clique_data):

        self.clique_data = clique_data
        clique_type,clique_state_history,clique_first_timestep,clique_last_timestep,clique_edges,clique_future_state,clique_map,clique_node_size,clique_is_robot,clique_lane,clique_lane_dev,clique_fut_lane_dev = zip(*clique_data)
        self.clique_type = clique_type
        self.clique_state_history = clique_state_history
        self.clique_first_timestep = clique_first_timestep
        self.clique_last_timestep = clique_last_timestep
        self.clique_edges = clique_edges
        self.clique_future_state = clique_future_state
        self.clique_map = clique_map
        self.clique_node_size = clique_node_size
        self.clique_is_robot = clique_is_robot
        self.clique_lane = clique_lane
        self.clique_lane_dev = clique_lane_dev 
        self.clique_fut_lane_dev = clique_fut_lane_dev


    def __len__(self):
        return len(self.clique_data)

    def pin_memory(self):
        self.clique_type = self.clique_type.pin_memory()
        self.clique_state_history = self.clique_state_history.pin_memory()
        self.clique_first_timestep = self.clique_first_timestep.pin_memory()
        self.clique_last_timestep = self.clique_last_timestep.pin_memory()
        self.clique_edges = self.clique_edges.pin_memory()
        self.clique_future_state = self.clique_future_state.pin_memory()
        self.clique_map = self.clique_map.pin_memory()
        self.clique_node_size = self.clique_node_size.pin_memory()
        self.clique_is_robot = self.clique_is_robot.pin_memory()
        self.clique_lane = self.clique_lane.pin_memory()
        self.clique_lane_dev = self.clique_lane_dev.pin_memory()
        self.clique_fut_lane_dev = self.clique_fut_lane_dev.pin_memory()


    def __getitem__(self, idx):
        return self.clique_type[idx],self.clique_state_history[idx],self.clique_first_timestep[idx],self.clique_last_timestep[idx],\
               self.clique_edges[idx],self.clique_future_state[idx],self.clique_map[idx],self.clique_node_size[idx],self.clique_is_robot[idx],\
               self.clique_lane[idx],self.clique_lane_dev[idx],self.clique_fut_lane_dev[idx]

class IRL_dataset(data.Dataset):
    def __init__(self, data):

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nb_types,state_history,node_input,first_timestep,nb_history,nb_first_time,node_map,node_size,nb_node_size,lane,lane_dev,nb_deflt_traj = self.data[idx]
        return nb_types,state_history,node_input,first_timestep,nb_history,nb_first_time,node_map,node_size,nb_node_size,lane,lane_dev,nb_deflt_traj


