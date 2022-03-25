import sys
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import dill
import argparse
from tqdm import tqdm
from pyquaternion import Quaternion
from kalman_filter import NonlinearKinematicBicycle
import pdb
from nuscenes.map_expansion import arcline_path_utils


nu_path = "./devkit/python-sdk/"
sys.path.append(nu_path)
sys.path.append("../../ScePT")
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from environment import Environment, Scene, Node, GeometricMap, derivative_of

# Multiprocessing
from functools import partial
from pathos.multiprocessing import ProcessPool as Pool

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration", "heading"], ["x", "y"]]
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_tuples([("heading", "°"), ("heading", "d°")])
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_product([["velocity", "acceleration"], ["norm"]])
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_tuples([("lane", "delta_y"), ("lane", "delta_heading")])
)

data_columns_pedestrian = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration"], ["x", "y"]]
)

curv_0_2 = 0
curv_0_1 = 0
total = 0


def smooth_angle_kinks(theta0):
    theta = np.array(theta0)
    for i in range(1, theta.shape[0]):
        if theta[i] > theta[i - 1] + np.pi:
            theta[i] -= 2 * np.pi
        elif theta[i] < theta[i - 1] - np.pi:
            theta[i] += 2 * np.pi
    return theta


standardization = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    },
    "VEHICLE": {
        "position": {"x": {"mean": 0, "std": 80}, "y": {"mean": 0, "std": 80}},
        "velocity": {
            "x": {"mean": 0, "std": 15},
            "y": {"mean": 0, "std": 15},
            "norm": {"mean": 0, "std": 15},
        },
        "acceleration": {
            "x": {"mean": 0, "std": 4},
            "y": {"mean": 0, "std": 4},
            "norm": {"mean": 0, "std": 4},
        },
        "heading": {
            "x": {"mean": 0, "std": 1},
            "y": {"mean": 0, "std": 1},
            "°": {"mean": 0, "std": np.pi},
            "d°": {"mean": 0, "std": 1},
        },
    },
}


def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns_vehicle = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration", "heading"], ["x", "y"]]
    )
    data_columns_vehicle = data_columns_vehicle.append(
        pd.MultiIndex.from_tuples([("heading", "°"), ("heading", "d°")])
    )
    data_columns_vehicle = data_columns_vehicle.append(
        pd.MultiIndex.from_product([["velocity", "acceleration"], ["norm"]])
    )

    data_columns_pedestrian = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]]
    )

    scene_aug = Scene(
        timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene
    )

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == "PEDESTRIAN":
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
            }

            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(
                node_type=node.type,
                node_id=node.id,
                data=node_data,
                first_timestep=node.first_timestep,
            )
        elif node.type == "VEHICLE":
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()
            delta_y = node.data.lane.delta_y.copy()
            delta_heading = node.data.lane.delta_heading.copy()

            heading = getattr(node.data.heading, "°").copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.0))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            a_norm = np.divide(
                ax * vx + ay * vy,
                v_norm[..., 0],
                out=np.zeros_like(ax),
                where=(v_norm[..., 0] > 1.0),
            )
            d_heading = derivative_of(heading, scene.dt, radian=True)
            data_dict = {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("velocity", "norm"): np.linalg.norm(
                    np.stack((vx, vy), axis=-1), axis=-1
                ),
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
                ("acceleration", "norm"): a_norm,
                ("heading", "x"): heading_x,
                ("heading", "y"): heading_y,
                ("heading", "°"): heading,
                ("heading", "d°"): d_heading,
                ("lane", "delta_y"): delta_y,
                ("lane", "delta_heading"): delta_heading,
            }

            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            node = Node(
                node_type=node.type,
                node_id=node.id,
                data=node_data,
                first_timestep=node.first_timestep,
                non_aug_node=node,
            )

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.0):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def aggregate_samples(nusc, all_sample_tokens, start, data_class):
    sample = nusc.get("sample", start)
    samples = [sample]

    past_idx = 0
    while past_idx < 4:  # Only go back 2 seconds
        if sample["prev"]:
            sample = nusc.get("sample", sample["prev"])
            samples.insert(0, sample)

            if sample["token"] in all_sample_tokens:
                past_idx = 0
            else:
                past_idx += 1
        else:
            break

    sample = samples[-1]
    future_idx = 0
    while future_idx < 6:  # Only go forward 3 seconds
        if sample["next"]:
            sample = nusc.get("sample", sample["next"])
            samples.append(sample)

            if sample["token"] in all_sample_tokens:
                future_idx = 0
            else:
                future_idx += 1
        else:
            break

    return samples


def process_scene(token_samples, env, data_path, data_class):
    sample_token, samples, map_name, ado_data, ego_data = token_samples
    data = pd.DataFrame(
        columns=[
            "frame_id",
            "type",
            "node_id",
            "robot",
            "x",
            "y",
            "z",
            "length",
            "width",
            "height",
            "heading",
        ]
    )

    attribute_dict = defaultdict(set)

    frame_id = 0
    for sample in samples:
        annotation_tokens = sample["anns"]
        for annotation_token in annotation_tokens:
            if annotation_token not in ado_data:
                # This happens when there's no attribute for a non-vehicle agent (see lines 500-506)
                continue

            annotation, category, attribute = ado_data[annotation_token]

            if (
                "pedestrian" in category
                and "stroller" not in category
                and "wheelchair" not in category
            ):
                our_category = env.NodeType.PEDESTRIAN
            elif (
                "vehicle" in category
                and "bicycle" not in category
                and "motorcycle" not in category
            ):  # and 'parked' not in attribute:
                our_category = env.NodeType.VEHICLE
            else:
                continue

            attribute_dict[annotation["instance_token"]].add(attribute)

            data_point = pd.Series(
                {
                    "frame_id": frame_id,
                    "type": our_category,
                    "node_id": annotation["instance_token"],
                    "robot": False,
                    "x": annotation["translation"][0],
                    "y": annotation["translation"][1],
                    "z": annotation["translation"][2],
                    "length": annotation["size"][0],
                    "width": annotation["size"][1],
                    "height": annotation["size"][2],
                    "heading": Quaternion(annotation["rotation"]).yaw_pitch_roll[0],
                }
            )
            data = data.append(data_point, ignore_index=True)

        # Ego Vehicle
        our_category = env.NodeType.VEHICLE
        sample_data_token = sample["data"]["CAM_FRONT"]
        annotation = ego_data[sample_data_token]

        data_point = pd.Series(
            {
                "frame_id": frame_id,
                "type": our_category,
                "node_id": "ego",
                "robot": True,
                "x": annotation["translation"][0],
                "y": annotation["translation"][1],
                "z": annotation["translation"][2],
                "length": 4,
                "width": 1.7,
                "height": 1.5,
                "heading": Quaternion(annotation["rotation"]).yaw_pitch_roll[0],
                "orientation": None,
            }
        )
        data = data.append(data_point, ignore_index=True)

        frame_id += 1

    if len(data.index) == 0:
        return None

    data.sort_values("frame_id", inplace=True)
    max_timesteps = data["frame_id"].max()

    x_min = np.round(data["x"].min() - 50)
    x_max = np.round(data["x"].max() + 50)
    y_min = np.round(data["y"].min() - 50)
    y_max = np.round(data["y"].max() + 50)

    data["x"] = data["x"] - x_min
    data["y"] = data["y"] - y_min

    # Generate Maps
    nusc_map = NuScenesMap(dataroot=data_path, map_name=map_name)
    scene = Scene(
        timesteps=max_timesteps + 1,
        dt=env.dt,
        name=sample_token,
        aug_func=augment,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        map_name=map_name,
    )

    type_map = dict()
    x_size = x_max - x_min
    y_size = y_max - y_min
    patch_box = (
        x_min + 0.5 * (x_max - x_min),
        y_min + 0.5 * (y_max - y_min),
        y_size,
        x_size,
    )
    patch_angle = 0  # Default orientation where North is up
    canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
    homography = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    layer_names = [
        "lane",
        "road_segment",
        "drivable_area",
        "road_divider",
        "lane_divider",
        "stop_line",
        "ped_crossing",
        "stop_line",
        "ped_crossing",
        "walkway",
    ]
    map_mask = (
        nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0
    ).astype(np.uint8)
    map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
    # PEDESTRIANS
    map_mask_pedestrian = np.stack(
        (map_mask[9], map_mask[8], np.max(map_mask[:3], axis=0)), axis=0
    )
    type_map["PEDESTRIAN"] = GeometricMap(
        data=map_mask_pedestrian,
        homography=homography,
        description=", ".join(layer_names),
    )
    # VEHICLES
    map_mask_vehicle = np.stack(
        (np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0
    )
    type_map["VEHICLE"] = GeometricMap(
        data=map_mask_vehicle, homography=homography, description=", ".join(layer_names)
    )

    map_mask_plot = np.stack(
        (
            (
                np.max(map_mask[:3], axis=0)
                - (map_mask[3] + 0.5 * map_mask[4]).clip(max=255)
            )
            .clip(min=0)
            .astype(np.uint8),
            map_mask[8],
            map_mask[9],
        ),
        axis=0,
    )
    type_map["VISUALIZATION"] = GeometricMap(
        data=map_mask_plot, homography=homography, description=", ".join(layer_names)
    )

    scene.map = type_map
    scene.nusc_map = map_name
    del map_mask
    del map_mask_pedestrian
    del map_mask_vehicle
    del map_mask_plot

    for node_id in pd.unique(data["node_id"]):
        node_frequency_multiplier = 1
        node_df = data[data["node_id"] == node_id]

        # Catching instances of vehicles that were parked and then moving, but not allowing
        # only parked vehicles through.
        if (
            len(attribute_dict[node_id]) == 1
            and "vehicle.parked" in attribute_dict[node_id]
        ):
            continue

        if node_df["x"].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df["frame_id"]) == 1):
            min_index = node_df["frame_id"].min()
            max_index = node_df["frame_id"].max()
            node_df = (
                node_df.set_index("frame_id")
                .reindex(range(min_index, max_index + 1))
                .interpolate()
                .reset_index()
            )
            node_df["type"] = node_df["type"].mode()[0]
            node_df["node_id"] = node_id
            node_df["robot"] = False
            # print('Occlusion')
            # continue  # TODO Make better

        node_values = node_df[["x", "y"]].values
        x = node_values[:, 0]
        y = node_values[:, 1]

        heading0 = node_df["heading"].values
        heading = smooth_angle_kinks(heading0)

        if node_df.iloc[0]["type"] == env.NodeType.VEHICLE and not node_id == "ego":
            # Kalman filter Agent
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

            filter_veh = NonlinearKinematicBicycle(dt=scene.dt, sMeasurement=1.0)
            P_matrix = None
            for i in range(len(x)):
                if i == 0:  # initalize KF
                    # initial P_matrix
                    P_matrix = np.identity(4)
                elif i < len(x):
                    # assign new est values
                    x[i] = x_vec_est_new[0][0]
                    y[i] = x_vec_est_new[1][0]
                    heading[i] = x_vec_est_new[2][0]
                    velocity[i] = x_vec_est_new[3][0]

                if i < len(x) - 1:  # no action on last data
                    # filtering
                    x_vec_est = np.array([[x[i]], [y[i]], [heading[i]], [velocity[i]]])
                    z_new = np.array(
                        [[x[i + 1]], [y[i + 1]], [heading[i + 1]], [velocity[i + 1]]]
                    )
                    x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                        x_vec_est=x_vec_est,
                        u_vec=np.array([[0.0], [0.0]]),
                        P_matrix=P_matrix,
                        z_new=z_new,
                    )
                    P_matrix = P_matrix_new

            curvature, pl, _ = trajectory_curvature(np.stack((x, y), axis=-1))
            if pl < 1.0:  # vehicle is "not" moving
                x = x[0].repeat(max_timesteps + 1)
                y = y[0].repeat(max_timesteps + 1)
                heading = heading[0].repeat(max_timesteps + 1)
            global total
            global curv_0_2
            global curv_0_1
            total += 1
            if pl > 1.0:
                if curvature > 0.2:
                    curv_0_2 += 1
                    node_frequency_multiplier = 3 * int(np.floor(total / curv_0_2))
                elif curvature > 0.1:
                    curv_0_1 += 1
                    node_frequency_multiplier = 3 * int(np.floor(total / curv_0_1))

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        width = node_df["width"].mean()
        length = node_df["length"].mean()
        if width > length:
            temp = width
            width = length
            length = temp

        if node_df.iloc[0]["type"] == env.NodeType.VEHICLE:
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.0))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            a_norm = np.divide(
                ax * vx + ay * vy,
                v_norm[..., 0],
                out=np.zeros_like(ax),
                where=(v_norm[..., 0] > 1.0),
            )
            d_heading = derivative_of(heading, scene.dt, radian=True)

            delta_y = np.zeros_like(x)
            delta_heading = np.zeros_like(heading)
            for t in range(x.shape[0]):
                map_x, map_y, yaw = x[t] + x_min, y[t] + y_min, heading[t]
                closest_lane = nusc_map.get_closest_lane(map_x, map_y, radius=2)
                if closest_lane != "":
                    lane_record = nusc_map.get_arcline_path(closest_lane)
                    (
                        closest_pose_on_lane,
                        distance_along_lane,
                    ) = arcline_path_utils.project_pose_to_lane(
                        (map_x, map_y, yaw), lane_record
                    )
                    x0, y0, yaw0 = closest_pose_on_lane
                    delta_y[t] = -(map_x - x0) * np.sin(yaw0) - (map_y - y0) * np.cos(
                        yaw0
                    )
                    delta_heading[t] = yaw - yaw0

            data_dict = {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("velocity", "norm"): np.linalg.norm(
                    np.stack((vx, vy), axis=-1), axis=-1
                ),
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
                ("acceleration", "norm"): a_norm,
                ("heading", "x"): heading_x,
                ("heading", "y"): heading_y,
                ("heading", "°"): heading,
                ("heading", "d°"): d_heading,
                ("lane", "delta_y"): delta_y,
                ("lane", "delta_heading"): delta_heading,
            }
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        else:
            data_dict = {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
            }
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

        node = Node(
            node_type=node_df.iloc[0]["type"],
            node_id=node_id,
            data=node_data,
            length=length,
            width=width,
            frequency_multiplier=node_frequency_multiplier,
        )
        node.first_timestep = node_df["frame_id"].iloc[0]
        if node_df.iloc[0]["robot"] == True:
            node.is_robot = True
            scene.robot = node

        scene.nodes.append(node)

        if data_class == "train":
            scene.augmented = list()
            angles = np.arange(0, 360, 15)
            for angle in angles:
                scene.augmented.append(augment_scene(scene, angle))
    return scene


def process_data(data_path, version, output_path, num_workers):
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    helper = PredictHelper(nusc)
    for data_class in ["train_val", "val", "train"]:
        # for data_class in ['mini_train', 'mini_val']: # ['mini_train', 'mini_val']:
        env = Environment(
            node_type_list=["VEHICLE", "PEDESTRIAN"],
            standardization=standardization,
            dt=dt,
        )
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE

        instance_sample_tokens = get_prediction_challenge_split(
            data_class, dataroot=data_path
        )
        all_sample_tokens = set(
            [token.split("_")[1] for token in instance_sample_tokens]
        )

        processed_sample_tokens = set()
        samples_list = list()
        for instance_sample_token in tqdm(
            instance_sample_tokens, desc=f"Preprocessing {data_class} Samples"
        ):
            _, sample_token = instance_sample_token.split("_")
            if sample_token in processed_sample_tokens:
                continue
            samples = aggregate_samples(
                nusc, all_sample_tokens, start=sample_token, data_class=data_class
            )
            processed_sample_tokens.update([s["token"] for s in samples])
            map_name = helper.get_map_name_from_sample_token(sample_token)

            ado_data = dict()
            ego_data = dict()
            for sample in samples:
                annotation_tokens = sample["anns"]

                for annotation_token in annotation_tokens:
                    annotation = nusc.get("sample_annotation", annotation_token)
                    category = annotation["category_name"]
                    if len(annotation["attribute_tokens"]):
                        attribute = nusc.get(
                            "attribute", annotation["attribute_tokens"][0]
                        )["name"]
                    else:
                        if "vehicle" in category:
                            attribute = ""
                        else:
                            continue

                    ado_data[annotation_token] = (annotation, category, attribute)

                sample_data_token = sample["data"]["CAM_FRONT"]
                sample_data = nusc.get("sample_data", sample_data_token)
                ego_data[sample_data_token] = nusc.get(
                    "ego_pose", sample_data["ego_pose_token"]
                )

            samples_list.append((sample_token, samples, map_name, ado_data, ego_data))

        if num_workers > 1:
            with Pool(num_workers) as pool:
                scenes = list(
                    tqdm(
                        pool.imap(
                            partial(
                                process_scene,
                                env=env,
                                data_path=data_path,
                                data_class=data_class,
                            ),
                            samples_list,
                        ),
                        desc=f"Processing {data_class} Scenes ({num_workers} CPUs)",
                        total=len(samples_list),
                    )
                )
        else:
            scenes = list()
            for sample_set in samples_list:

                scenes.append(
                    process_scene(
                        sample_set, env=env, data_path=data_path, data_class=data_class
                    )
                )

        print(f"Processed {len(scenes)} scenes")

        env.scenes = scenes

        if len(scenes) > 0:
            data_dict_path = os.path.join(
                output_path, "nuScenes_" + data_class + ".pkl"
            )
            with open(data_dict_path, "wb") as f:
                dill.dump(env, f, protocol=4)  # For Python 3.6 and 3.8 compatability.
            print("Saved Environment!")

        global total
        global curv_0_2
        global curv_0_1
        print(f"Total Nodes: {total}")
        print(f"Curvature > 0.1 Nodes: {curv_0_1}")
        print(f"Curvature > 0.2 Nodes: {curv_0_2}")
        total = 0
        curv_0_1 = 0
        curv_0_2 = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    args = parser.parse_args()
    process_data(args.data, args.version, args.output_path, args.num_workers)
