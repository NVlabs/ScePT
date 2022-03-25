from scipy import linalg, interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib import animation
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import pdb


def plot_trajectories(
    ax,
    prediction_dict,
    histories_dict,
    futures_dict,
    line_alpha=0.7,
    line_width=0.2,
    edge_width=2,
    circle_edge_width=0.5,
    node_circle_size=0.3,
    batch_num=0,
    kde=False,
):

    cmap = ["k", "b", "y", "g", "r"]

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], "k--")

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(
                        predictions[batch_num, :, t, 0],
                        predictions[batch_num, :, t, 1],
                        ax=ax,
                        shade=True,
                        shade_lowest=False,
                        color=np.random.choice(cmap),
                        alpha=0.8,
                    )

            ax.plot(
                predictions[batch_num, sample_num, :, 0],
                predictions[batch_num, sample_num, :, 1],
                color=cmap[node.type.value],
                linewidth=line_width,
                alpha=line_alpha,
            )

            ax.plot(
                future[:, 0],
                future[:, 1],
                "w--",
                path_effects=[
                    pe.Stroke(linewidth=edge_width, foreground="k"),
                    pe.Normal(),
                ],
            )

            # Current Node Position
            circle = plt.Circle(
                (history[-1, 0], history[-1, 1]),
                node_circle_size,
                facecolor="g",
                edgecolor="k",
                lw=circle_edge_width,
                zorder=3,
            )
            ax.add_artist(circle)

    ax.axis("equal")


def plot_trajectories_clique(
    clique_type,
    clique_last_timestep,
    clique_state_history,
    clique_future_state,
    clique_state_pred,
    clique_ref_traj,
    map,
    clique_node_size,
    clique_is_robot,
    limits,
    emphasized_nodes=None,
    line_alpha=0.7,
    line_width=0.9,
    edge_width=2,
    circle_edge_width=0.5,
    node_circle_size=0.3,
    show_clique=False,
):

    cmap = ["k", "b", "m", "g"]

    emph_color = ["tab:red", "tab:orange", "c", "tab:brown", "tab:blue", "tab:red"]
    if map is not None:

        map_shape = map.as_image().shape
        fig, ax = plt.subplots(figsize=(35, 35 / map_shape[1] * map_shape[0]))
        ax.clear()
        ax.grid(False)
        ax.imshow(map.as_image(), origin="lower", alpha=0.3)

        scale = map.homography[0, 0]
        if not limits is None:
            xlim = [
                (map_shape[1] - limits[0] * scale) / 2,
                (map_shape[1] + limits[0] * scale) / 2,
            ]
            ylim = [
                (map_shape[0] - limits[1] * scale) / 2,
                (map_shape[0] + limits[1] * scale) / 2,
            ]
            plt.xlim(xlim)
            plt.ylim(ylim)

        # plot nodes

        for n in range(len(clique_type)):
            map_coords = list()
            for i in range(len(clique_type[n])):
                map_coords.append(
                    map.to_map_points(clique_state_history[n][i][-1:, 0:2])[0]
                )
                # for i in range(len(clique_type[n])):

                # ax.text(
                #     map_coords[i][0],
                #     map_coords[i][1],
                #     str(f"{n},{i}"),
                #     fontsize=12,
                #     verticalalignment="top",
                # )
                if clique_type[n][i] == "PEDESTRIAN":
                    if emphasized_nodes is not None and (n, i) in emphasized_nodes:
                        color = emph_color[emphasized_nodes.index((n, i))]
                    else:
                        color = cmap[clique_type[n][i].value]
                    circle = plt.Circle(
                        (map_coords[i][0], map_coords[i][1]),
                        clique_node_size[n][i][0] * scale,
                        facecolor=color,
                        edgecolor=color,
                        lw=circle_edge_width,
                        zorder=3,
                    )
                    ax.add_artist(circle)
                    for k in range(len(clique_state_pred[n][i])):
                        traj = map.to_map_points(clique_state_pred[n][i][k][:, 0:2])
                        traj = np.vstack((map_coords[i], traj))
                        ax.plot(
                            traj[:, 0],
                            traj[:, 1],
                            color=cmap[clique_type[n][i].value],
                            linewidth=line_width,
                            alpha=line_alpha,
                        )

                elif clique_type[n][i] == "VEHICLE":
                    zorder = 1
                    if (n, i) == (3, 3):
                        zorder = 2

                    # coords = ts.transform([map_coords[i][0],map_coords[i][1]])
                    L = clique_node_size[n][i][0] * scale
                    W = clique_node_size[n][i][1] * scale
                    psi = clique_state_history[n][i][-1, 3]

                    tran = np.array(
                        [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
                    )
                    edges = np.array(
                        [
                            [-L / 2, -W / 2],
                            [-L / 2, W / 2],
                            [L / 2, W / 2],
                            [L / 2, -W / 2],
                        ]
                    )
                    rotated_edge = edges @ (tran.T)
                    xy = np.array([map_coords[i][0], map_coords[i][1]]) + rotated_edge
                    if emphasized_nodes is not None and (n, i) in emphasized_nodes:
                        color = emph_color[emphasized_nodes.index((n, i))]
                    else:
                        color = cmap[clique_type[n][i].value]

                    patch = plt.Polygon(xy, fc=color, ec="w", lw=1.2, zorder=zorder)

                    ax.add_artist(patch)

                    # ax.add_artist(patch)
                    if emphasized_nodes is not None and (n, i) in emphasized_nodes:
                        # for k in range(len(clique_state_pred[n][i])):
                        if len(clique_state_pred[n][i]) >= 2:
                            k = 0
                        else:
                            k = 0
                        ft = clique_state_pred[n][i][k].shape[0]
                        x0y0 = map.to_map_points(clique_state_pred[n][i][k][:, 0:2])
                        for t1 in range(ft):
                            psi = clique_state_pred[n][i][k][t1, 3]
                            tran = np.array(
                                [
                                    [np.cos(psi), -np.sin(psi)],
                                    [np.sin(psi), np.cos(psi)],
                                ]
                            )
                            edges = np.array(
                                [
                                    [-L / 2, -W / 2],
                                    [-L / 2, W / 2],
                                    [L / 2, W / 2],
                                    [L / 2, -W / 2],
                                ]
                            )
                            rotated_edge = edges @ (tran.T)

                            xy = x0y0[t1] + rotated_edge
                            color = emph_color[emphasized_nodes.index((n, i))]
                            alpha = (ft - t1) / ft * 0.4 + 0.2
                            # alpha = 0.5

                            patch = plt.Polygon(
                                xy,
                                ec=color,
                                alpha=alpha,
                                lw=2,
                                fc=color,
                                zorder=zorder,
                            )
                            ax.add_artist(patch)
                        traj = map.to_map_points(clique_state_pred[n][i][k][:, 0:2])
                        traj = np.vstack((map_coords[i], traj))
                        ax.plot(
                            traj[:, 0],
                            traj[:, 1],
                            color=color,
                            marker=".",
                            markersize=8,
                            linewidth=line_width,
                            alpha=line_alpha,
                        )
                    else:
                        for k in range(len(clique_state_pred[n][i])):
                            traj = map.to_map_points(clique_state_pred[n][i][k][:, 0:2])
                            traj = np.vstack((map_coords[i], traj))
                            # ax.plot(
                            #     traj[:, 0],
                            #     traj[:, 1],
                            #     color=cmap[clique_type[n][i].value],
                            #     linewidth=line_width,
                            #     alpha=line_alpha,
                            # )

                        # traj = map.to_map_points(clique_future_state[n][i][:clique_last_timestep[n][i],0:2])
                        # traj = np.vstack((map_coords[i],traj))
                        # ax.plot(traj[:,0], traj[:,1],
                        #                 color='g',
                        #                 linewidth=line_width*2)
                if show_clique:
                    for j in range(i + 1, len(clique_type[n])):
                        ax.plot(
                            [map_coords[i][0], map_coords[j][0]],
                            [map_coords[i][1], map_coords[j][1]],
                            color="r",
                            linewidth=line_width,
                            alpha=line_alpha,
                        )

    else:
        fig, ax = plt.subplots(figsize=(25, 25))
        ts = ax.transData

        # plot nodes

        for n in range(len(clique_type)):
            coords = list()
            for i in range(len(clique_type[n])):
                coords.append(clique_state_history[n][i][-1, 0:2])
            for i in range(len(clique_type[n])):
                if clique_type[n][i] == "PEDESTRIAN":
                    circle = plt.Circle(
                        (coords[i][0], coords[i][1]),
                        clique_node_size[n][i][0],
                        facecolor=cmap[clique_type[n][i].value],
                        edgecolor="k",
                        lw=circle_edge_width,
                        zorder=3,
                    )
                    ax.add_artist(circle)

                elif clique_type[n][i] == "VEHICLE":
                    coords = ts.transform([coords[i][0], coords[i][1]])
                    patch = plt.Rectangle(
                        (
                            coords[i][0] - clique_node_size[n][i][0] / 2,
                            coords[i][1] - clique_node_size[n][i][1] / 2,
                        ),
                        clique_node_size[n][i][0],
                        clique_node_size[n][i][1],
                        fc=cmap[clique_type[n][i].value],
                        zorder=1,
                    )
                    tr = matplotlib.transforms.Affine2D().rotate_around(
                        coords[0], coords[1], clique_state_history[n][i][-1, 3]
                    )
                    patch.set_transform(ts + tr)
                    ax.add_artist(patch)
                for k in range(len(clique_state_pred[n][i])):
                    traj = clique_state_pred[n][i][k][:, 0:2]
                    traj = np.vstack((coords[i], traj))
                    ax.plot(
                        traj[:, 0],
                        traj[:, 1],
                        color=cmap[clique_type[n][i].value],
                        linewidth=line_width,
                        alpha=line_alpha,
                    )
                if show_clique:
                    for j in range(i + 1, len(clique_type[n])):
                        ax.plot(
                            [coords[i][0], coords[j][0]],
                            [coords[i][1], coords[j][1]],
                            color="r",
                            linewidth=line_width,
                            alpha=line_alpha,
                        )

    # ax.axis('equal')
    # plt.show()
    return fig, ax


def animate_traj_pred_clique(
    dt,
    clique_type,
    clique_last_timestep,
    clique_state_history,
    clique_future_state,
    clique_state_pred,
    clique_ref_traj,
    map,
    clique_node_size,
    clique_is_robot,
    limits,
    output,
    interp_N=1,
    emphasized_nodes=None,
    line_alpha=0.7,
    line_width=0.9,
    edge_width=2,
    circle_edge_width=0.5,
    node_circle_size=0.3,
):

    bs = len(clique_state_history)
    ft = clique_state_pred[0][0][0].shape[0]
    clique_traj = [None] * bs
    for i in range(bs):
        clique_traj[i] = [None] * len(clique_future_state[i])
        for j in range(len(clique_future_state[i])):
            traj1 = np.vstack(
                (clique_state_history[i][j][-1], clique_state_pred[i][j][0])
            )
            if map is not None:
                map_coord = map.to_map_points(traj1[:, 0:2])
                traj1[:, 0:2] = map_coord
            f = interpolate.interp1d(np.arange(0, ft + 1), traj1, axis=0)
            clique_traj[i][j] = f(np.linspace(0, ft, ft * interp_N))

    ts = dt / interp_N
    nframe = clique_state_pred[0][0][0].shape[0] * interp_N

    if output:
        matplotlib.use("Agg")

    if not map is None:
        map_shape = map.as_image().shape
        fig, ax = plt.subplots(figsize=(25, 25 / map_shape[1] * map_shape[0]))
    else:
        fig, ax = plt.subplots(figsize=(25, 25))

    def animate(t, clique_type, clique_traj, clique_node_size, map):
        cmap = ["k", "b", "y", "g"]

        emph_color = [
            "m",
            "c",
            "tab:blue",
            "tab:orange",
            "tab:purple",
            "tab:pink",
            "tab:brown",
        ]

        ax.clear()
        if map is not None:

            ax.imshow(map.as_image(), origin="lower", alpha=0.5)

            scale = map.homography[0, 0]
            if not limits is None:
                xlim = [
                    (map_shape[1] - limits[0] * scale) / 2,
                    (map_shape[1] + limits[0] * scale) / 2,
                ]
                ylim = [
                    (map_shape[0] - limits[1] * scale) / 2,
                    (map_shape[0] + limits[1] * scale) / 2,
                ]
                plt.xlim(xlim)
                plt.ylim(ylim)
            # plot nodes

            for n in range(len(clique_type)):
                for i in range(len(clique_type[n])):
                    ax.text(
                        clique_traj[n][i][t, 0],
                        clique_traj[n][i][t, 1],
                        str(f"{n},{i}"),
                        fontsize=10,
                        verticalalignment="top",
                    )
                    if clique_type[n][i] == "PEDESTRIAN":
                        if emphasized_nodes is not None and (n, i) in emphasized_nodes:
                            color = emph_color[emphasized_nodes.index((n, i))]
                        else:
                            color = cmap[clique_type[n][i].value]
                        circle = plt.Circle(
                            (clique_traj[n][i][t, 0], clique_traj[n][i][t, 1]),
                            clique_node_size[n][i][0] * scale,
                            facecolor=color,
                            edgecolor=color,
                            lw=circle_edge_width,
                            zorder=3,
                        )
                        ax.add_artist(circle)

                    elif clique_type[n][i] == "VEHICLE":

                        # coords = ts.transform([map_coords[i][0],map_coords[i][1]])
                        L = clique_node_size[n][i][0] * scale
                        W = clique_node_size[n][i][1] * scale
                        psi = clique_traj[n][i][t, 3]

                        tran = np.array(
                            [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
                        )
                        edges = np.array(
                            [
                                [-L / 2, -W / 2],
                                [-L / 2, W / 2],
                                [L / 2, W / 2],
                                [L / 2, -W / 2],
                            ]
                        )
                        rotated_edge = edges @ (tran.T)
                        xy = (
                            np.array([clique_traj[n][i][t, 0], clique_traj[n][i][t, 1]])
                            + rotated_edge
                        )
                        if emphasized_nodes is not None and (n, i) in emphasized_nodes:
                            color = emph_color[emphasized_nodes.index((n, i))]
                        else:
                            color = cmap[clique_type[n][i].value]

                        patch = plt.Polygon(xy, fc=color, ec=color, lw=1.2, zorder=1)
                        ax.add_artist(patch)

                        # ax.add_artist(patch)

                    for j in range(i + 1, len(clique_type[n])):
                        ax.plot(
                            [clique_traj[n][i][t, 0], clique_traj[n][j][t, 0]],
                            [clique_traj[n][i][t, 1], clique_traj[n][j][t, 1]],
                            color="r",
                            linewidth=line_width,
                            alpha=line_alpha,
                        )

        else:

            for n in range(len(clique_type)):
                for i in range(len(clique_type[n])):
                    if clique_type[n][i] == "PEDESTRIAN":
                        if emphasized_nodes is not None and (n, i) in emphasized_nodes:
                            color = emph_color[emphasized_nodes.index((n, i))]
                        else:
                            color = cmap[clique_type[n][i].value]
                        circle = plt.Circle(
                            (clique_traj[n][i][t, 0], clique_traj[n][i][t, 1]),
                            clique_node_size[n][i][0],
                            facecolor=color,
                            edgecolor=color,
                            lw=circle_edge_width,
                            zorder=3,
                        )
                        ax.add_artist(circle)

                    elif clique_type[n][i] == "VEHICLE":

                        # coords = ts.transform([map_coords[i][0],map_coords[i][1]])
                        L = clique_node_size[n][i][0]
                        W = clique_node_size[n][i][1]
                        psi = clique_traj[n][i][t, 3]

                        tran = np.array(
                            [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
                        )
                        edges = np.array(
                            [
                                [-L / 2, -W / 2],
                                [-L / 2, W / 2],
                                [L / 2, W / 2],
                                [L / 2, -W / 2],
                            ]
                        )
                        rotated_edge = edges @ (tran.T)
                        xy = (
                            np.array([clique_traj[n][i][t, 0], clique_traj[n][i][t, 1]])
                            + rotated_edge
                        )
                        if emphasized_nodes is not None and (n, i) in emphasized_nodes:
                            color = emph_color[emphasized_nodes.index((n, i))]
                        else:
                            color = cmap[clique_type[n][i].value]

                        patch = plt.Polygon(xy, fc=color, ec=color, lw=1.2, zorder=1)

                        ax.add_artist(patch)

                        # ax.add_artist(patch)

                    for j in range(i + 1, len(clique_type[n])):
                        ax.plot(
                            [clique_traj[n][i][t, 0], clique_traj[n][j][t, 0]],
                            [clique_traj[n][i][t, 1], clique_traj[n][j][t, 1]],
                            color="r",
                            linewidth=line_width,
                            alpha=line_alpha,
                        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        fargs=(
            clique_type,
            clique_traj,
            clique_node_size,
            map,
        ),
        frames=nframe,
        interval=int(1000 * ts),
        blit=False,
        repeat=False,
    )

    if output:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=int(1.0 / ts), metadata=dict(artist="Me"), bitrate=1800)
        anim_name = output
        anim.save(anim_name, writer=writer)
    else:
        plt.show()



def sim_clique_prediction(
    results,
    map,
    dt,
    num_traj_show,
    output=None,
    limits=None,
    robot_plan=None,
    extra_node_info=None,
    focus_node=None,
    circle_edge_width=0.5,
    line_alpha=0.7,
    line_width=3,
):
    if output:
        matplotlib.use("Agg")

    nframe = len(results)

    if not map is None:
        map_shape = map.as_image().shape
        fig, ax = plt.subplots(figsize=(25, 25 / map_shape[1] * map_shape[0]))
    else:
        fig, ax = plt.subplots(figsize=(25, 25))

    if robot_plan is None:
        robot, traj_plan = None, None
    else:
        robot, traj_plan = robot_plan

    def animate(t, results, map, robot, traj_plan, extra_node_info):

        cmap = ["k", "tab:brown", "g", "tab:orange"]

        (
            clique_nodes,
            clique_state_history,
            clique_state_pred,
            clique_node_size,
            _,
            _,
        ) = results[t]

        ax.clear()
        ax.grid(False)
        if map is not None:

            ax.imshow(map.as_image(), origin="lower", alpha=0.3)
            scale = map.homography[0, 0]
            if not limits is None:
                xlim = [
                    (map_shape[1] - limits[0] * scale) / 2,
                    (map_shape[1] + limits[0] * scale) / 2,
                ]
                ylim = [
                    (map_shape[0] - limits[1] * scale) / 2,
                    (map_shape[0] + limits[1] * scale) / 2,
                ]
                plt.xlim(xlim)
                plt.ylim(ylim)
            # plot nodes

            for n in range(len(clique_nodes)):
                map_coords = list()
                for i in range(len(clique_nodes[n])):
                    map_coords.append(
                        map.to_map_points(clique_state_history[n][i][-1:, 0:2])[0]
                    )
                for i in range(len(clique_nodes[n])):
                    if clique_nodes[n][i].type == "PEDESTRIAN":
                        circle = plt.Circle(
                            (map_coords[i][0], map_coords[i][1]),
                            clique_node_size[n][i][0] * scale,
                            facecolor="m",
                            edgecolor="k",
                            lw=circle_edge_width,
                            zorder=3,
                        )
                        ax.add_artist(circle)

                    elif clique_nodes[n][i].type == "VEHICLE":

                        L = clique_nodes[n][i].length * scale
                        W = clique_nodes[n][i].width * scale
                        psi = clique_state_history[n][i][-1, 3]

                        tran = np.array(
                            [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
                        )
                        edges = np.array(
                            [
                                [-L / 2, -W / 2],
                                [-L / 2, W / 2],
                                [L / 2, W / 2],
                                [L / 2, -W / 2],
                            ]
                        )
                        rotated_edge = edges @ (tran.T)
                        xy = (
                            np.array([map_coords[i][0], map_coords[i][1]])
                            + rotated_edge
                        )

                        if clique_nodes[n][i].is_robot:
                            patch = plt.Polygon(xy, fc="k", ec="w", lw=1.2, zorder=1)
                        else:
                            if robot in clique_nodes[n]:
                                patch = plt.Polygon(xy, fc=cmap[i], zorder=1)
                            else:
                                patch = plt.Polygon(xy, fc="b", zorder=1)
                        ax.add_artist(patch)
                    if clique_nodes[n][i] == robot:
                        for j in range(traj_plan[t].shape[0]):
                            traj = map.to_map_points(traj_plan[t][j, :, 0:2])
                            ax.plot(
                                traj[:, 0],
                                traj[:, 1],
                                marker=".",
                                color="c",
                                linewidth=2 * line_width,
                                alpha=line_alpha,
                            )

                        if not limits is None:
                            xlim = [
                                map_coords[i][0] - limits[0] * scale / 2,
                                map_coords[i][0] + limits[0] * scale / 2,
                            ]
                            ylim = [
                                map_coords[i][1] - limits[1] * scale / 2,
                                map_coords[i][1] + limits[1] * scale / 2,
                            ]
                            plt.xlim(xlim)
                            plt.ylim(ylim)
                    else:
                        for k in range(
                            min(len(clique_state_pred[n][i]), num_traj_show)
                        ):
                            traj = map.to_map_points(clique_state_pred[n][i][k][:, 0:2])
                            traj = np.vstack((map_coords[i], traj))
                            ax.plot(
                                traj[:, 0],
                                traj[:, 1],
                                marker=".",
                                color=cmap[i],
                                linewidth=line_width,
                                alpha=line_alpha,
                            )
                    for j in range(i + 1, len(clique_nodes[n])):
                        ax.plot(
                            [map_coords[i][0], map_coords[j][0]],
                            [map_coords[i][1], map_coords[j][1]],
                            color="r",
                            linewidth=line_width,
                            alpha=line_alpha,
                        )

            if extra_node_info is not None:
                extra_node_type, extra_node_state, extra_node_size = extra_node_info[t]
                for i in range(len(extra_node_type)):
                    if extra_node_type[i] == "PEDESTRIAN":
                        coords = map.to_map_points(
                            np.expand_dims(extra_node_state[i][0:2], axis=0)
                        )[0]
                        circle = plt.Circle(
                            (coords[0], coords[1]),
                            extra_node_size[i][0] * scale,
                            facecolor="m",
                            edgecolor="k",
                            lw=circle_edge_width,
                            zorder=3,
                        )
                        ax.add_artist(circle)

                    elif extra_node_type[i] == "VEHICLE":
                        coords = map.to_map_points(
                            np.expand_dims(extra_node_state[i][0:2], axis=0)
                        )[0]
                        L = extra_node_size[i][0] * scale
                        W = extra_node_size[i][1] * scale
                        psi = extra_node_state[i][3]

                        tran = np.array(
                            [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
                        )
                        edges = np.array(
                            [
                                [-L / 2, -W / 2],
                                [-L / 2, W / 2],
                                [L / 2, W / 2],
                                [L / 2, -W / 2],
                            ]
                        )
                        rotated_edge = edges @ (tran.T)
                        xy = np.array([coords[0], coords[1]]) + rotated_edge

                        patch = plt.Polygon(xy, fc="b", zorder=1)
                        ax.add_artist(patch)
        else:
            if not limits is None:

                plt.xlim(limits[0:2])
                plt.ylim(limits[2:4])
            ts = ax.transData

            # plot nodes

            for n in range(len(clique_nodes)):
                coords = list()
                for i in range(len(clique_nodes[n])):
                    coords.append(clique_state_history[n][i][-1, 0:2])
                for i in range(len(clique_nodes[n])):
                    if clique_nodes[n][i].type == "PEDESTRIAN":

                        circle = plt.Circle(
                            (coords[i][0], coords[i][1]),
                            clique_node_size[n][i][0],
                            facecolor=cmap[clique_nodes[n][i].type.value],
                            edgecolor="k",
                            lw=circle_edge_width,
                            zorder=3,
                        )

                        ax.add_artist(circle)

                    elif clique_nodes[n][i].type == "VEHICLE":
                        coords = ts.transform([coords[i][0], coords[i][1]])
                        patch = plt.Rectangle(
                            (
                                coords[i][0] - clique_node_size[n][i][0] / 2,
                                coords[i][1] - clique_node_size[n][i][1] / 2,
                            ),
                            clique_node_size[n][i][0],
                            clique_node_size[n][i][1],
                            fc=cmap[clique_nodes[n][i].type.value],
                            zorder=1,
                        )
                        tr = matplotlib.transforms.Affine2D().rotate_around(
                            coords[0], coords[1], clique_state_history[n][i][-1, 3]
                        )
                        patch.set_transform(ts + tr)
                        ax.add_artist(patch)
                    for k in range(min(len(clique_state_pred[n][i]), num_traj_show)):
                        traj = clique_state_pred[n][i][k][:, 0:2]
                        traj = np.vstack((coords[i], traj))
                        ax.plot(
                            traj[:, 0],
                            traj[:, 1],
                            color=cmap[clique_nodes[n][i].type.value],
                            linewidth=line_width,
                            alpha=line_alpha,
                        )
                    for j in range(i + 1, len(clique_nodes[n])):
                        ax.plot(
                            [coords[i][0], coords[j][0]],
                            [coords[i][1], coords[j][1]],
                            color="r",
                            linewidth=line_width,
                            alpha=line_alpha,
                        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        fargs=(
            results,
            map,
            robot,
            traj_plan,
            extra_node_info,
        ),
        frames=nframe,
        interval=(1000 * dt),
        blit=False,
        repeat=False,
    )

    if output:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=int(1.0 / dt), metadata=dict(artist="Me"), bitrate=1800)
        anim_name = output
        anim.save(anim_name, writer=writer)
    else:
        plt.show()


def sim_IRL(
    scene,
    time_steps,
    weight_dicts,
    dt,
    hyperparams,
    output=None,
    limits=None,
    focus_node=None,
    circle_edge_width=0.5,
    line_alpha=0.7,
    line_width=3,
):
    if output:
        matplotlib.use("Agg")
    num_nodes = len(scene.nodes)
    T = scene.timesteps
    presence_table = np.zeros([num_nodes, T], dtype=np.bool)

    for i in range(0, num_nodes):
        presence_table[i][
            scene.nodes[i].first_timestep : scene.nodes[i].last_timestep
        ] = True

    nframe = len(time_steps)

    if not scene.map is None:
        map_shape = scene.map["VISUALIZATION"].as_image().shape
        fig, ax = plt.subplots(figsize=(25, 25 / map_shape[1] * map_shape[0]))
    else:
        fig, ax = plt.subplots(figsize=(25, 25))

    state = hyperparams["state"]

    def animate(t, time_steps, scene, presence_table, weight_dicts):

        cmap = ["k", "b", "m", "g"]
        active_idx = np.where((presence_table[:, time_steps[t]] == True))[0]

        ax.clear()
        if scene.map is not None:

            ax.imshow(scene.map["VISUALIZATION"].as_image(), origin="lower", alpha=0.5)
            scale = scene.map["VISUALIZATION"].homography[0, 0]
            if not limits is None:
                xlim = [
                    (map_shape[1] - limits[0] * scale) / 2,
                    (map_shape[1] + limits[0] * scale) / 2,
                ]
                ylim = [
                    (map_shape[0] - limits[1] * scale) / 2,
                    (map_shape[0] + limits[1] * scale) / 2,
                ]
                plt.xlim(xlim)
                plt.ylim(ylim)
            ts = ax.transData
            # plot nodes

            map_coords = list()
            for i in range(active_idx.shape[0]):
                node = scene.nodes[active_idx[i]]
                x = node.get(
                    np.array([time_steps[t], time_steps[t] + 1]),
                    state[node.type],
                    padding=0.0,
                )
                map_coords.append(scene.map[node.type].to_map_points(x[-1:, 0:2])[0])

            for i in range(active_idx.shape[0]):
                node = scene.nodes[active_idx[i]]
                x = node.get(
                    np.array([time_steps[t], time_steps[t] + 1]),
                    state[node.type],
                    padding=0.0,
                )
                if node.type == "PEDESTRIAN":
                    circle = plt.Circle(
                        (map_coords[i][0], map_coords[i][1]),
                        node.length * scale,
                        facecolor=cmap[node.type.value],
                        edgecolor="k",
                        lw=circle_edge_width,
                        zorder=3,
                    )
                    ax.add_artist(circle)

                elif node.type == "VEHICLE":
                    L = node.length * scale
                    W = node.width * scale
                    psi = x[-1, 3]

                    tran = np.array(
                        [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
                    )
                    edges = np.array(
                        [
                            [-L / 2, -W / 2],
                            [-L / 2, W / 2],
                            [L / 2, W / 2],
                            [L / 2, -W / 2],
                        ]
                    )
                    rotated_edge = edges @ (tran.T)
                    xy = np.array([map_coords[i][0], map_coords[i][1]]) + rotated_edge

                    if node == focus_node:
                        patch = plt.Polygon(
                            xy, fc=cmap[node.type.value], ec="w", lw=1.2, zorder=1
                        )

                    else:
                        patch = plt.Polygon(xy, fc=cmap[node.type.value], zorder=1)
                    ax.add_artist(patch)

                if node == focus_node:
                    nbs = {
                        pair: weight_dicts[t][pair]
                        for pair in weight_dicts[t].keys()
                        if pair[0] == focus_node
                    }
                    for j in range(active_idx.shape[0]):
                        nb = scene.nodes[active_idx[j]]

                        if nb.type == "VEHICLE" and (node, nb) in weight_dicts[t]:
                            # pdb.set_trace()

                            w1 = weight_dicts[t][(node, nb)]
                            if (nb, node) in weight_dicts[t]:
                                w2 = weight_dicts[t][(nb, node)]
                            else:
                                w2 = 0
                            if w1 != 0 or w2 != 0:

                                middlepoint = map_coords[i] + np.array(
                                    w1 / (w1 + w2)
                                ) * (map_coords[j] - map_coords[i])
                                ax.plot(
                                    [map_coords[i][0], middlepoint[0]],
                                    [map_coords[i][1], middlepoint[1]],
                                    color="r",
                                    linewidth=line_width,
                                    alpha=line_alpha,
                                )
                                ax.plot(
                                    [middlepoint[0], map_coords[j][0]],
                                    [middlepoint[1], map_coords[j][1]],
                                    color="g",
                                    linewidth=line_width,
                                    alpha=line_alpha,
                                )
                # for j in range(i+1,len(clique_nodes[n])):
                #     ax.plot([map_coords[i][0],map_coords[j][0]], [map_coords[i][1],map_coords[j][1]],
                #             color='r',
                #             linewidth=line_width, alpha=line_alpha)
        # else:
        #     if not limits is None:

        #         plt.xlim(limits[0:2])
        #         plt.ylim(limits[2:4])
        #     ts = ax.transData

        #     # plot nodes

        #     for n in range(len(clique_nodes)):
        #         coords = list()
        #         for i in range(len(clique_nodes[n])):
        #             coords.append(clique_state_history[n][i][-1,0:2])
        #         for i in range(len(clique_nodes[n])):
        #             if clique_nodes[n][i].type=='PEDESTRIAN':

        #                 circle = plt.Circle((coords[i][0],
        #                              coords[i][1]),
        #                              clique_node_size[n][i][0],
        #                              facecolor=cmap[clique_nodes[n][i].type.value],
        #                              edgecolor='k',
        #                              lw=circle_edge_width,
        #                              zorder=3)

        #                 ax.add_artist(circle)

        #             elif clique_nodes[n][i].type=='VEHICLE':
        #                 coords = ts.transform([coords[i][0],coords[i][1]])
        #                 patch = plt.Rectangle((coords[i][0]-clique_node_size[n][i][0]/2,coords[i][1]-clique_node_size[n][i][1]/2), clique_node_size[n][i][0],clique_node_size[n][i][1], fc=cmap[clique_nodes[n][i].type.value], zorder=1)
        #                 tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], clique_state_history[n][i][-1,3])
        #                 patch.set_transform(ts+tr)
        #                 ax.add_artist(patch)
        #             for k in range(min(len(clique_state_pred[n][i]),num_traj_show)):
        #                 traj = clique_state_pred[n][i][k][:,0:2]
        #                 traj = np.vstack((coords[i],traj))
        #                 ax.plot(traj[:,0], traj[:,1],
        #                         color=cmap[clique_nodes[n][i].type.value],
        #                         linewidth=line_width, alpha=line_alpha)
        #             for j in range(i+1,len(clique_nodes[n])):
        #                 ax.plot([coords[i][0],coords[j][0]], [coords[i][1],coords[j][1]],
        #                         color='r',
        #                         linewidth=line_width, alpha=line_alpha)

    anim = animation.FuncAnimation(
        fig,
        animate,
        fargs=(
            time_steps,
            scene,
            presence_table,
            weight_dicts,
        ),
        frames=nframe,
        interval=50,
        blit=False,
        repeat=False,
    )

    if output:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=int(1.0 / dt), metadata=dict(artist="Me"), bitrate=1800)
        anim_name = output
        anim.save(anim_name, writer=writer)
    else:
        plt.show()


def visualize_distribution(
    ax, prediction_distribution_dict, map=None, pi_threshold=0.05, **kwargs
):
    if map is not None:
        ax.imshow(map.as_image(), origin="lower", alpha=0.5)

    for node, pred_dist in prediction_distribution_dict.items():
        if pred_dist.mus.shape[:2] != (1, 1):
            return

        means = pred_dist.mus.squeeze().cpu().numpy()
        covs = pred_dist.get_covariance_matrix().squeeze().cpu().numpy()
        pis = pred_dist.pis_cat_dist.probs.squeeze().cpu().numpy()

        for timestep in range(means.shape[0]):
            for z_val in range(means.shape[1]):
                mean = means[timestep, z_val]
                covar = covs[timestep, z_val]
                pi = pis[timestep, z_val]

                if pi < pi_threshold:
                    continue

                v, w = linalg.eigh(covar)
                v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180.0 * angle / np.pi  # convert to degrees
                ell = patches.Ellipse(
                    mean,
                    v[0],
                    v[1],
                    180.0 + angle,
                    color="blue" if node.type.name == "VEHICLE" else "orange",
                )
                ell.set_edgecolor(None)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(pi / 10)
                ax.add_artist(ell)


def round_2pi(x):
    while x > np.pi:
        x -= 2 * np.pi
    while x < -np.pi:
        x += 2 * np.pi
    return x
