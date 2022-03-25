import torch
import torch.nn.utils.rnn as rnn
from enum import Enum
import functools
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import pdb
from scipy.interpolate import interp1d


from numpy import sin, cos


def Rot_matr_2d(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


def Rot_matr_2d_tensor(theta):
    return torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )


class simplelinear(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim=[64, 32], scale=None):
        super(simplelinear, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.hidden_layers = len(hidden_dim)
        self.fhidden = [None] * (self.hidden_layers - 1)

        if isinstance(scale, torch.Tensor):
            self.scale = scale.to(self.device)
        else:
            self.scale = scale

        for i in range(1, self.hidden_layers):
            self.fhidden[i - 1] = nn.Linear(hidden_dim[i - 1], hidden_dim[i]).to(
                self.device
            )
        self.f1 = nn.Linear(input_dim, hidden_dim[0]).to(self.device)
        self.f2 = nn.Linear(hidden_dim[-1], output_dim).to(self.device)

    def forward(self, x):
        hidden = self.f1(x)
        for i in range(1, self.hidden_layers):
            hidden = self.fhidden[i - 1](F.relu(hidden))
        if not self.scale is None:
            return torch.tanh(self.f2(F.relu(hidden))) * self.scale
        else:
            return self.f2(F.relu(hidden))


def list2onehot(listofwords):
    onehots = torch.eye(len(listofwords))
    map = {}
    for i in range(len(listofwords)):
        map[listofwords[i]] = onehots[i]
    return map


class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, decay=1.0):
    # Lambda function to calculate the LR
    lr_lambda = (
        lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize) * decay ** it
    )

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x))

    return lr_lambda


def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


def exp_anneal(anneal_kws):
    device = anneal_kws["device"]
    start = torch.tensor(anneal_kws["start"], device=device)
    finish = torch.tensor(anneal_kws["finish"], device=device)
    rate = torch.tensor(anneal_kws["rate"], device=device)
    return lambda step: finish - (finish - start) * torch.pow(
        rate, torch.tensor(step, dtype=torch.float, device=device)
    )


def sigmoid_anneal(anneal_kws):
    device = anneal_kws["device"]
    start = torch.tensor(anneal_kws["start"], device=device)
    finish = torch.tensor(anneal_kws["finish"], device=device)
    center_step = torch.tensor(
        anneal_kws["center_step"], device=device, dtype=torch.float
    )
    steps_lo_to_hi = torch.tensor(
        anneal_kws["steps_lo_to_hi"], device=device, dtype=torch.float
    )
    return lambda step: start + (finish - start) * torch.sigmoid(
        (torch.tensor(float(step), device=device) - center_step)
        * (1.0 / steps_lo_to_hi)
    )


class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [
            lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
        ]


def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()


def run_lstm_on_variable_length_seqs(
    lstm_module,
    original_seqs,
    lower_indices=None,
    upper_indices=None,
    total_length=None,
    batch_first=True,
):

    if batch_first:
        bs, tf = original_seqs.shape[:2]
    else:
        tf, bs = original_seqs.shape[:2]
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
    if total_length is None:
        total_length = max(upper_indices) + 1

    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1
    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        if batch_first:
            pad_list.append(original_seqs[i, lower_indices[i] : seq_len])
        else:
            pad_list.append(original_seqs[lower_indices[i] : seq_len, i])

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)

    output, _ = rnn.pad_packed_sequence(
        packed_output, batch_first=batch_first, total_length=total_length
    )

    return output, (h_n, c_n)


def extract_subtensor_per_batch_element(tensor, indices):
    batch_idxs = torch.arange(start=0, end=len(indices))

    batch_idxs = batch_idxs[~torch.isnan(indices)]
    indices = indices[~torch.isnan(indices)]
    if indices.size == 0:
        return None
    else:
        indices = indices.long()
    if tensor.is_cuda:
        batch_idxs = batch_idxs.to(tensor.get_device())
        indices = indices.to(tensor.get_device())
    return tensor[batch_idxs, indices]


def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def discrete_KL_div(log_pi1, log_pi2):
    pi1 = torch.exp(log_pi1)
    pi1_sum = torch.unsqueeze(torch.sum(pi1, dim=-1), dim=-1).repeat(pi1.shape[-1])
    if pi1.shape[-1] == 1:
        pi1 = torch.ones_like(pi1).to(pi1.device)
    else:
        pi1 = pi1 / pi1_sum
    log_pi1_normalized = torch.log(pi1)

    pi2 = torch.exp(log_pi2)
    pi2_sum = torch.unsqueeze(torch.sum(pi2, dim=-1), dim=-1).repeat(pi2.shape[-1])
    if pi2.shape[-1] == 1:
        pi2 = torch.ones_like(pi2).to(pi2.device)
    else:
        pi2 = pi2 / pi2_sum
    log_pi2_normalized = torch.log(pi2)
    return torch.sum(pi1 * (log_pi1_normalized - log_pi2_normalized))


class PED_PED_encode(nn.Module):
    def __init__(self, obs_enc_dim, device, hidden_dim=None):
        super(PED_PED_encode, self).__init__()
        self.device = device
        if hidden_dim is None:
            self.FC = simplelinear(10, obs_enc_dim, device=device)
        else:
            self.FC = simplelinear(10, obs_enc_dim, hidden_dim, device=device)

    def forward(self, x1, x2, size1, size2):
        deltax = x2[..., 0:2] - x1[..., 0:2]
        input = torch.cat((deltax, x1[..., 2:4], x2[..., 2:4], size1, size2), dim=-1)
        return self.FC(input)


class PED_VEH_encode(nn.Module):
    def __init__(self, obs_enc_dim, device, hidden_dim=None):
        super(PED_VEH_encode, self).__init__()
        self.device = device
        if hidden_dim is None:
            self.FC = simplelinear(10, obs_enc_dim, device=device)
        else:
            self.FC = simplelinear(10, obs_enc_dim, hidden_dim, device=device)

    def forward(self, x1, x2, size1, size2):
        deltax = x2[..., 0:2] - x1[..., 0:2]
        veh_vel = torch.cat(
            (
                torch.unsqueeze(x2[..., 2] * torch.cos(x2[..., 3]), dim=-1),
                torch.unsqueeze(x2[..., 2] * torch.sin(x2[..., 3]), dim=-1),
            ),
            dim=-1,
        )
        input = torch.cat((deltax, x1[..., 2:4], veh_vel, size1, size2), dim=-1)
        return self.FC(input)


class VEH_PED_encode(nn.Module):
    def __init__(self, obs_enc_dim, device, hidden_dim=None):
        super(VEH_PED_encode, self).__init__()
        self.device = device
        if hidden_dim is None:
            self.FC = simplelinear(9, obs_enc_dim, device=device)
        else:
            self.FC = simplelinear(9, obs_enc_dim, hidden_dim, device=device)

    def forward(self, x1, x2, size1, size2):
        dx0 = x2[..., 0:2] - x1[..., 0:2]
        theta = x1[..., 3]
        dx = torch.cat(
            (
                torch.unsqueeze(
                    dx0[..., 0] * torch.cos(theta) + torch.sin(theta) * dx0[..., 1],
                    dim=-1,
                ),
                torch.unsqueeze(
                    dx0[..., 1] * torch.cos(theta) - torch.sin(theta) * dx0[..., 0],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        dv = torch.cat(
            (
                torch.unsqueeze(
                    x2[..., 2] * torch.cos(theta)
                    + torch.sin(theta) * x2[..., 3]
                    - x1[..., 2],
                    dim=-1,
                ),
                torch.unsqueeze(
                    x2[..., 3] * torch.cos(theta) - torch.sin(theta) * x2[..., 2],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        input = torch.cat(
            (dx, torch.unsqueeze(x1[..., 2], dim=-1), dv, size1, size2), dim=-1
        )
        return self.FC(input)


class VEH_VEH_encode(nn.Module):
    def __init__(self, obs_enc_dim, device, hidden_dim=None):
        super(VEH_VEH_encode, self).__init__()
        self.device = device
        if hidden_dim is None:
            self.FC = simplelinear(11, obs_enc_dim, device=device)
        else:
            self.FC = simplelinear(11, obs_enc_dim, hidden_dim, device=device)

    def forward(self, x1, x2, size1, size2):
        dx0 = x2[..., 0:2] - x1[..., 0:2]
        theta = x1[..., 3]
        dx = torch.cat(
            (
                torch.unsqueeze(
                    dx0[..., 0] * torch.cos(theta) + torch.sin(theta) * dx0[..., 1],
                    dim=-1,
                ),
                torch.unsqueeze(
                    dx0[..., 1] * torch.cos(theta) - torch.sin(theta) * dx0[..., 0],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        dtheta = x2[..., 3] - x1[..., 3]
        dv = torch.cat(
            (
                torch.unsqueeze(x2[..., 2] * torch.cos(dtheta) - x1[..., 2], dim=-1),
                torch.unsqueeze(torch.sin(dtheta) * x2[..., 2], dim=-1),
            ),
            dim=-1,
        )
        input = torch.cat(
            (
                dx,
                torch.unsqueeze(x1[..., 2], dim=-1),
                dv,
                torch.unsqueeze(torch.cos(dtheta), dim=-1),
                torch.unsqueeze(torch.sin(dtheta), dim=-1),
                size1,
                size2,
            ),
            dim=-1,
        )
        return self.FC(input)


def PED_rel_state(x, x0):
    rel_x = torch.clone(x)
    rel_x[..., 0:2] -= x0[..., 0:2]
    return rel_x


def VEH_rel_state(x, x0):
    rel_XY = x[..., 0:2] - x0[..., 0:2]
    theta = x0[..., 3]
    rel_x = torch.stack(
        [
            rel_XY[..., 0] * torch.cos(theta) + rel_XY[..., 1] * torch.sin(theta),
            rel_XY[..., 1] * torch.cos(theta) - rel_XY[..., 0] * torch.sin(theta),
            x[..., 2],
            x[..., 3] - x0[..., 3],
        ],
        dim=-1,
    )
    rel_x[..., 3] = batch_round_pi(rel_x[..., 3])
    return rel_x


class PED_pre_encode(nn.Module):
    def __init__(self, enc_dim, device, hidden_dim=None, use_lane_info=False):
        super(PED_pre_encode, self).__init__()
        self.device = device
        if hidden_dim is None:
            self.FC = simplelinear(4, enc_dim, device=device)
        else:
            self.FC = simplelinear(4, enc_dim, hidden_dim, device=device)

    def forward(self, x):
        return self.FC(x)


class VEH_pre_encode(nn.Module):
    def __init__(self, enc_dim, device, hidden_dim=None, use_lane_info=False):
        super(VEH_pre_encode, self).__init__()
        self.device = device
        self.use_lane_info = use_lane_info
        if hidden_dim is None:
            if use_lane_info:
                self.FC = simplelinear(8, enc_dim, device=device)
            else:
                self.FC = simplelinear(5, enc_dim, device=device)
        else:
            if use_lane_info:
                self.FC = simplelinear(8, enc_dim, hidden_dim, device=device)
            else:
                self.FC = simplelinear(5, enc_dim, hidden_dim, device=device)

    def forward(self, x):
        if self.use_lane_info:
            input = torch.cat(
                (
                    x[..., 0:3],
                    torch.cos(x[..., 3:4]),
                    torch.sin(x[..., 3:4]),
                    x[..., 4:5],
                    torch.cos(x[..., 5:6]),
                    torch.sin(x[..., 5:6]),
                ),
                dim=-1,
            )
        else:
            input = torch.cat(
                (x[..., 0:3], torch.cos(x[..., 3:]), torch.sin(x[..., 3:])), dim=-1
            )
        return self.FC(input)


def PED_PED_collision(p1, p2, S1, S2, device, alpha=3, return_dis=False, offset=0.0):
    if isinstance(p1, torch.Tensor):
        if return_dis:
            return (
                torch.linalg.norm(p1[..., 0:2] - p2[..., 0:2], dim=-1)
                - (S1[..., 0] + S2[..., 0]) / 2
            )
        else:
            mask = torch.logical_or(
                torch.linalg.norm(p1[..., 2:4], dim=-1) > 0.1,
                torch.linalg.norm(p2[..., 2:4], dim=-1) > 0.1,
            ).detach()
            return (
                torch.sigmoid(
                    alpha
                    * (
                        (S1[..., 0] + S2[..., 0]) / 2
                        - offset
                        - torch.linalg.norm(p1[..., 0:2] - p2[..., 0:2])
                    )
                )
                * mask
            )
    elif isinstance(p1, np.ndarray):
        if return_dis:
            return (
                np.linalg.norm(p1[..., 0:2] - p2[..., 0:2], axis=-1)
                - (S1[..., 0] + S2[..., 0]) / 2
            )
        else:
            mask = np.logical_or(
                np.linalg.norm(p1[..., 2:4], axis=-1) > 0.1,
                np.linalg.norm(p2[..., 2:4], axis=-1) > 0.1,
            )
            return (
                1
                / (
                    1
                    + np.exp(
                        -alpha
                        * (
                            (S1[..., 0] + S2[..., 0]) / 2
                            - offset
                            - np.linalg.norm(p1[..., 0:2] - p2[..., 0:2])
                        )
                    )
                )
                * mask
            )
    else:
        raise NotImplementedError


def VEH_VEH_collision(
    p1, p2, S1, S2, device, alpha=5, return_dis=False, offsetX=1.0, offsetY=0.3
):
    if isinstance(p1, torch.Tensor):

        mask = torch.logical_or(
            torch.abs(p1[..., 2]) > 0.1, torch.abs(p2[..., 2]) > 0.1
        ).detach()
        cornersX = torch.kron(
            S1[..., 0] + offsetX, torch.tensor([0.5, 0.5, -0.5, -0.5]).to(device)
        )
        cornersY = torch.kron(
            S1[..., 1] + offsetY, torch.tensor([0.5, -0.5, 0.5, -0.5]).to(device)
        )
        corners = torch.stack([cornersX, cornersY], dim=-1)
        theta1 = p1[..., 3]
        theta2 = p2[..., 3]
        dx = (p1[..., 0:2] - p2[..., 0:2]).repeat_interleave(4, dim=-2)
        delta_x1 = batch_rotate_2D(corners, theta1.repeat_interleave(4, dim=-1)) + dx
        delta_x2 = batch_rotate_2D(delta_x1, -theta2.repeat_interleave(4, dim=-1))
        dis = torch.maximum(
            torch.abs(delta_x2[..., 0]) - 0.5 * S2[..., 0].repeat_interleave(4, dim=-1),
            torch.abs(delta_x2[..., 1]) - 0.5 * S2[..., 1].repeat_interleave(4, dim=-1),
        ).view(*S1.shape[:-1], 4)
        min_dis, _ = torch.min(dis, dim=-1)
        if return_dis:
            return min_dis
        else:
            score = torch.sigmoid(alpha * -min_dis) * mask
            return score
    elif isinstance(p1, np.ndarray):
        mask = np.logical_or(np.abs(p1[..., 2]) > 0.1, np.abs(p2[..., 2]) > 0.1)
        cornersX = np.kron(S1[..., 0] + offsetX, np.array([0.5, 0.5, -0.5, -0.5]))
        cornersY = np.kron(S1[..., 1] + offsetY, np.array([0.5, -0.5, 0.5, -0.5]))
        corners = np.concatenate((cornersX, cornersY), axis=-1)
        theta1 = p1[..., 3]
        theta2 = p2[..., 3]
        dx = (p1[..., 0:2] - p2[..., 0:2]).repeat(4, axis=-2)
        delta_x1 = batch_rotate_2D(corners, theta1.repeat(4, axis=-1)) + dx
        delta_x2 = batch_rotate_2D(delta_x1, -theta2.repeat(4, axis=-1))
        dis = np.maximum(
            np.abs(delta_x2[..., 0]) - 0.5 * S2[..., 0].repeat(4, axis=-1),
            np.abs(delta_x2[..., 1]) - 0.5 * S2[..., 1].repeat(4, axis=-1),
        ).reshape(*S1.shape[:-1], 4)
        min_dis = np.min(dis, axis=-1)
        if return_dis:
            return min_dis
        else:
            score = 1 / (1 + np.exp(alpha * min_dis)) * mask
            return score
    else:
        raise NotImplementedError


def VEH_PED_collision(p1, p2, S1, S2, device, alpha=5, return_dis=False, offset=0.5):
    if isinstance(p1, torch.Tensor):

        mask = torch.logical_or(
            torch.abs(p1[..., 2]) > 0.1, torch.linalg.norm(p2[..., 2:4], dim=-1) > 0.1
        ).detach()
        theta = p1[..., 3]
        dx = batch_rotate_2D(p2[..., 0:2] - p1[..., 0:2], -theta)
        if return_dis:
            return torch.maximum(
                torch.abs(dx[..., 0]) - S1[..., 0] / 2 - S2[..., 0] / 2,
                torch.abs(dx[..., 1]) - S1[..., 1] / 2 - S2[..., 0] / 2,
            )
        else:
            return (
                torch.sigmoid(
                    alpha
                    * torch.minimum(
                        S1[..., 0] / 2
                        + S2[..., 0] / 2
                        - offset
                        - torch.abs(dx[..., 0]),
                        S1[..., 1] / 2
                        + S2[..., 0] / 2
                        - offset
                        - torch.abs(dx[..., 1]),
                    )
                )
                * mask
            )
    elif isinstance(p1, np.ndarray):

        mask = np.logical_or(
            np.abs(p1[..., 2]) > 0.1, np.linalg.norm(p2[..., 2:4], axis=-1) > 0.1
        )
        theta = p1[..., 3]
        dx = batch_rotate_2D(p2[..., 0:2] - p1[..., 0:2], -theta)
        if return_dis:
            return np.maximum(
                np.abs(dx[..., 0]) - S1[..., 0] / 2 - S2[..., 0] / 2,
                np.abs(dx[..., 1]) - S1[..., 1] / 2 - S2[..., 0] / 2,
            )
        else:
            return (
                1
                / (
                    1
                    + np.exp(
                        -alpha
                        * np.minimum(
                            S1[..., 0] / 2
                            + S2[..., 0] / 2
                            - offset
                            - np.abs(dx[..., 0]),
                            S1[..., 1] / 2
                            + S2[..., 0] / 2
                            - offset
                            - np.abs(dx[..., 1]),
                        )
                    )
                )
                * mask
            )
    else:
        raise NotImplementedError


def PED_VEH_collision(p1, p2, S1, S2, device, alpha=5, return_dis=False, offset=0.5):
    if isinstance(p1, torch.Tensor):
        mask = torch.logical_or(
            torch.abs(p2[..., 2]) > 0.1, torch.linalg.norm(p1[..., 2:4], dim=-1) > 0.1
        ).detach()
        theta = p2[..., 3]
        dx = batch_rotate_2D(p1[..., 0:2] - p2[..., 0:2], -theta)
        if return_dis:
            return torch.maximum(
                torch.abs(dx[..., 0]) - S1[..., 0] / 2 - S2[..., 0] / 2,
                torch.abs(dx[..., 1]) - S1[..., 0] / 2 - S2[..., 1] / 2,
            )
        else:
            return (
                torch.sigmoid(
                    alpha
                    * torch.minimum(
                        S1[..., 0] / 2
                        + S2[..., 0] / 2
                        - offset
                        - torch.abs(dx[..., 0]),
                        S1[..., 0] / 2
                        + S2[..., 1] / 2
                        - offset
                        - torch.abs(dx[..., 1]),
                    )
                )
                * mask
            )
    elif isinstance(p1, np.ndarray):
        mask = np.logical_or(
            np.abs(p2[..., 2]) > 0.1, np.linalg.norm(p1[..., 2:4], axis=-1) > 0.1
        )
        theta = p2[..., 3]
        dx = batch_rotate_2D(p1[..., 0:2] - p2[..., 0:2], -theta)
        if return_dis:
            return np.maximum(
                np.abs(dx[..., 0]) - S1[..., 0] / 2 - S2[..., 0] / 2,
                np.abs(dx[..., 1]) - S1[..., 0] / 2 - S2[..., 1] / 2,
            )
        else:
            return (
                1
                / (
                    1
                    + np.exp(
                        -alpha
                        * np.minimum(
                            S1[..., 0] / 2
                            + S2[..., 0] / 2
                            - offset
                            - np.abs(dx[..., 0]),
                            S1[..., 0] / 2
                            + S2[..., 1] / 2
                            - offset
                            - np.abs(dx[..., 1]),
                        )
                    )
                )
                * mask
            )
    else:
        raise NotImplementedError


def PED_no_control(x, device="cpu"):
    if isinstance(x, np.ndarray):
        return np.zeros([*x.shape[:-1], 2])
    elif isinstance(x, torch.Tensor):
        return torch.zeros([*x.shape[:-1], 2]).to(device)


def VEH_no_control(x, device="cpu"):
    if isinstance(x, np.ndarray):
        return np.zeros([*x.shape[:-1], 2])
    elif isinstance(x, torch.Tensor):
        return torch.zeros([*x.shape[:-1], 2]).to(device)


def VEH_LK_control(x, line, Ky, Kpsi, device="cpu"):
    delta_y, delta_psi, ref_pt = batch_proj(x[..., [0, 1, 3]], line)
    if isinstance(x, np.ndarray):
        steer = np.clip(
            -delta_y * Ky / np.maximum(x[..., 2], 1.0) - delta_psi * Kpsi, -0.3, 0.3
        )
        acce = -4.0 * (x[..., 2] > 2) + (-x[..., 2]) * (x[..., 2] <= 2)
        return np.hstack((acce, steer))
    elif isinstance(x, torch.Tensor):
        steer = torch.clamp(
            -delta_y.to(device) * Ky / torch.clamp(x[..., 2:3], min=1.0)
            - delta_psi.to(device) * Kpsi,
            min=-0.3,
            max=0.3,
        )
        # acce = torch.zeros_like(steer).to(device)
        acce = torch.unsqueeze(
            -4.0 * (x[..., 2] > 2) + (-x[..., 2]) * (x[..., 2] <= 2), dim=-1
        ).to(device)
        # if (torch.abs(steer)>3).any():
        #     pdb.set_trace()
        return torch.cat((acce, steer), dim=-1)


def batch_proj(x, line):
    # x:[batch,n], line:[batch,N,n]
    line_length = line.shape[-2]
    batch_dim = x.ndim - 1
    if isinstance(x, torch.Tensor):
        delta_x = line[..., 0:2] - torch.unsqueeze(x[..., 0:2], dim=-2).repeat(
            *([1] * batch_dim), line_length, 1
        )
        dis = torch.linalg.norm(delta_x, axis=-1)
        idx = torch.argmin(dis, dim=-1)
        idx = idx.view(*line.shape[:-2], 1, 1).repeat(
            *([1] * (batch_dim + 1)), line.shape[-1]
        )
        line_min = torch.squeeze(torch.gather(line, -2, idx), dim=-2)
        dx = x[..., 0] - line_min[..., 0]
        dy = x[..., 1] - line_min[..., 1]
        delta_y = -dx * torch.sin(line_min[..., 2]) + dy * torch.cos(line_min[..., 2])
        delta_x = dx * torch.cos(line_min[..., 2]) + dy * torch.sin(line_min[..., 2])
        ref_pts = torch.stack(
            [
                line_min[..., 0] + delta_x * torch.cos(line_min[..., 2]),
                line_min[..., 1] + delta_x * torch.sin(line_min[..., 2]),
                line_min[..., 2],
            ],
            dim=-1,
        )
        delta_psi = batch_round_pi(x[..., 2] - line_min[..., 2])
        return (
            torch.unsqueeze(delta_y, dim=-1),
            torch.unsqueeze(delta_psi, dim=-1),
            ref_pts,
        )
    elif isinstance(x, np.ndarray):
        delta_x = line[..., 0:2] - np.repeat(
            x[..., np.newaxis, 0:2], line_length, axis=-2
        )
        dis = np.linalg.norm(delta_x, axis=-1)
        idx = np.argmin(dis, axis=-1)
        idx = idx.reshape(*line.shape[:-2], 1, 1).repeat(line.shape[-1], axis=-1)
        line_min = np.squeeze(np.take_along_axis(line, idx, axis=-2), axis=-2)
        dx = x[..., 0] - line_min[..., 0]
        dy = x[..., 1] - line_min[..., 1]
        delta_y = -dx * np.sin(line_min[..., 2]) + dy * np.cos(line_min[..., 2])
        delta_x = dx * np.cos(line_min[..., 2]) + dy * np.sin(line_min[..., 2])
        line_min[..., 0] += delta_x * np.cos(line_min[..., 2])
        line_min[..., 1] += delta_x * np.sin(line_min[..., 2])
        delta_psi = batch_round_pi(x[..., 2] - line_min[..., 2])
        return (
            np.expand_dims(delta_y, axis=-1),
            np.expand_dims(delta_psi, axis=-1),
            line_min,
        )


def obtain_ref(line, x, v, N, dt):
    line_length = line.shape[0]
    delta_x = line[..., 0:2] - np.repeat(x[..., np.newaxis, 0:2], line_length, axis=-2)
    dis = np.linalg.norm(delta_x, axis=-1)
    idx = np.argmin(dis, axis=-1)
    line_min = line[idx]
    dx = x[0] - line_min[0]
    dy = x[1] - line_min[1]
    delta_y = -dx * np.sin(line_min[2]) + dy * np.cos(line_min[2])
    delta_x = dx * np.cos(line_min[2]) + dy * np.sin(line_min[2])
    refx0 = np.array(
        [
            line_min[0] + delta_x * np.cos(line_min[2]),
            line_min[1] + delta_x * np.sin(line_min[2]),
            line_min[2],
        ]
    )
    s = [np.linalg.norm(line[idx + 1, 0:2] - refx0[0:2])]
    for i in range(idx + 2, line_length):
        s.append(s[-1] + np.linalg.norm(line[i, 0:2] - line[i - 1, 0:2]))
    f = interp1d(
        np.array(s),
        line[idx + 1 :],
        kind="linear",
        axis=0,
        copy=True,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    s1 = v * np.arange(1, N + 1) * dt
    refx = f(s1)

    return refx


def safety_measure(traj1, traj2, S1, S2, col_fun, device=None, gamma=1):
    T = traj1.shape[0]
    if isinstance(traj1, torch.Tensor):

        dis = col_fun(
            traj1,
            traj2,
            torch.unsqueeze(S1, dim=0).repeat_interleave(T, dim=0),
            torch.unsqueeze(S2, dim=0).repeat_interleave(T, dim=0),
            return_dis=True,
        )

        # return torch.min(torch.exp(-gamma*torch.arange(T)*dt)*dis,dim=0)
        return torch.min(dis, dim=0)
    elif isinstance(traj1, np.ndarray):
        dis = col_fun(
            traj1,
            traj2,
            np.expand_dims(S1, axis=0).repeat(T, axis=0),
            np.expand_dims(S2, axis=0).repeat(T, axis=0),
            return_dis=True,
        )
        # return np.min(np.exp(-gamma*np.arange(T)*dt)*dis,axis=0)
        return np.min(dis, axis=0)
    else:
        raise NotImplementedError


def propagate_traj(x0, dyn, controller, dt, T, device="cpu"):
    u0 = controller(x0)
    x_dim = x0.shape[-1]
    u_dim = u0.shape[-1]
    if isinstance(x0, np.ndarray):
        x_traj = np.zeros([T + 1, *x0.shape])
        u_traj = np.zeros([T, *x0.shape[:-1], u_dim])
        x_traj[0] = x0
        for t in range(0, T):
            u_traj[t] = controller(x_traj[t])
            x_traj[t + 1] = dyn(x_traj[t], u_traj[t], dt)
    elif isinstance(x0, torch.Tensor):
        x_traj = [x0]
        u_traj = list()
        for t in range(0, T):
            u_traj.append(controller(x_traj[-1], device=device))
            x_traj.append(dyn(x_traj[-1], u_traj[-1], dt))

        x_traj = torch.stack(x_traj, dim=0)
        u_traj = torch.stack(u_traj, dim=0)
    return x_traj, u_traj


def batch_round_pi(x):
    count = 0
    while not (x <= np.pi).all() and count < 10:
        x -= (x > np.pi) * 2 * np.pi
        count += 1
    count = 0
    while not (x >= -np.pi).all() and count < 10:
        x += (x < -np.pi) * 2 * np.pi
        count += 1
    return x


def batch_rotate_2D(xy, theta):
    if isinstance(xy, torch.Tensor):
        x1 = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
        y1 = xy[..., 1] * torch.cos(theta) + xy[..., 0] * torch.sin(theta)
        return torch.stack([x1, y1], dim=-1)
    elif isinstance(xy, np.ndarray):
        x1 = xy[..., 0] * np.cos(theta) - xy[..., 1] * np.sin(theta)
        y1 = xy[..., 1] * np.cos(theta) + xy[..., 0] * np.sin(theta)
        return np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1)), axis=-1)
