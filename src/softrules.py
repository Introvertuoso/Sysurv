import numpy as np
import torch
from torch import nn


# Exactly as provided by the authors of Xu et al. (2024)
class AndFinder(nn.Module):
    def __init__(self, cut_points, temperature=0.2, epsilon=1e-5, bin_deviation=0.20, use_weights=True):
        super().__init__()
        n_variables = cut_points.shape[0]
        self.cut_points = nn.Parameter(cut_points.clone().detach(), requires_grad=True)

        self.zero = nn.Parameter(torch.zeros([n_variables, 1], dtype=torch.float64), requires_grad=False)
        self.epsilon = epsilon
        self.temperature = temperature
        D = cut_points.shape[1]
        if D != 2:
            raise ValueError("And finder only works for two given cutpoints per feature")
        self.fixed_weights = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1, dtype=torch.float64), [D + 1])
        # repeat fixed weights for each variable
        self.fixed_weights = nn.Parameter(self.fixed_weights.clone().detach(), requires_grad=False)

        initial_weights = torch.rand([n_variables, ], dtype=torch.float64)
        initial_weights[:] = 1

        self.and_weights = nn.Parameter(initial_weights, requires_grad=use_weights)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

        limits = cut_points.clone().detach()
        # scale by 10% of the range
        limits[:, 0] = limits[:, 0] - bin_deviation * (limits[:, 1] - limits[:, 0])
        limits[:, 1] = limits[:, 1] + bin_deviation * (limits[:, 1] - limits[:, 0])
        self.limits = nn.Parameter(limits, requires_grad=False)

    def forward(self, x):
        cut_points = self.cut_points
        b = torch.cumsum(torch.cat([self.zero, -cut_points], 1), 1)
        # repeat x along new dimension for each fixed weight
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, self.fixed_weights.shape[0])
        weights = self.fixed_weights.repeat(x.shape[0], x.shape[1], 1)
        h = x * weights
        # add b to the batch
        b = b.repeat(x.shape[0], 1, 1)
        h = h + b
        h = h / self.temperature
        bins = self.softmax(h)

        importance = self.relu(self.and_weights)
        c = ((1 + self.epsilon) / (bins[:, :, 1] + self.epsilon)) @ importance
        res = torch.sum(importance) / c
        self.c = torch.sum(c).item() / c.shape[0]

        res = torch.stack([1 - res, res], dim=1)
        return res

    def get_rules(self, data_limits, feature_names=None, scaler=None, X=None):
        cut_points = self.cut_points.data
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(cut_points.shape[0])]
        if scaler is not None:
            cut_points = scaler.inverse_transform(cut_points.detach().cpu().numpy().T).T
            data_limits = scaler.inverse_transform(data_limits.detach().cpu().numpy().T).T
        else:
            cut_points = cut_points.detach().cpu().numpy()
            data_limits = data_limits.detach().cpu().numpy()

        rule = []
        for i in range(cut_points.shape[0]):
            lower_bound, upper_bound = cut_points[i, :]
            if lower_bound < data_limits[i, 0] and upper_bound > data_limits[i, 1]:
                continue
            and_weight = self.and_weights[i]
            lower_bound = np.max([data_limits[i, 0], lower_bound])
            upper_bound = np.min([data_limits[i, 1], upper_bound])
            if and_weight < 0.1:
                continue

            nel = np.unique(X[:, i])
            if len(nel) == 2:
                if upper_bound < 1:
                    rule.append("¬" + feature_names[i])
                else:
                    rule.append(feature_names[i])
            else:
                rule.append(f"{lower_bound:.2f} < {feature_names[i]} < {upper_bound:.2f}")

        return " ∧ ".join(rule)

    def get_utilized_features(self, data_limits, feature_names=None, scaler=None):
        cut_points = self.cut_points.data
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(cut_points.shape[0])]
        if scaler is not None:
            cut_points = scaler.inverse_transform(cut_points.detach().cpu().numpy().T).T
            data_limits = scaler.inverse_transform(data_limits.detach().cpu().numpy().T).T
        else:
            cut_points = cut_points.detach().cpu().numpy()
            data_limits = data_limits.detach().cpu().numpy()

        used_features = []
        for i in range(cut_points.shape[0]):
            lower_bound, upper_bound = cut_points[i, :]
            if lower_bound < data_limits[i, 0] and upper_bound > data_limits[i, 1]:
                continue
            and_weight = self.and_weights[i]
            if and_weight < 0.1:
                continue
            used_features.append(i)
        return used_features

    def get_and_weights(self):
        return self.and_weights

    def fix_parameters(self):
        # sort cut points
        self.cut_points.data, _ = torch.sort(self.cut_points.data)
        for i in range(self.cut_points.shape[0]):
            limits = self.limits[i, :]
            self.cut_points.data[i, :] = torch.maximum(self.cut_points.data[i, :], limits[0])
            self.cut_points.data[i, :] = torch.minimum(self.cut_points.data[i, :], limits[1])