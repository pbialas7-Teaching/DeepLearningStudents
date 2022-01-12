import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(ax, trajectory, n_batches=1, **kwargs):
    epochs = np.linspace(0, len(trajectory) / n_batches, len(trajectory))
    ax.plot(epochs, trajectory[:, -1], '.', **kwargs)


def plot_grad_and_trajectory(ax, trajectory, n_batches=1):
    epochs = np.linspace(0, len(trajectory) / n_batches, len(trajectory))
    ax.plot(epochs, trajectory[:, -1], '.')
    tw = ax.twinx()
    tw.plot(epochs, np.linalg.norm(trajectory[:, :2], axis=1), '.r')

    l1, u1 = ax.get_ylim()
    l2, u2 = tw.get_ylim();
    nl2 = (u2 - l2) / (u1 - l1) * l1
    dl2 = nl2 - l2
    ax.set_ylabel("MSE", color="blue")
    tw.set_ylim(l2 + dl2, u2 + dl2);
    tw.spines['right'].set_color('red')
    tw.tick_params(axis='y', colors='red')
    tw.set_ylabel("||grad||", color="red");
    ax.set_xlabel("epoch")


def fitf(x, o, t):
    return np.sin(x * o + t)


def fitf_tensor(x, o, t):
    return np.moveaxis(np.sin(np.tensordot(np.atleast_1d(x), o, 0) + t), 0, -1)


def mse(f, x, y, o, t):
    err = f(x, o, t) - y
    return 0.5 * np.sum(err * err, axis=-1) / len(x)


def grad(x, y, o, t):
    return np.array((
        -2 * np.sum((y - np.sin(o * x + t)) * np.cos(o * x + t) * x),
        -2 * np.sum((y - np.sin(o * x + t)) * np.cos(o * x + t))
    )) / len(x)


def run_example(pstart, optim, loader, otg, vg):
    loss_f = torch.nn.MSELoss()
    p = torch.FloatTensor(pstart)
    p.requires_grad_(True);
    fig_gd_mom, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].contourf(otg[0], otg[1], vg, levels=20)
    ax[0].scatter([p[0].item()], [p[1].item()], c='none', s=20, edgecolor='red')
    trajectory_list = []
    n_iter = 50
    for i in range(n_iter):
        n_batches = 0
        for x, y in loader:
            n_batches += 1
            optim.zero_grad()

            prediction = torch.sin(x * p[0] + p[1])
            loss = loss_f(prediction, y)
            loss.backward()
            optim.step()
            np_p = p.detach().numpy()
            trajectory_list.append(np.concatenate((p.grad.numpy(), np_p, [mse(fitf, rxs, rys, *np_p)])))
            ax[0].scatter([np_p[0]], [np_p[1]], c='red', s=20, edgecolor='red')

    trajectory = np.stack(trajectory_list)
    utils.plot_grad_and_trajectory(ax[1], trajectory, n_batches)
    ax[1].set_xlabel("epoch")

    return fig_gd, ax, trajectory


class SinFitExample:
    def __init__(self, data, n_samples=400):
        self.data = data.copy()
        self.n_samples = n_samples
        self.rxs = self.data[:n_samples, 0]
        self.rys = self.data[:n_samples, 1]

        self.grid_size = 400
        self.os = np.linspace(0, 7, self.grid_size)
        self.ts = np.linspace(-np.pi, np.pi, self.grid_size)
        self.otg = np.meshgrid(self.os, self.ts)

        self.vg = mse(fitf_tensor, self.rxs, self.rys, self.otg[0], self.otg[1])
        self.loss_f = torch.nn.MSELoss()

    def display_data(self):
        fig, ax = plt.subplots()
        ax.scatter(self.rxs, self.rys, alpha=0.7, color='none', edgecolor="black");
        return fig, ax

    def run_example(self, p, optim, loader, n_iter=50):

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].contourf(self.otg[0], self.otg[1], self.vg, levels=20)
        ax[0].scatter([p[0].item()], [p[1].item()], c='none', s=20, edgecolor='red')
        trajectory_list = []
        for i in range(n_iter):
            n_batches = 0
            for x, y in loader:
                n_batches += 1
                optim.zero_grad()
                prediction = torch.sin(x * p[0] + p[1])
                loss = self.loss_f(prediction, y)
                loss.backward()
                optim.step()
                np_p = p.detach().numpy()
                trajectory_list.append(np.concatenate((p.grad.numpy(), np_p, [mse(fitf, self.rxs, self.rys, *np_p)])))
                ax[0].scatter([np_p[0]], [np_p[1]], c='red', s=20, edgecolor='red')

        trajectory = np.stack(trajectory_list)
        plot_grad_and_trajectory(ax[1], trajectory, n_batches)
        ax[1].set_xlabel("epoch")

        return fig, ax, trajectory


class RavineExample:
    def __init__(self, par):
        self.par = par
        grid_size = 400
        self.os = np.linspace(-10, 10, grid_size)
        self.ts = np.linspace(-10, 10, grid_size)
        self.otg = np.meshgrid(self.os, self.ts)
        self.vg = self.ravine(self.otg[0], self.otg[1])

    def ravine(self, p0, p1):
        return 0.5 * ((self.par[0] * p0) ** 2 + (self.par[1] * p1) ** 2)

    def descend(self, p, optim, n_iter, tol=0.0):
        trajectory_list = []

        for i in range(n_iter):
            optim.zero_grad()
            loss = 0.5 * torch.sum((torch.from_numpy(self.par) * p) ** 2)
            loss.backward()
            np_p = p.detach().numpy()
            trajectory_list.append(np.concatenate((p.grad.numpy(), np_p, [self.ravine(*np_p)])))
            if (loss.item() < tol):
                break
            optim.step()

        trajectory = np.stack(trajectory_list)

        return trajectory

    def plot(self, ax, trajectory=None):
        ax.contourf(self.otg[0], self.otg[1], self.vg, levels=20)
        if trajectory is not None:
            ax.scatter(trajectory[:, 2], trajectory[:, 3], c='none', s=20, edgecolor='red')
            ax.plot(trajectory[:, 2], trajectory[:, 3], 'r-')
        return ax

    def run_example(self, p, optim, n_iter, tol=0.0):
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].set_xlim(-10, 10)
        ax[0].set_ylim(-10, 10)
        ax[1].set_xlim(0, 100)
        trajectory = self.descend(p, optim, n_iter, tol=tol)
        self.plot(ax[0], trajectory)
        plot_grad_and_trajectory(ax[1], trajectory)
        return fig, ax
