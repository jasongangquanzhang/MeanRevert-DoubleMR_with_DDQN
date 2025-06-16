# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:39:56 2022

@author: sebja
"""

from MR_env import MR_env as Environment

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

import copy

import pdb

from datetime import datetime


class DQN(nn.Module):
    def __init__(
        self,
        n_in=2,
        n_out=4,
        nNodes=24,
        nLayers=4,
        activation="relu",
        normalization=False,
    ):
        super(DQN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(n_in, nNodes))
        for _ in range(nLayers - 1):
            self.hidden_layers.append(nn.Linear(nNodes, nNodes))
        self.output_layer = nn.Linear(nNodes, n_out)
        # Add LayerNorm for each hidden layer
        if normalization:
            self.norms = nn.ModuleList([nn.LayerNorm(nNodes) for _ in range(nLayers)])
        else:
            # If normalization is not used, we still create norms for the hidden layers
            # but they will not be applied in the forward pass.
            # This is to maintain the same structure as the original code.
            self.norms = nn.ModuleList([nn.Identity() for _ in range(nLayers)])

        # Activation function
        if activation == "silu":
            self.g = nn.SiLU()
        elif activation == "relu":
            self.g = nn.ReLU()
        elif activation == "gelu":
            self.g = nn.GELU()
        elif activation == "leakyrelu":
            self.g = nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = self.g(self.norms[i](layer(x)))
        return self.output_layer(x)


class DDQN:

    def __init__(
        self,
        env: Environment,
        I_max=10,
        gamma=0.99,
        n_nodes=36,
        n_layers=6,
        lr=1e-3,
        sched_step_size=100,
        name="",
    ):

        self.env = env
        self.gamma = gamma
        self.I_max = I_max
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.name = name
        self.sched_step_size = sched_step_size
        self.lr = lr

        self.__initialize_NNs__()

        self.S = []
        self.I = []
        self.q = []
        self.r = []
        self.epsilon = []

        self.Q_loss = []

        self.tau = 0.001

    def __initialize_NNs__(self):

        # policy approximation
        #
        # features = S, I
        #

        # Q - function approximation
        #
        # features = S, I, I_p
        #
        self.Q_main = {
            "net": DQN(n_in=2, n_out=2, nNodes=self.n_nodes, nLayers=self.n_layers)
        }

        self.Q_main["optimizer"], self.Q_main["scheduler"] = self.__get_optim_sched__(
            self.Q_main
        )

        self.Q_target = copy.deepcopy(self.Q_main)

    def __get_optim_sched__(self, net):

        optimizer = optim.AdamW(net["net"].parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.sched_step_size, gamma=0.99
        )

        return optimizer, scheduler

    def __stack_state__(self, S, I):

        return torch.cat(
            (S.unsqueeze(-1) / self.env.S_0 - 1.0, I.unsqueeze(-1) / self.I_max),
            axis=-1,
        )

    def __grab_mini_batch__(self, mini_batch_size):

        t = torch.rand((mini_batch_size)) * self.env.N
        # t[-int(mini_batch_size*0.05):] = self.env.N

        S, I = self.env.Randomize_Start(mini_batch_size)

        return t, S, I

    def update_Q(self, n_iter=10, mini_batch_size=256, epsilon=0.02):
        for i in range(n_iter):
            _, S, I = self.__grab_mini_batch__(mini_batch_size)
            self.Q_main["optimizer"].zero_grad()

            # concatenate states
            X = self.__stack_state__(S, I)
            Q = self.Q_main["net"](X)  # (batch_size, n_actions)

            # --- Epsilon-greedy action selection ---
            batch_size = Q.shape[0]
            rand_vals = torch.rand(batch_size)
            random_actions = torch.randint(0, Q.shape[1], (batch_size,))
            greedy_actions = Q.argmax(dim=1)
            # With probability epsilon choose random, else greedy
            actions = torch.where(rand_vals < epsilon, random_actions, greedy_actions)
            I_p = (2 * actions - 1) * self.I_max  # Map action index to environment action

            # Gather Q-value for chosen action
            Q_value = Q.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Step in environment to get next state and reward
            S_p, I_p_env, r = self.env.step(0, S, I, I_p)
            # New state
            X_p = self.__stack_state__(S_p, I_p_env)
            Q_p = self.Q_main["net"](X_p)
            # Next greedy action for Double DQN
            next_greedy_actions = Q_p.argmax(dim=1, keepdim=True)
            # Target value using target net
            target_q_values = self.Q_target["net"](X_p).gather(1, next_greedy_actions).squeeze(1)

            # Compute target
            target = r + self.gamma * target_q_values
            target = target.detach()

            # Loss
            loss = torch.mean((Q_value - target) ** 2)
            loss.backward()
            self.Q_main["optimizer"].step()
            self.Q_main["scheduler"].step()
            self.Q_loss.append(loss.item())

            # Target network soft update
            self.soft_update(self.Q_main["net"], self.Q_target["net"])


    def soft_update(self, main, target):

        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train(
        self, n_iter=1_000, n_iter_Q=10, mini_batch_size=256, n_plot=100
    ):

        self.run_strategy(
            nsims=1_000, name=datetime.now().strftime("%H_%M_%S")
        )  # intital evaluation

        C = 50
        D = 100

        if len(self.epsilon) == 0:
            self.count = 0

        # for i in tqdm(range(n_iter)):
        for i in range(n_iter):

            epsilon = np.maximum(C / (D + self.count), 0.02)
            self.epsilon.append(epsilon)
            self.count += 1

            # pdb.set_trace()

            self.update_Q(
                n_iter=n_iter_Q, mini_batch_size=mini_batch_size, epsilon=epsilon
            )


            if np.mod(i + 1, n_plot) == 0:

                self.loss_plots()
                self.run_strategy(
                    1_000, name=datetime.now().strftime("%H_%M_%S"), N=100
                )
                self.plot_policy()
                # self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))

    def moving_average(self, x, n):

        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan

        for i in range(1, len(x)):

            if i < n:
                y[i] = np.mean(x[:i])
                y_err[i] = np.std(x[:i])
            else:
                y[i] = np.mean(x[i - n : i])
                y_err[i] = np.std(x[i - n : i])

        return y, y_err

    def loss_plots(self):

        def plot(x, label, show_band=True):

            mv, mv_err = self.moving_average(x, 100)

            if show_band:
                plt.fill_between(
                    np.arange(len(mv)), mv - mv_err, mv + mv_err, alpha=0.2
                )
            plt.plot(mv, label=label, linewidth=1)
            plt.legend()
            plt.ylabel("loss")
            plt.yscale("symlog")

        fig = plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plot(self.Q_loss, r"$Q$", show_band=False)


        plt.tight_layout()
        plt.show()

    def run_strategy(self, nsims: int = 10_000, name: str = "", N: int = None):
        """Run the trading strategy simulation. Seem to be an evaluation for current policy network.
        The function simulates the evolution of the system over a specified number of time steps and plots the results.

        Args:
            nsims (int, optional): The number of simulations to run. Defaults to 10_000.
            name (str, optional): The name of the simulation. Defaults to "".
            N (int, optional): The number of time steps to simulate. Defaults to None.

        Returns:
            _type_: _description_
        """

        if N is None:
            N = self.env.N  # number of time steps

        S = torch.zeros((nsims, N + 1)).float()
        I = torch.zeros((nsims, N + 1)).float()
        I_p = torch.zeros((nsims, N + 1)).float()
        r = torch.zeros((nsims, N)).float()

        S0 = self.env.S_0
        I0 = 0

        S[:, 0] = S0
        I[:, 0] = 0

        ones = torch.ones(nsims)

        for t in range(N):

            X = self.__stack_state__(S[:, t], I[:, t])
            Q = self.Q_main["net"](X)
            I_p[:, t] = (2*Q.argmax(dim=1, keepdim=False)-1) * self.I_max

            S[:, t + 1], I[:, t + 1], r[:, t] = self.env.step(
                t * ones, S[:, t], I[:, t], I_p[:, t]
            )

        S = S.detach().numpy()
        I = I.detach().numpy()
        I_p = I_p.detach().numpy()
        r = r.detach().numpy()

        t = self.env.dt * np.arange(0, N + 1) / self.env.T

        plt.figure(figsize=(5, 5))
        n_paths = 3

        def plot(t, x, plt_i, title):

            # print(x.shape)
            # pdb.set_trace()

            qtl = np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            # print(qtl.shape)

            plt.subplot(2, 2, plt_i)

            plt.fill_between(t, qtl[0, :], qtl[2, :], alpha=0.5)
            plt.plot(t, qtl[1, :], color="k", linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)

            # plt.xticks([0,0.5,1])
            plt.title(title)
            plt.xlabel(r"$t$")

        plot(t, (S), 1, r"$S_t$")
        plot(t, I, 2, r"$I_t$")
        plot(t[:-1], np.cumsum(r, axis=1), 3, r"$r_t$")

        plt.subplot(2, 2, 4)
        plt.hist(np.sum(r, axis=1), bins=51)

        plt.tight_layout()

        # plt.savefig(
        #     "path_" + self.name + "_" + name + ".pdf", format="pdf", bbox_inches="tight"
        # )
        plt.show()

        # zy0 = self.env.swap_price(zx[0,0], rx[0,0], ry[0,0])
        # plt.hist(zy[:,-1],bins=np.linspace(51780,51810,31), density=True, label='optimal')
        # qtl_levels = [0.05,0.5,0.95]
        # qtl = np.quantile(zy[:,-1],qtl_levels)
        # c=['r','b','g']
        # for i, q in enumerate(qtl):
        #     plt.axvline(qtl[i], linestyle='--',
        #                 linewidth=2,
        #                 color=c[i],
        #                 label=r'${0:0.2f}$'.format(qtl_levels[i]))
        # plt.axvline(zy0,linestyle='--',color='k', label='swap-all')
        # plt.xlabel(r'$z_T^y$')
        # plt.legend()
        # plt.savefig('ddqn_zy_T.pdf', format='pdf',bbox_inches='tight')
        # plt.show()

        # print(zy0, np.mean(zy[:,-1]>zy0))
        # print(qtl)

        return t, S, I, I_p

    def plot_policy(self, name=""):

        NS = 101

        S = torch.linspace(
            self.env.S_0 - 3 * self.env.inv_vol, self.env.S_0 + 3 * self.env.inv_vol, NS
        )
        NI = 51
        I = torch.linspace(-self.I_max, self.I_max, NI)

        Sm, Im = torch.meshgrid(S, I, indexing="ij")

        def plot(a, title):

            fig, ax = plt.subplots()
            plt.title("Inventory vs Price Heatmap for Time T")

            cs = plt.contourf(
                Sm.numpy(),
                Im.numpy(),
                a,
                levels=np.linspace(-self.I_max, self.I_max, 21),
                cmap="RdBu",
            )
            plt.axvline(self.env.S_0, linestyle="--", color="g")
            plt.axvline(self.env.S_0 - 2 * self.env.inv_vol, linestyle="--", color="k")
            plt.axvline(self.env.S_0 + 2 * self.env.inv_vol, linestyle="--", color="k")
            plt.axhline(0, linestyle="--", color="k")
            plt.axhline(self.I_max / 2, linestyle="--", color="k")
            plt.axhline(-self.I_max / 2, linestyle="--", color="k")
            ax.set_xlabel("Price")
            ax.set_ylabel("Inventory")
            ax.set_title(title)

            cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
            cbar.set_ticks(np.linspace(-self.I_max, self.I_max, 11))
            cbar.ax.set_ylabel("Action")

            plt.tight_layout()
            plt.show()

        # X = torch.cat( ((Sm.unsqueeze(-1)/self.env.S_0-1.0),
        #                 Im.unsqueeze(-1)/self.I_max), axis=-1)

        X = self.__stack_state__(Sm, Im)

        a = ((2*self.Q_main["net"](X).argmax(dim=2, keepdim=False)-1)*self.I_max).detach().squeeze()

        plot(a, r"")
