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
import torch.nn.functional as F

import random

from tqdm import tqdm

import copy

import pdb

from datetime import datetime

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, n_nodes=16, n_hidden_layers=6):
        assert n_hidden_layers >= 1, "You must have at least one hidden layer"
        super(QNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_size, n_nodes))
        for _ in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(n_nodes, n_nodes))
        self.output_layer = nn.Linear(n_nodes, action_size)
        # self.fc1 = nn.Linear(state_size, 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0
    
    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state)
        self.index = (self.index + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = [], [], [], []
        for i in batch:
            state, action, reward, next_state = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
        return (
            torch.stack(states).float(),
            torch.tensor(actions).long(),
            torch.stack(rewards).float(),
            torch.stack(next_states).float(),
        )
    
    def __len__(self):
        return len(self.buffer)

class DDQNAgent:
    def __init__(self, env: Environment, state_size, action_size, seed, lr=1e-3, capacity=1000000,
                 gamma=0.99, tau=1e-3, update_every=4, batch_size=64, I_max=10, name=''):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.learning_rate = lr
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.I_max = I_max
        self.steps = 0

        self.action_spaces = torch.linspace(-self.I_max, self.I_max, action_size)

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity)
        self.update_target_network()

        self.epsilon = []
        self.losses = []

    def step(self, state, action, reward, next_state):
        # Save experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state)

        # Learn every update_every steps
        self.steps += 1
        if self.steps % self.update_every == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.learn(experiences)

    def act(self, state, eps=0.0):
        state = state.float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + self.gamma * Q_targets_next

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.view(-1, 1))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss for plotting
        self.losses.append(loss.item())

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def update_target_network(self):
        # Update target network parameters with polyak averaging
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def train(self, n_iter=10_000, n_plot=200, eps_start=1.0, eps_end=0.01, eps_decay=0.999):
        eps = eps_start

        rewards = []
        scores = []

        if len(self.epsilon) == 0:
            self.count = 0

        for i in tqdm(range(n_iter)):
            eps = max(eps_end, eps_decay * eps)
            self.epsilon.append(eps)
            self.count += 1

            # Run the episode
            S, I = self.env.Randomize_Start(1)
            state = self.__stack_state__(S, I)
            score = 0

            for t in range(self.env.N):
                # Select an action and take a step in the environment
                action = self.act(state, eps)
                action_value = self.action_spaces[action].unsqueeze(0)
                S_p, I_p, reward = self.env.step(t, S, I, action_value)
                # Store the experience in the replay buffer and learn from it
                next_state = self.__stack_state__(S_p, I_p)
                self.step(state, action, reward, next_state)
                # Update the state and the score
                state = next_state
                S, I = S_p, I_p
                score += reward
            
            if i != 0 and i % n_plot == 0:
                print(f"\tScore: {score}, Epsilon: {eps}")
                self.loss_plots()
                self.run_strategy(10_000, name=f"episode_{i:04d}")
                self.plot_policy(name=f"episode_{i:04d}")
            # Save the rewards and scores
            rewards.append(score.item() if torch.is_tensor(score) else score)
            scores.append(np.mean(rewards[-100:]))

        # torch.save(self.qnetwork_local.state_dict(), "mr_ddqn_model.pth")

        # plt.ylabel("Score")
        # plt.xlabel("Episode")
        # plt.plot(range(len(rewards)), rewards)
        # plt.plot(range(len(scores)), scores)
        # plt.legend(['Reward', "Score"])
        # plt.savefig("mr_ddqn_scores.png")
        # plt.show()

        # Plot the loss
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss over Time")
        plt.grid(True)
        plt.savefig("loss_curve.png")
        plt.show()

        # Plot the policy heatmap and simulation results
        print(f'Epsilon: {eps}')
        self.loss_plots()
        self.run_strategy(10_000, name=datetime.now().strftime("%H_%M_%S"))
        self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
        

    
    def __stack_state__(self, S, I, dim=0):
        return torch.stack([
            S.squeeze() / self.env.S_0 - 1.0,
            I.squeeze() / self.I_max
        ], dim=dim)

    def moving_average(self, x, n):
        
        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan
        
        for i in range(1,len(x)):
            
            if i < n:
                y[i] = np.mean(x[:i])
                y_err[i] = np.std(x[:i])
            else:
                y[i] = np.mean(x[i-n:i])
                y_err[i] = np.std(x[i-n:i])
                
        return y, y_err                
            
    def loss_plots(self):
        
        def plot(x, label, show_band=True):

            mv, mv_err = self.moving_average(x, 100)
        
            if show_band:
                plt.fill_between(np.arange(len(mv)), mv-mv_err, mv+mv_err, alpha=0.2)
            plt.plot(mv, label=label, linewidth=1) 
            plt.legend()
            plt.ylabel('loss')
            plt.yscale('symlog')
        
        fig = plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plot(self.losses, r'$Q$', show_band=False)
        
        plt.tight_layout()
        plt.show()
        
    def run_strategy(self, nsims=10_000, name="", N = None):
        
        if N is None:
            N = self.env.N
        
        S = torch.zeros((nsims, N+1)).float()
        I  = torch.zeros((nsims, N+1)).float()
        I_p = torch.zeros((nsims, N+1)).float()
        r = torch.zeros((nsims, N)).float()

        S0 = self.env.S_0
        I0 = 0

        S[:,0] = S0
        I[:,0] = 0
        
        ones = torch.ones(nsims)
        
        for t in range(N):

            X = self.__stack_state__(S[:,t], I[:,t], dim=1)
            
            with torch.no_grad():
                actions = torch.argmax(self.qnetwork_local(X), dim=1)
                I_p[:, t] = self.action_spaces[actions]

            # I_p[:,t] = self.pi_main['net'](X).reshape(-1)

            S[:,t+1], I[:,t+1], r[:,t] = \
                self.env.step(t*ones, S[:,t], I[:,t], I_p[:,t])
                
        S = S.detach().numpy()
        I  = I.detach().numpy()
        I_p = I_p.detach().numpy()
        r = r.detach().numpy()

        t = self.env.dt*np.arange(0, N+1)/self.env.T
        
        plt.figure(figsize=(5,5))
        n_paths = 3
        
        def plot(t, x, plt_i, title ):
            
            # print(x.shape)
            # pdb.set_trace()
            
            qtl= np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            # print(qtl.shape)
            
            plt.subplot(2, 2, plt_i)
            
            plt.fill_between(t, qtl[0,:], qtl[2,:], alpha=0.5)
            plt.plot(t, qtl[1,:], color='k', linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)
            
            # plt.xticks([0,0.5,1])
            plt.title(title)
            plt.xlabel(r"$t$")
            
        plot(t, (S-S[:,0].reshape(S.shape[0],-1)), 1, r"$S_t-S_0$" )
        plot(t, I, 2, r"$I_t$")
        plot(t[:-1], np.cumsum(r, axis=1), 3, r"$r_t$")

        plt.subplot(2,2, 4)
        plt.hist(np.sum(r,axis=1), bins=51)


        plt.tight_layout()
        
        # plt.savefig("path_" + name + ".pdf", format='pdf', bbox_inches='tight')
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
        S = torch.linspace(self.env.S_0 - 3*self.env.inv_vol, 
                           self.env.S_0 + 3*self.env.inv_vol,
                           NS)
        NI = 51
        I = torch.linspace(-self.I_max, self.I_max, NI)
        
        Sm, Im = torch.meshgrid(S, I,indexing='ij')
        X = self.__stack_state__(Sm.reshape(-1), Im.reshape(-1), dim=1)  # shape [NS * NI, 2]
        with torch.no_grad():
            Q_values = self.qnetwork_local(X)
            greedy_actions = torch.argmax(Q_values, dim=1).cpu()
        
        # Convert discrete actions back to real values
        A = self.action_spaces[greedy_actions.numpy()].reshape(Sm.shape)

        fig, ax = plt.subplots()
        plt.title("Optimal Policy Heatmap for Time T")
        cs = plt.contourf(Sm.numpy(), Im.numpy(), A,
                          levels=np.linspace(-self.I_max, self.I_max, 21),
                          cmap='RdBu')
        ax.axvline(self.env.S_0, linestyle='--', color='g')
        ax.axvline(self.env.S_0 - 2 * self.env.inv_vol, linestyle='--', color='k')
        ax.axvline(self.env.S_0 + 2 * self.env.inv_vol, linestyle='--', color='k')
        ax.axhline(0, linestyle='--', color='k')
        ax.axhline(self.I_max / 2, linestyle='--', color='k')
        ax.axhline(-self.I_max / 2, linestyle='--', color='k')
        ax.set_xlabel("Price")
        ax.set_ylabel("Inventory")

        cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
        cbar.set_ticks(np.linspace(-self.I_max, self.I_max, 11))
        cbar.ax.set_ylabel('Action')

        plt.tight_layout()
        # plt.savefig("policy_" + name + ".pdf", format='pdf', bbox_inches='tight')
        plt.show()