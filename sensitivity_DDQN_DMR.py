# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

from MR_env import MR_env
from DMR_env import DMR_env
from DDQN import DDQNAgent

def run_once_with_parameter(lambd, sigma, kappa):
    env = DMR_env(S_0=1, kappa=kappa, sigma=sigma, theta_a=0.7, theta_b=1.3,
                dt=0.25, T=int(20),
                I_max=10, lambd=lambd)
    ddqn = DDQNAgent(env, state_size=2, action_size=15, seed=0,
                     I_max=10,
                     gamma=0.999,
                     lr=1e-3,
                     name=f"test_lambd_{lambd}_sigma_{sigma}_kappa_{kappa}")
    ddqn.train(n_iter=1_000, n_plot=1_000,
               eps_start=1.0, eps_end=0.01, eps_decay=0.993)
    return ddqn

lambd0 = 0.05
sigma0 = 0.2
kappa0 = 1

# lambd_values = [0.01, 0.05, 0.1]
# sigma_values = [0.1, 0.2, 0.3]
# kappa_values = [0.5, 1, 2]
lambd_values = [0.001, 0.05, 0.1]
sigma_values = [0.01, 0.2, 0.4]
kappa_values = [0.1, 1, 2]

# Run experiments for different parameter values
for lambd in lambd_values:
    print(f"Running with lambd={lambd}, sigma={sigma0} (default), kappa={kappa0} (default)")
    run_once_with_parameter(lambd, sigma0, kappa0)
for sigma in sigma_values:
    print(f"Running with lambd={lambd0} (default), sigma={sigma}, kappa={kappa0} (default)")
    run_once_with_parameter(lambd0, sigma, kappa0)
for kappa in kappa_values:
    print(f"Running with lambd={lambd0} (default), sigma={sigma0} (default), kappa={kappa}")
    run_once_with_parameter(lambd0, sigma0, kappa)