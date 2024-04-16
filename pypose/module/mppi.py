import torch
import functools
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal



class MPPI():


    def __init__(self,
                 dynamics,
                 running_cost,
                 nx,
                 noise_sigma,
                 num_samples=100,
                 horizon=15.0,
                 device="cpu",
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 noise_abs_cost=False,
                 batch_size=1
                 ):
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_
        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)
        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        self.u_max = u_max
        self.u_min = u_min
        if self.u_max is not None and self.u_min is None:
            self.u_max = torch.tensor(self.u_max) if not torch.is_tensor(self.u_max) else self.u_max
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            self.u_min = torch.tensor(self.u_min) if not torch.is_tensor(self.u_min) else self.u_min
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)
        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, \
            covariance_matrix=self.noise_sigma)
        self.system = dynamics.state_transition
        self.running_cost = running_cost
        self.noise_abs_cost = noise_abs_cost
        self.state = None
        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None
        self.batch_size = batch_size



    def forward(self, state, u_init=None):
        state = torch.tensor(state, dtype=self.dtype, device=self.d) if not torch.is_tensor(state) else state.to(dtype=self.dtype, device=self.d)
        u_init = torch.zeros_like(self.noise_mu, device=self.d) if u_init is None else u_init.to(self.d)

        batch_size = state.shape[0]  # Get the batch size from the input state

        U_base = self.noise_dist.sample((self.T,))
        U_base = torch.roll(U_base, -1, dims=0)
        U_base[-1] = u_init

        noise = self.noise_dist.sample((self.K, self.T))
        perturbed_action = U_base + noise

        if self.u_max is not None:
            for t in range(self.T):
                u = perturbed_action[:, slice(t * self.nu, (t + 1) * self.nu)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                perturbed_action[:, slice(t * self.nu, (t + 1) * self.nu)] = cu

        noise = perturbed_action - U_base
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(noise) @ self.noise_sigma_inv
        else:
            action_cost = self.lambda_ * noise @ self.noise_sigma_inv

        K, T, nu = perturbed_action.shape
        assert nu == self.nu
        cost_total_temp = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total_temp
        states_temp = []
        actions_temp = []

        # Repeat the initial state to match the batch size
        state_temp = state.repeat(K, 1)

        for t in range(T):
            u = perturbed_action[:, t]
            state_temp = self.system(state_temp, u)
            #import pdb; pdb.set_trace()
            c = self.running_cost(state_temp, u, t)
            cost_samples = cost_samples + c
            states_temp.append(state_temp)
            actions_temp.append(u)

        cost_total_temp = cost_total_temp + cost_samples.mean(dim=0)
        cost_total = cost_total_temp

        # action perturbation cost
        perturbation_cost = torch.sum(U_base * action_cost, dim=(1, 2))
        cost_total = cost_total + perturbation_cost

        beta = torch.min(cost_total)
        cost_total_non_zero = torch.exp(-(1 / self.lambda_) * (cost_total - beta))
        eta = torch.sum(cost_total_non_zero)
        self.omega = (1. / eta) * cost_total_non_zero
        for t in range(self.T):
            U_base[t] = U_base[t] + \
                torch.sum(self.omega.view(-1, 1) * noise[:, t], dim=0)

        action = U_base[:self.T]

        #import pdb; pdb.set_trace()
        action2states = [state]
        # Start with initial state in a list

        for i in range(T):
            cur_state = self.system(action2states[i], U_base[i])
            action2states.append(cur_state)

        # Convert list of tensors to a single tensor
        action2states = torch.stack(action2states, dim=0)

        return U_base, action2states
