import torch
from torch import nn
import qpth
from qpth.qp import QPFunction
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(precision=10)
total_cost_all=[]
class UAVTrajectoryPlanner(nn.Module):
    def __init__(self,waypoints, total_time, poly_order, start_vel, start_acc, end_vel, end_acc, dtype=torch.float32, device='cpu'):
        super(UAVTrajectoryPlanner, self).__init__()
        self.waypoints=waypoints
        self.total_time = total_time
        self.poly_order = poly_order
        self.start_vel = start_vel
        self.start_acc = start_acc
        self.end_vel = end_vel
        self.end_acc = end_acc
        self.dtype = dtype
        self.device = device


    def init_time_segments1(self, num_segments):
        segment_time = self.total_time / num_segments
        time_stamps = torch.arange(0, self.total_time + segment_time, segment_time, dtype=self.dtype, device=self.device)
        # print(time_stamps)
        return time_stamps  # Adjusted to exclude the last redundant timestamp
    def init_time_segments(self, waypoints, total_time):
        """
        Arrange waypoints in time.

        Args:
            waypoints (Tensor): waypoints tensor
            total_time (float): total time for the trajectory

        Returns:
            Tensor: arranged time tensor
        """
        differences = waypoints[:, 1:] - waypoints[:, :-1]
        distances = torch.sqrt(torch.sum(differences ** 2, dim=0))
        time_fraction = total_time / torch.sum(distances)
        arranged_time = torch.cat([torch.tensor([0]), torch.cumsum(distances * time_fraction, dim=0)])
        return arranged_time


    def evaluate_trajectory(self, T, plot_trajectory=False):
        # Direct use of T_current to compute polynomial coefficients
        polys_x = self.solve_minimum_snap1(self.waypoints[0], T, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
        polys_y = self.solve_minimum_snap1(self.waypoints[1], T, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
        # Calculate cost function based on the computed polynomials
        if plot_trajectory:
            self.plot_trajectory(self.waypoints, polys_x, polys_y, T, clear_plot=False,flag=False)
        cost = self.evaluate_cost_function(polys_x, polys_y, T)
        return cost
    def evaluate_cost_function(self, polys_x, polys_y, T):
        # Assuming polys_x and polys_y are tensors with shape [6, num_segments]
        # T has shape [num_segments + 1]
        total_cost = 0
        num_segments = polys_x.shape[1]

        for segment in range(num_segments):
            segment_duration = T[segment + 1] - T[segment]
            snap_x = polys_x[5, segment]
            snap_y = polys_y[5, segment]
            segment_cost = (snap_x**2 + snap_y**2) * segment_duration
            total_cost += segment_cost

        return total_cost
    def optimize_time_segments(self, waypoints, total_time):
        num_segments = waypoints.size(1) - 1
        T = self.init_time_segments(waypoints, total_time)  # Initial time segments
        T = torch.nn.Parameter(T)  # Make T a parameter that requires gradient

        optimizer = torch.optim.Adam([T], lr=2.0)  # Using Adam optimizer with a learning rate of 0.1
        loss_all=[]
        for iteration in range(10):  # Adjust the number of iterations as needed
            optimizer.zero_grad()  # Clear gradients

            loss = self.evaluate_trajectory(T)  # Assuming this function computes the total cost based on T
            loss.backward()  # Compute gradients automatically
            # loss_all.append(loss)
            # print(loss_all)

            optimizer.step()  # Update T based on computed gradients

            # Normalize T to maintain total_time after adjustment
            with torch.no_grad():  # Ensure no gradient computation is done here
                T.clamp_(min=0)  # Ensure all time segments are non-negative
                T /= T.sum()
                T *= total_time

            print(f'Iteration {iteration+1}, Loss: {loss.item()}, Updated Times: {T.data}')
        plt.show()
        return T

    def calculate_time_vector(self,time, order, derivative_order):
        """
        Calculate time vector for a given time, polynomial order, and derivative order.

        Args:
            time (float): the time at which the vector is calculated
            order (int): order of the polynomial
            derivative_order (int): derivative order

        Returns:
            Tensor: calculated time vector
        """
        time_vector = torch.zeros(order + 1)
        for i in range(derivative_order + 1, order + 2):
            if i - derivative_order - 1 > 0:
                product = torch.prod(torch.arange(i - derivative_order, i, dtype=torch.float32))
            else:
                product = torch.prod(torch.arange(i - derivative_order, i, dtype=torch.float32))
            time_vector[i - 1] = product * (time ** (i - derivative_order - 1))
        return time_vector
    def compute_Q_matrix(self,poly_order, derivative_order, start_time, end_time):
        """
        Compute the Q matrix for minimum snap problem.

        Args:
            poly_order (int): order of the polynomial
            derivative_order (int): derivative order
            start_time (float): start time
            end_time (float): end time

        Returns:
            Tensor: Q matrix
        """
        time_diff_powers = torch.zeros((poly_order - derivative_order) * 2 + 1, dtype=torch.float64)
        for i in range((poly_order - derivative_order) * 2 + 1):
            time_diff_powers[i] = end_time ** (i + 1) - start_time ** (i + 1)

        Q_matrix = torch.zeros(poly_order + 1, poly_order + 1, dtype=torch.float64)
        for i in range(derivative_order + 1, poly_order + 2):
            for j in range(i, poly_order + 2):
                k1 = i - derivative_order - 1
                k2 = j - derivative_order - 1
                k = k1 + k2 + 1
                prod_k1 = torch.prod(torch.tensor(range(k1 + 1, k1 + derivative_order + 1), dtype=torch.float64))
                prod_k2 = torch.prod(torch.tensor(range(k2 + 1, k2 + derivative_order + 1), dtype=torch.float64))
                Q_matrix[i - 1, j - 1] = prod_k1 * prod_k2 / k * time_diff_powers[k - 1]
                Q_matrix[j - 1, i - 1] = Q_matrix[i - 1, j - 1]

        return Q_matrix

    def solve_minimum_snap(self, waypoints, time_stamps, poly_order, start_vel, start_acc, end_vel, end_acc):
        """
        Solve the minimum snap problem for a single axis.

        Args:
            waypoints (Tensor): waypoints
            time_stamps (Tensor): time stamps for each segment
            poly_order (int): polynomial order
            start_vel (float): start velocity
            start_acc (float): start acceleration
            end_vel (float): end velocity
            end_acc (float): end acceleration
            dtype (data type): data type for tensors
            device (str): computation device ('cpu' or 'cuda')

        Returns:
            Tensor: coefficients of the polynomial
        """
        # print(waypoints,time_stamps)
        start_pos = waypoints[0]
        end_pos = waypoints[-1]
        num_segments = len(waypoints) - 1
        num_coefficients = poly_order + 1

        # Compute Q matrix for all segments
        Q_all = torch.block_diag(*[self.compute_Q_matrix(poly_order, 3, time_stamps[i], time_stamps[i + 1]) for i in range(num_segments)])
        b_all = torch.zeros(Q_all.shape[0])

        # Setup equality constraints
        Aeq = torch.zeros(4 * num_segments + 2, num_coefficients * num_segments)
        beq = torch.zeros(4 * num_segments + 2)
        Aeq[0:3, :num_coefficients] = torch.stack([
            self.calculate_time_vector(time_stamps[0], poly_order, 0),
            self.calculate_time_vector(time_stamps[0], poly_order, 1),
            self.calculate_time_vector(time_stamps[0], poly_order, 2)])
        Aeq[3:6, -num_coefficients:] = torch.stack([
            self.calculate_time_vector(time_stamps[-1], poly_order, 0),
            self.calculate_time_vector(time_stamps[-1], poly_order, 1),
            self.calculate_time_vector(time_stamps[-1], poly_order, 2)])
        beq[0:6] = torch.tensor([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc])
        # beq[:6] = torch.cat((start_pos, start_vel, start_acc, end_pos, end_vel, end_acc))


        # Middle waypoints constraints
        num_eq_constraints = 6
        for i in range(1, num_segments):
            Aeq[num_eq_constraints, i * num_coefficients:(i + 1) * num_coefficients] = self.calculate_time_vector(time_stamps[i], poly_order, 0)
            beq[num_eq_constraints] = waypoints[i]
            num_eq_constraints += 1

        # Continuity constraints
        for i in range(1, num_segments):
            time_vector_p = self.calculate_time_vector(time_stamps[i], poly_order, 0)
            time_vector_v = self.calculate_time_vector(time_stamps[i], poly_order, 1)
            time_vector_a = self.calculate_time_vector(time_stamps[i], poly_order, 2)
            Aeq[num_eq_constraints:num_eq_constraints + 3, (i - 1) * num_coefficients:(i + 1) * num_coefficients] = torch.stack([
                torch.cat([time_vector_p, -time_vector_p]),
                torch.cat([time_vector_v, -time_vector_v]),
                torch.cat([time_vector_a, -time_vector_a])])
            num_eq_constraints += 3

        # Convert to the specified data type and device
        G_dummy = torch.zeros(1, Q_all.size(0), Q_all.size(0), dtype=torch.float64)
        h_dummy = torch.zeros(1, Q_all.size(0), dtype=torch.float64)
        Q_all += torch.eye(Q_all.size(0), dtype=torch.float64) * 1e-6
        Q_all = Q_all.to(dtype=self.dtype, device=self.device)
        b_all = b_all.to(dtype=self.dtype, device=self.device)
        Aeq = Aeq.to(dtype=self.dtype, device=self.device)
        beq = beq.to(dtype=self.dtype, device=self.device)
        G_dummy = G_dummy.to(dtype=self.dtype, device=self.device)
        h_dummy = h_dummy.to(dtype=self.dtype, device=self.device)

        # Solve the QP problem
        solver_options = {'eps':1e-24, 'maxIter': 100, 'solver': qpth.qp.QPSolvers.PDIPM_BATCHED}
        solution = QPFunction(verbose=-1, **solver_options)(Q_all, b_all, G_dummy, h_dummy, Aeq, beq)
        polynomial_coefficients = solution.view(num_segments, num_coefficients).transpose(0, 1)
        return polynomial_coefficients
    def solve_minimum_snap1(self, waypoints, time_stamps, poly_order, start_vel, start_acc, end_vel, end_acc):
        """
        Solve the minimum snap problem for a single axis.

        Args:
            waypoints (Tensor): waypoints
            time_stamps (Tensor): time stamps for each segment
            poly_order (int): polynomial order
            start_vel (float): start velocity
            start_acc (float): start acceleration
            end_vel (float): end velocity
            end_acc (float): end acceleration
            dtype (data type): data type for tensors
            device (str): computation device ('cpu' or 'cuda')

        Returns:
            Tensor: coefficients of the polynomial
        """
        # print(waypoints,time_stamps)
        start_pos = waypoints[0]
        end_pos = waypoints[-1]
        num_segments = len(waypoints) - 1
        num_coefficients = poly_order + 1

        # Compute Q matrix for all segments
        Q_all = torch.block_diag(*[self.compute_Q_matrix(poly_order, 3, time_stamps[i], time_stamps[i + 1]) for i in range(num_segments)])
        b_all = torch.zeros(Q_all.shape[0])
        # print(bool((Q_all == Q_all.T).all() and (torch.linalg.eigvals(Q_all).real>=0).all()))
        # Setup equality constraints
        Aeq = torch.zeros(4 * num_segments + 2, num_coefficients * num_segments)
        beq = torch.zeros(4 * num_segments + 2)
        Aeq[0:3, :num_coefficients] = torch.stack([
            self.calculate_time_vector(time_stamps[0], poly_order, 0),
            self.calculate_time_vector(time_stamps[0], poly_order, 1),
            self.calculate_time_vector(time_stamps[0], poly_order, 2)])
        Aeq[3:6, -num_coefficients:] = torch.stack([
            self.calculate_time_vector(time_stamps[-1], poly_order, 0),
            self.calculate_time_vector(time_stamps[-1], poly_order, 1),
            self.calculate_time_vector(time_stamps[-1], poly_order, 2)])
        beq[0:6] = torch.tensor([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc])
        # beq[:6] = torch.cat((start_pos, start_vel, start_acc, end_pos, end_vel, end_acc))


        # Middle waypoints constraints
        num_eq_constraints = 6
        for i in range(1, num_segments):
            Aeq[num_eq_constraints, i * num_coefficients:(i + 1) * num_coefficients] = self.calculate_time_vector(time_stamps[i], poly_order, 0)
            beq[num_eq_constraints] = waypoints[i]
            num_eq_constraints += 1

        # Continuity constraints
        for i in range(1, num_segments):
            time_vector_p = self.calculate_time_vector(time_stamps[i], poly_order, 0)
            time_vector_v = self.calculate_time_vector(time_stamps[i], poly_order, 1)
            time_vector_a = self.calculate_time_vector(time_stamps[i], poly_order, 2)
            Aeq[num_eq_constraints:num_eq_constraints + 3, (i - 1) * num_coefficients:(i + 1) * num_coefficients] = torch.stack([
                torch.cat([time_vector_p, -time_vector_p]),
                torch.cat([time_vector_v, -time_vector_v]),
                torch.cat([time_vector_a, -time_vector_a])])
            num_eq_constraints += 3

        # Convert to the specified data type and device
        G_dummy = torch.zeros(1, Q_all.size(0), Q_all.size(0), dtype=torch.float64)
        h_dummy = torch.zeros(1, Q_all.size(0), dtype=torch.float64)
        Q_all += torch.eye(Q_all.size(0), dtype=torch.float64) * 1e-6
        Q_all = Q_all.to(dtype=self.dtype, device=self.device)
        b_all = b_all.to(dtype=self.dtype, device=self.device)
        Aeq = Aeq.to(dtype=self.dtype, device=self.device)
        beq = beq.to(dtype=self.dtype, device=self.device)
        G_dummy = G_dummy.to(dtype=self.dtype, device=self.device)
        h_dummy = h_dummy.to(dtype=self.dtype, device=self.device)

        # Solve the QP problem
        solver_options = {'eps':1e-24, 'maxIter': 100, 'solver': qpth.qp.QPSolvers.PDIPM_BATCHED}
        solution = QPFunction(verbose=-1, **solver_options)(Q_all, b_all, G_dummy, h_dummy, Aeq, beq)
        polynomial_coefficients = solution.view(num_segments, num_coefficients).transpose(0, 1)
        return polynomial_coefficients

    def evaluate_polynomial(self,polynomial_coefficients, time, derivative_order):
        """
        Evaluate a polynomial at a given time.

        Args:
            polynomial_coefficients (Tensor): coefficients of the polynomial
            time (float): time at which to evaluate the polynomial
            derivative_order (int): derivative order

        Returns:
            float: value of the polynomial at the given time
        """
        value = 0
        polynomial_order = len(polynomial_coefficients) - 1
        if derivative_order <= 0:
            for i in range(polynomial_order + 1):
                value += polynomial_coefficients[i] * time ** i
        else:
            for i in range(derivative_order, polynomial_order + 1):
                value += polynomial_coefficients[i] * np.prod(range(i - derivative_order + 1, i + 1)) * time ** (i - derivative_order)
        return value

    def evaluate_polynomials(self,polynomial_coefficients, time_stamps, times, derivative_order):
        """
        Evaluate polynomials over a time range.

        Args:
            polynomial_coefficients (Tensor): coefficients of the polynomials
            time_stamps (Tensor): time stamps for each segment
            times (Tensor): times at which to evaluate the polynomials
            derivative_order (int): derivative order

        Returns:
            Tensor: values of the polynomials at the given times
        """
        num_points = times.size(0)
        values = torch.zeros(num_points)
        index = 0
        for i in range(num_points):
            time = times[i]
            if time < time_stamps[index]:
                values[i] = 0
            else:
                while index < len(time_stamps) - 1 and time > time_stamps[index + 1] + 0.0001:
                    index += 1
                values[i] = self.evaluate_polynomial(polynomial_coefficients[:, index], time, derivative_order)
        return values

    def plot_trajectory(self,waypoints, polys_x, polys_y, time_stamps,clear_plot=False,flag=False):
        """
        Plot the minimum snap trajectory.

        Args:
            waypoints (Tensor): waypoints
            polys_x (Tensor): polynomial coefficients for x axis
            polys_y (Tensor): polynomial coefficients for y axis
            time_stamps (Tensor): time stamps for each segment
        """
        if clear_plot:
            plt.clf()  # Clear the current figure only if specified
        plt.plot(waypoints[0], waypoints[1], '*r')
        plt.plot(waypoints[0], waypoints[1], 'b--')
        plt.title('Minimum Snap Trajectory')
        plt.xlim([-0.5, 3.0])  # Set your desired limits for x-axis
        plt.ylim([-0.5, 2.5])  # Set your desired limits for y-axis
        colors = ['g', 'r', 'c', 'm', 'y', 'k']
        for i in range(polys_x.shape[1]):
            # times = torch.arange(time_stamps[i], time_stamps[i+1], 0.01)
            times = torch.arange(time_stamps[i].item(), time_stamps[i+1].item(), 0.01, device=self.device, dtype=self.dtype)
            x_values = self.evaluate_polynomials(polys_x, time_stamps, times, 0)
            y_values = self.evaluate_polynomials(polys_y, time_stamps, times, 0)
            plt.plot(x_values.detach().numpy(), y_values.detach().numpy(), colors[i % len(colors)])
        if flag:
            plt.show()

    def plot_trajectory_and_derivatives(self, waypoints, polys_x, polys_y, time_stamps):
        # Plot trajectory
        self.plot_trajectory(waypoints, polys_x, polys_y, time_stamps)

        # Prepare the time vector for plotting derivatives
        # times = torch.arange(0, self.total_time, 0.01)
        for i in range(polys_x.shape[1]):
            times = torch.arange(time_stamps[i].item(), time_stamps[i+1].item(), 0.01, device=self.device, dtype=self.dtype)

        # Evaluate derivatives
        x_positions = self.evaluate_polynomials(polys_x, time_stamps, times, 0)
        y_positions = self.evaluate_polynomials(polys_y, time_stamps, times, 0)
        x_velocities = self.evaluate_polynomials(polys_x, time_stamps, times, 1)
        y_velocities = self.evaluate_polynomials(polys_y, time_stamps, times, 1)
        x_accelerations = self.evaluate_polynomials(polys_x, time_stamps, times, 2)
        y_accelerations = self.evaluate_polynomials(polys_y, time_stamps, times, 2)
        x_jerks = self.evaluate_polynomials(polys_x, time_stamps, times, 3)
        y_jerks = self.evaluate_polynomials(polys_y, time_stamps, times, 3)
        x_snap = self.evaluate_polynomials(polys_x, time_stamps, times, 4)
        y_snap = self.evaluate_polynomials(polys_y, time_stamps, times, 4)

        # Plot derivatives
        fig, axs = plt.subplots(5, 2, figsize=(12, 16))
        axs[0, 0].plot(times.detach().numpy(), x_positions.detach().numpy())
        axs[0, 0].set_title('X Position')
        axs[0, 1].plot(times.detach().numpy(), y_positions.detach().numpy())
        axs[0, 1].set_title('Y Position')

        axs[1, 0].plot(times.detach().numpy(), x_velocities.detach().numpy())
        axs[1, 0].set_title('X Velocity')
        axs[1, 1].plot(times.detach().numpy(), y_velocities.detach().numpy())
        axs[1, 1].set_title('Y Velocity')

        axs[2, 0].plot(times.detach().numpy(), x_accelerations.detach().numpy())
        axs[2, 0].set_title('X Acceleration')
        axs[2, 1].plot(times.detach().numpy(), y_accelerations.detach().numpy())
        axs[2, 1].set_title('Y Acceleration')

        axs[3, 0].plot(times.detach().numpy(), x_jerks.detach().numpy())
        axs[3, 0].set_title('X Jerk')
        axs[3, 1].plot(times.detach().numpy(), y_jerks.detach().numpy())
        axs[3, 1].set_title('Y Jerk')

        axs[4, 0].plot(times.detach().numpy(), x_snap.detach().numpy())
        axs[4, 0].set_title('X Snap')
        axs[4, 1].plot(times.detach().numpy(), y_snap.detach().numpy())
        axs[4, 1].set_title('Y Snap')

        for ax in axs.flat:
            ax.set(xlabel='Time (s)', ylabel='Value')
            ax.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()


    def forward(self, waypoints,total_time):
        # num_segments = waypoints.size(1) - 1  # Assuming waypoints is 2 x N

        # time_stamps = self.init_time_segments(num_segments)
        optimized_time_segments = self.optimize_time_segments(waypoints,total_time)
        x=[j-i for i, j in zip(optimized_time_segments[:-1], optimized_time_segments[1:])]
        print("difference is :", x)
        print(optimized_time_segments)
        if torch.cuda.is_available():
            polys_x = self.solve_minimum_snap(waypoints[0], optimized_time_segments, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
            polys_y = self.solve_minimum_snap(waypoints[1], optimized_time_segments, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
        else:
            polys_x = self.solve_minimum_snap(waypoints[0], optimized_time_segments, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
            polys_y = self.solve_minimum_snap(waypoints[1], optimized_time_segments, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
        return polys_x, polys_y,optimized_time_segments

def demo_minimum_snap_simple():
    waypoints = torch.tensor([[0, 0], [1, 2], [2, -1], [4, 8], [5, 2]], dtype=torch.float64).t()
    # waypoints = torch.tensor([[0, 0], [1, 0], [1, 2], [0,2]], dtype=torch.float64).t()
    start_vel, start_acc, end_vel, end_acc = torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0])
    total_time = 25.0
    poly_order = 5

    planner = UAVTrajectoryPlanner(waypoints,total_time, poly_order, start_vel, start_acc, end_vel, end_acc, dtype=torch.float64, device='cpu')
    polys_x, polys_y, time_stamps = planner(waypoints,total_time)
    # print(polys_x,polys_y)
    # print(time_stamps)

    # planner.plot_trajectory(waypoints, polys_x, polys_y, time_stamps, clear_plot=True,flag=True)
    # # planner.plot_trajectory_and_derivatives(waypoints, polys_x, polys_y, time_stamps)
    # values = [t.item() for t in total_cost_all]

    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(values, marker='o', linestyle='-', color='b')
    # plt.title("Total Cost Over Time")
    # plt.xlabel("Time Step")
    # plt.ylabel("Cost")
    # plt.grid(True)
    # plt.show()


demo_minimum_snap_simple()
