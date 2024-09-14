import gym
from gym import spaces
import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import animation
from matplotlib.collections import LineCollection

# ===========================
# 1. Generate Sample Racetrack Data
# ===========================

def generate_circular_track(radius=100, num_points=360, w_tr_left=10, w_tr_right=10):
    """
    Generates a circular racetrack DataFrame for validation purposes.

    Parameters:
    - radius: Radius of the circular track centerline.
    - num_points: Number of points defining the track.
    - w_tr_left: Fixed left boundary width.
    - w_tr_right: Fixed right boundary width.

    Returns:
    - df: pandas DataFrame with columns 'x_m', 'y_m', 'w_tr_left_m', 'w_tr_right_m'.
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_center = radius * np.cos(angles)
    y_center = radius * np.sin(angles)
    w_tr_left_m = np.full(num_points, w_tr_left)
    w_tr_right_m = np.full(num_points, w_tr_right)

    df = pd.DataFrame({
        'x_m': x_center,
        'y_m': y_center,
        'w_tr_left_m': w_tr_left_m,
        'w_tr_right_m': w_tr_right_m
    })
    return df

def generate_oval_track(long_radius=120, short_radius=80, num_points=360, w_tr_left=10, w_tr_right=10):
    """
    Generates an oval racetrack DataFrame.
    
    Parameters:
    - long_radius: Radius of the long axis of the oval.
    - short_radius: Radius of the short axis of the oval.
    - num_points: Number of points defining the track.
    - w_tr_left: Fixed left boundary width.
    - w_tr_right: Fixed right boundary width.
    
    Returns:
    - df: pandas DataFrame with columns 'x_m', 'y_m', 'w_tr_left_m', 'w_tr_right_m'.
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_center = long_radius * np.cos(angles)
    y_center = short_radius * np.sin(angles)
    w_tr_left_m = np.full(num_points, w_tr_left)
    w_tr_right_m = np.full(num_points, w_tr_right)

    df = pd.DataFrame({
        'x_m': x_center,
        'y_m': y_center,
        'w_tr_left_m': w_tr_left_m,
        'w_tr_right_m': w_tr_right_m
    })
    return df

# Generate sample track data
df = generate_circular_track()
# df = generate_oval_track()
# df = pd.read_csv('Data/test_model.csv')

# ===========================
# 2. Define Plotting Function
# ===========================


def plot_racetrack_and_controls(df, path, throttle_values, steering_values, episode_num, done, total_reward, save=False, save_dir='plots', orientation_angle=None):
    """
    Plots the racetrack boundaries, the agent's path, and the throttle and steering time series.
    
    Parameters:
    - df: DataFrame containing racetrack data.
    - path: List or array of [x, y] positions taken by the agent.
    - throttle_values: List of throttle values over the episode.
    - steering_values: List of steering values over the episode.
    - episode_num: Current episode number.
    - done: Boolean indicating if the episode ended successfully or due to a failure.
    - total_reward: Total reward obtained in the episode.
    - save: Whether to save the plot as an image.
    - save_dir: Directory to save the plot images.
    - orientation_angle: Angle (in radians) indicating the agent's orientation at the end of the path.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Create two subplots: one for racetrack and one for time series

    # Plot the racetrack and path
    # Calculate left and right boundaries
    left_boundary = []
    right_boundary = []
    num_points = len(df)
    for idx in range(num_points):
        if idx < num_points - 1:
            track_dx = df['x_m'][idx + 1] - df['x_m'][idx]
            track_dy = df['y_m'][idx + 1] - df['y_m'][idx]
        else:
            track_dx = df['x_m'][idx] - df['x_m'][idx - 1]
            track_dy = df['y_m'][idx] - df['y_m'][idx - 1]
        track_angle = math.atan2(track_dy, track_dx)
        perp_angle = track_angle + math.pi / 2

        boundary_left_x = df['x_m'][idx] + df['w_tr_left_m'][idx] * math.cos(perp_angle)
        boundary_left_y = df['y_m'][idx] + df['w_tr_left_m'][idx] * math.sin(perp_angle)
        left_boundary.append((boundary_left_x, boundary_left_y))

        boundary_right_x = df['x_m'][idx] - df['w_tr_right_m'][idx] * math.cos(perp_angle)
        boundary_right_y = df['y_m'][idx] - df['w_tr_right_m'][idx] * math.sin(perp_angle)
        right_boundary.append((boundary_right_x, boundary_right_y))

    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)

    # Plot the centerline and boundaries
    axs[0].plot(df['x_m'], df['y_m'], 'k--', label='Centerline')
    axs[0].plot(left_boundary[:, 0], left_boundary[:, 1], 'r', label='Left Boundary')
    axs[0].plot(right_boundary[:, 0], right_boundary[:, 1], 'b', label='Right Boundary')

    # Plot the agent's path
    path = np.array(path)
    axs[0].plot(path[:, 0], path[:, 1], 'g-', label='Agent Path')
    axs[0].plot(path[0, 0], path[0, 1], 'go', label='Start')
    axs[0].plot(path[-1, 0], path[-1, 1], 'ro', label='End')

    # Plot orientation arrow at the end of the path
    if orientation_angle is not None:
        arrow_length = 5  # Adjust as needed
        axs[0].arrow(path[-1, 0], path[-1, 1],
                     arrow_length * math.cos(orientation_angle),
                     arrow_length * math.sin(orientation_angle),
                     head_width=2, head_length=2, fc='k', ec='k', label='Orientation')

    # Add the start/finish line
    axs[0].plot([env.start_line['start'][0], env.start_line['end'][0]],
                [env.start_line['start'][1], env.start_line['end'][1]], 'm-', label='Start/Finish Line', linewidth=2)

    # Add annotations
    status = 'Success' if done and _check_lap_completion_env(env) else 'Failure'
    axs[0].set_title(f'Episode {episode_num} - {status}\nTotal Reward: {total_reward:.2f}')
    axs[0].set_xlabel('X Position (m)')
    axs[0].set_ylabel('Y Position (m)')
    axs[0].legend()
    axs[0].axis('equal')
    axs[0].grid(True)

    # Plot the throttle and steering time series
    timesteps = np.arange(len(throttle_values))
    axs[1].plot(timesteps, throttle_values, label='Throttle', color='blue')
    axs[1].plot(timesteps, steering_values, label='Steering', color='green')
    axs[1].set_title('Throttle and Steering Time Series')
    axs[1].set_xlabel('Timesteps')
    axs[1].set_ylabel('Value')
    axs[1].legend()
    axs[1].grid(True)

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/episode_{episode_num}_with_controls.png")
    else:
        plt.show()

    plt.close()
    
def plot_racetrack_and_path(df, path, episode_num, done, total_reward, save=False, save_dir='plots', orientation_angle=None):
    """
    Plots the racetrack boundaries and the agent's path with annotations.

    Parameters:
    - df: DataFrame containing racetrack data.
    - path: List or array of [x, y] positions taken by the agent.
    - episode_num: Current episode number.
    - done: Boolean indicating if the episode ended successfully or due to a failure.
    - total_reward: Total reward obtained in the episode.
    - save: Whether to save the plot as an image.
    - save_dir: Directory to save the plot images.
    - orientation_angle: Angle (in radians) indicating the agent's orientation at the end of the path.
    """
    plt.figure(figsize=(10, 6))

    # Calculate left and right boundaries
    left_boundary = []
    right_boundary = []
    num_points = len(df)
    for idx in range(num_points):
        if idx < num_points - 1:
            track_dx = df['x_m'][idx + 1] - df['x_m'][idx]
            track_dy = df['y_m'][idx + 1] - df['y_m'][idx]
        else:
            track_dx = df['x_m'][idx] - df['x_m'][idx - 1]
            track_dy = df['y_m'][idx] - df['y_m'][idx - 1]
        track_angle = math.atan2(track_dy, track_dx)
        perp_angle = track_angle + math.pi / 2

        boundary_left_x = df['x_m'][idx] + df['w_tr_left_m'][idx] * math.cos(perp_angle)
        boundary_left_y = df['y_m'][idx] + df['w_tr_left_m'][idx] * math.sin(perp_angle)
        left_boundary.append((boundary_left_x, boundary_left_y))

        boundary_right_x = df['x_m'][idx] - df['w_tr_right_m'][idx] * math.cos(perp_angle)
        boundary_right_y = df['y_m'][idx] - df['w_tr_right_m'][idx] * math.sin(perp_angle)
        right_boundary.append((boundary_right_x, boundary_right_y))

    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)

    # Plot centerline
    plt.plot(df['x_m'], df['y_m'], 'k--', label='Centerline')

    # Plot left and right boundaries
    plt.plot(left_boundary[:, 0], left_boundary[:, 1], 'r', label='Left Boundary')
    plt.plot(right_boundary[:, 0], right_boundary[:, 1], 'b', label='Right Boundary')

    # Plot agent's path
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'g-', label='Agent Path')
    plt.plot(path[0, 0], path[0, 1], 'go', label='Start')
    plt.plot(path[-1, 0], path[-1, 1], 'ro', label='End')

    # Plot orientation arrow at the end of the path
    if orientation_angle is not None:
        arrow_length = 5  # Adjust as needed
        plt.arrow(path[-1, 0], path[-1, 1],
                  arrow_length * math.cos(orientation_angle),
                  arrow_length * math.sin(orientation_angle),
                  head_width=2, head_length=2, fc='k', ec='k', label='Orientation')

    # Add the start/finish line
    plt.plot([env.start_line['start'][0], env.start_line['end'][0]],
             [env.start_line['start'][1], env.start_line['end'][1]], 'm-', label='Start/Finish Line', linewidth=2)

    # Add annotations
    status = 'Success' if done and _check_lap_completion_env(env) else 'Failure'
    plt.title(f'Episode {episode_num} - {status}\nTotal Reward: {total_reward:.2f}')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/episode_{episode_num}.png")
    else:
        plt.show()

    plt.close()


def _check_lap_completion_env(env):
    """
    Helper function to check lap completion from the environment.
    """
    return env.lap > 0

# ===========================
# 3. Define RaceTrack Environment
# ===========================
class RaceTrackEnv(gym.Env):
    def __init__(self, df, max_steps=1000):
        super(RaceTrackEnv, self).__init__()
        
        # Load racetrack data
        self.x = df['x_m'].values
        self.y = df['y_m'].values
        self.w_tr_right = df['w_tr_right_m'].values
        self.w_tr_left = df['w_tr_left_m'].values
        
        # Define action and observation space
        # Actions: Steering (-1 to 1), Throttle (0 to 1)
        self.action_space_continuous = spaces.Box(low=np.array([-1.0, 0.0]), 
                                                 high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Discretize actions
        self.steering_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.throttle_values = np.array([0.0, 0.5, 1.0])
        self.action_space = spaces.Discrete(len(self.steering_values) * len(self.throttle_values))
        self.action_space_list = [ (s, t) for s in self.steering_values for t in self.throttle_values ]
        
        # Observations: x, y, velocity, sin(orientation), cos(orientation), distance_left, distance_right
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self._set_start_finish_line()

        # Initialize state
        self.max_steps = max_steps
        self.reset()
    
    
    def _set_start_finish_line(self):
        """
        Sets the start/finish line to be perpendicular to the first segment of the track.
        """
        # Calculate the first track segment direction
        track_dx = self.x[1] - self.x[0]
        track_dy = self.y[1] - self.y[0]
        track_angle = math.atan2(track_dy, track_dx)

        # Perpendicular to the track's direction (rotate by 90 degrees)
        perp_angle = track_angle + math.pi / 2

        # Extend the start/finish line across the track's width
        line_length = max(self.w_tr_left[0], self.w_tr_right[0]) * 1.5  # Slightly larger than track width

        # Start and end points of the line, positioned at the first point (x[0], y[0])
        self.start_line = {
            'start': np.array([
                self.x[0] + line_length * math.cos(perp_angle),
                self.y[0] + line_length * math.sin(perp_angle)
            ]),
            'end': np.array([
                self.x[0] - line_length * math.cos(perp_angle),
                self.y[0] - line_length * math.sin(perp_angle)
            ])
        }


    
    def reset(self):
        # Starting at the first point
        self.position = np.array([self.x[0], self.y[0]], dtype=np.float32)
        self.velocity = 0.0

        # Compute the initial orientation based on the track's direction at the starting point
        if len(self.x) > 1:
            track_dx = self.x[1] - self.x[0]
            track_dy = self.y[1] - self.y[0]
            track_angle = math.atan2(track_dy, track_dx)
            self.orientation = track_angle
        else:
            self.orientation = 0.0  # Fallback if track has only one point

        print(f"Initial Orientation: {math.degrees(self.orientation):.2f} degrees")

        self.lap = 0
        self.lap_completed = False  # Reset lap completion flag at the start of each episode
        self.done = False
        self.current_step = 0
        return self._get_obs()

        
    def step(self, action_idx):
        if self.done:
            # If episode is done, no further steps are processed
            print("Episode already done. No further steps are processed.")
            return self._get_obs(), 0.0, self.done, {}

        # Decode action
        steering, throttle = self.action_space_list[action_idx]
        self.current_step += 1

        # Update velocity
        self.velocity += throttle * 1.0  # Simplified acceleration
        self.velocity = max(min(self.velocity, 10.0), 0.0)  # Clamp velocity

        # Update orientation
        self.orientation += steering * 0.1  # Steering sensitivity

        # Normalize orientation between -pi and pi
        self.orientation = (self.orientation + math.pi) % (2 * math.pi) - math.pi

        # Update position
        dx = self.velocity * math.cos(self.orientation)
        dy = self.velocity * math.sin(self.orientation)
        previous_position = self.position.copy()  # Store previous position
        self.position += np.array([dx, dy], dtype=np.float32)

        # Calculate distance to boundaries
        distance_left, distance_right = self._calculate_boundary_distances()

        # Calculate reward
        reward = self.velocity * 0.2  # Encourage speed
        reward += throttle * 0.5

        # Proximity penalty
        min_distance = min(distance_left, distance_right)
        proximity_threshold = 5.0  # Threshold can be adjusted
        if min_distance < proximity_threshold:
            reward -= (proximity_threshold - min_distance) * 0.2

        # Check for boundary violation
        if distance_left < 0 or distance_right < 0:
            reward -= 10.0
            self.done = True  # Terminate episode immediately
            print(f"Boundary violation at step {self.current_step}. Episode terminated.")
            return self._get_obs(), reward, self.done, {}

        # Check for lap completion
        if self._check_lap_completion(previous_position):
            reward += 100.0
            self.done = True  # Terminate episode upon lap completion
            print(f"Lap completed at step {self.current_step}. Episode terminated.")
            return self._get_obs(), reward, self.done, {}

        # Check max steps
        if self.current_step >= self.max_steps:
            self.done = True  # Terminate episode upon reaching max steps
            print(f"Max steps reached ({self.max_steps}). Episode terminated.")

        return self._get_obs(), reward, self.done, {}

    
    
    def _get_obs(self):
        distance_left, distance_right = self._calculate_boundary_distances()
        # Normalize position (assuming track coordinates are within +/- 150 meters)
        normalized_x = self.position[0] / 150.0
        normalized_y = self.position[1] / 150.0
        # Normalize velocity
        normalized_velocity = self.velocity / 10.0  # Max velocity is 10
        # Represent orientation using sine and cosine
        orientation_sin = math.sin(self.orientation)
        orientation_cos = math.cos(self.orientation)
        # Normalize distances (assuming max track width is 50 meters)
        normalized_distance_left = distance_left / 50.0
        normalized_distance_right = distance_right / 50.0
        
        return np.array([
            normalized_x,
            normalized_y,
            normalized_velocity,
            orientation_sin,
            orientation_cos,
            normalized_distance_left,
            normalized_distance_right
        ], dtype=np.float32)
    
    def _calculate_boundary_distances(self):
        # Find the nearest track segment
        distances = np.sqrt((self.x - self.position[0])**2 + (self.y - self.position[1])**2)
        nearest_idx = np.argmin(distances)
        
        # Calculate the direction of the track at the nearest segment
        if nearest_idx < len(self.x) - 1:
            track_dx = self.x[nearest_idx + 1] - self.x[nearest_idx]
            track_dy = self.y[nearest_idx + 1] - self.y[nearest_idx]
        else:
            track_dx = self.x[nearest_idx] - self.x[nearest_idx - 1]
            track_dy = self.y[nearest_idx] - self.y[nearest_idx - 1]
        
        track_angle = math.atan2(track_dy, track_dx)
        
        # Calculate perpendicular direction
        perp_angle = track_angle + math.pi / 2
        
        # Calculate the boundary points
        boundary_left = self.position + np.array([
            self.w_tr_left[nearest_idx] * math.cos(perp_angle),
            self.w_tr_left[nearest_idx] * math.sin(perp_angle)
        ])
        boundary_right = self.position - np.array([
            self.w_tr_right[nearest_idx] * math.cos(perp_angle),
            self.w_tr_right[nearest_idx] * math.sin(perp_angle)
        ])
        
        # Calculate distances to the boundaries using vector projection
        # Vector from nearest track point to vehicle position
        vec_to_vehicle = self.position - np.array([self.x[nearest_idx], self.y[nearest_idx]])
        
        # Unit vector in the perpendicular direction
        perp_unit_vector = np.array([
            math.cos(perp_angle),
            math.sin(perp_angle)
        ])
        
        # Project the vector onto the perpendicular axis
        proj = np.dot(vec_to_vehicle, perp_unit_vector)
        
        # Calculate distances
        distance_left = self.w_tr_left[nearest_idx] - proj
        distance_right = self.w_tr_right[nearest_idx] + proj
        
        return distance_left, distance_right
    
    def _check_lap_completion(self, previous_position):
        """
        Checks if the agent has completed a lap by crossing the start line from below to above.
        
        Parameters:
        - previous_position: The agent's position in the previous step.
        
        Returns:
        - True if a lap is completed, False otherwise.
        """
        # Only check for lap completion after a certain number of steps to avoid false positives at the start
        lap_check_threshold = 10  # Only check for lap completion after 100 steps
        if self.current_step < lap_check_threshold:
            return False

        if self.lap_completed:
            return False  # Already completed a lap in this episode

        # Log position and previous position for debugging
        # print(f"Step {self.current_step}: Checking lap completion.")
        # print(f"Previous position: {previous_position}, Current position: {self.position}")

        # Check if the agent crosses the start/finish line
        if self._lines_intersect(previous_position, self.position, self.start_line['start'], self.start_line['end']):
            # Determine the direction of crossing
            movement_vector = self.position - previous_position
            start_line_vector = self.start_line['end'] - self.start_line['start']
            start_line_normal = np.array([-start_line_vector[1], start_line_vector[0]])  # Rotate 90 degrees

            # Compute the dot product to determine direction
            dot_product = np.dot(movement_vector, start_line_normal)
            print(f"Dot product for lap completion direction: {dot_product}")

            if dot_product > 0:
                # Crossing in the correct direction
                self.lap += 1
                self.lap_completed = True  # Prevent multiple lap counts
                print(f"Lap {self.lap} completed at step {self.current_step}.")
                return True
            else:
                # Crossing in the wrong direction; do not count
                print("Crossed in the wrong direction. Lap not counted.")
                return False
        return False

    
    def _lines_intersect(self, p1, p2, q1, q2):
        """
        Checks if the line segment p1-p2 intersects with the line segment q1-q2.

        Parameters:
        - p1, p2: Endpoints of the first line segment.
        - q1, q2: Endpoints of the second line segment.

        Returns:
        - True if the line segments intersect, False otherwise.
        """
        def ccw(a, b, c):
            return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
        
        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

# ===========================
# 4. Define the DQN Network
# ===========================

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ===========================
# 5. Initialize Environment and Networks
# ==========================

# Hyperparameters
state_size = 7  # x, y, velocity, sin(orientation), cos(orientation), distance_left, distance_right
action_size = 15  # 5 steering * 3 throttle
hidden_size = 128
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 1e-3
memory_size = 10000
target_update_freq = 10  # in episodes
num_episodes = 2000
plot_frequency = 50  # Plot every N episodes

# Initialize environment
env = RaceTrackEnv(df)

# Initialize DQN networks
policy_net = DQN(state_size, action_size, hidden_size)
target_net = DQN(state_size, action_size, hidden_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Initialize optimizer and loss function
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Initialize experience replay memory
memory = deque(maxlen=memory_size)

# Create directory for plots
plot_directory = 'plots'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

# ===========================
# 6. Training Loop with Enhanced Termination
# ===========================

# Initialize lists to store paths
paths_to_store = []  # To store every 100th episode
final_path = []      # To store the final episode's path

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    done = False
    path = [env.position.copy()]  # Initialize path with the starting position
    throttle_values = []
    steering_values = []

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()

        # Decode action into steering and throttle values
        steering, throttle = env.action_space_list[action_idx]
        throttle_values.append(throttle)
        steering_values.append(steering)

        # Execute action
        next_state, reward, done, _ = env.step(action_idx)
        memory.append((state, action_idx, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Record the position
        path.append(env.position.copy())

        # Experience replay
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones).unsqueeze(1)

            # Current Q values
            current_q = policy_net(states).gather(1, actions)

            # Target Q values
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + (gamma * max_next_q * (1 - dones))

            # Compute loss
            loss = loss_fn(current_q, target_q)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update target network
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Store the final episode's path and control values
    if episode == num_episodes:
        final_path = path.copy()
        final_throttle_values = throttle_values.copy()
        final_steering_values = steering_values.copy()

    # Plot the path and controls at specified frequency
    if episode % plot_frequency == 0 or episode == 1:
        # Determine orientation angle for plotting
        if len(path) > 1:
            dx = path[-1][0] - path[-2][0]
            dy = path[-1][1] - path[-2][1]
            orientation_angle = math.atan2(dy, dx)
        else:
            orientation_angle = None

        # plot_racetrack_and_controls(
        #     df, 
        #     path, 
        #     throttle_values, 
        #     steering_values, 
        #     episode_num=episode, 
        #     done=done, 
        #     total_reward=total_reward, 
        #     save=True, 
        #     save_dir=plot_directory,
        #     orientation_angle=orientation_angle
        # )
        paths_to_store.append(path)

    
    # Print progress
    print(f"====================\n"
          f"Episode {episode} , Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    # Optional: Print lap count
    print(f"Lap Count: {env.lap}")

print("Training complete.")


def run_final_episode(env, policy_net):
    """
    Runs a single episode using the trained policy network and records positions and speeds.
    
    Returns:
    - path: List of [x, y] positions.
    - speeds: List of speeds at each position.
    """
    state = env.reset()
    done = False
    path = [env.position.copy()]
    speeds = [env.velocity]
    
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action_idx = torch.argmax(q_values).item()
        
        next_state, reward, done, _ = env.step(action_idx)
        path.append(env.position.copy())
        speeds.append(env.velocity)
        state = next_state
    
    return path, speeds

def create_final_animation_with_speed(df, path, speeds, save=False, save_path='final_racing_animation.gif'):
    """
    Creates an animation of the final racing line with speed color map.
    
    Parameters:
    - df: DataFrame containing racetrack data.
    - path: List of [x, y] positions taken by the agent.
    - speeds: List of speeds corresponding to each position.
    - save: Whether to save the animation as a GIF.
    - save_path: Path to save the animation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate left and right boundaries
    left_boundary = []
    right_boundary = []
    num_points = len(df)
    for idx in range(num_points):
        if idx < num_points - 1:
            track_dx = df['x_m'][idx + 1] - df['x_m'][idx]
            track_dy = df['y_m'][idx + 1] - df['y_m'][idx]
        else:
            track_dx = df['x_m'][idx] - df['x_m'][idx - 1]
            track_dy = df['y_m'][idx] - df['y_m'][idx - 1]
        track_angle = math.atan2(track_dy, track_dx)
        perp_angle = track_angle + math.pi / 2
        
        boundary_left_x = df['x_m'][idx] + df['w_tr_left_m'][idx] * math.cos(perp_angle)
        boundary_left_y = df['y_m'][idx] + df['w_tr_left_m'][idx] * math.sin(perp_angle)
        left_boundary.append((boundary_left_x, boundary_left_y))
        
        boundary_right_x = df['x_m'][idx] - df['w_tr_right_m'][idx] * math.cos(perp_angle)
        boundary_right_y = df['y_m'][idx] - df['w_tr_right_m'][idx] * math.sin(perp_angle)
        right_boundary.append((boundary_right_x, boundary_right_y))
    
    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)
    
    # Plot centerline and boundaries
    ax.plot(df['x_m'], df['y_m'], 'k--', label='Centerline')
    ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'r', label='Left Boundary')
    ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'b', label='Right Boundary')
    
    # Convert path and speeds to numpy arrays
    path = np.array(path)
    speeds = np.array(speeds)
    
    # Normalize speeds for color mapping
    norm = plt.Normalize(speeds.min(), speeds.max())
    cmap = plt.cm.viridis
    
    # Create a scatter plot where color represents speed
    scatter = ax.scatter(path[:, 0], path[:, 1], c=speeds, cmap=cmap, norm=norm, s=10, label='Agent Path')
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed (m/s)')
    
    # Plot start and end points
    ax.plot(path[0, 0], path[0, 1], 'go', label='Start')
    ax.plot(path[-1, 0], path[-1, 1], 'ro', label='End')
    
    ax.set_title('Final Learned Racing Line with Speed Color Map')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    if save:
        ani = animation.FuncAnimation(fig, lambda i: None, frames=1, blit=False)
        ani.save(save_path, writer='imagemagick')
        print(f"Final racing line animation saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

# Run a final episode to collect the path and speeds
final_path, final_speeds = run_final_episode(env, policy_net)




def create_multiple_paths_animation(df, paths, save=False, save_path='multiple_paths_animation.gif'):
    """
    Creates an animation showing the paths of every 10th episode.
    
    Parameters:
    - df: DataFrame containing racetrack data.
    - paths: List of paths, where each path is a list of [x, y] positions.
    - save: Whether to save the animation as a GIF.
    - save_path: Path to save the animation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate left and right boundaries
    left_boundary = []
    right_boundary = []
    num_points = len(df)
    for idx in range(num_points):
        if idx < num_points - 1:
            track_dx = df['x_m'][idx + 1] - df['x_m'][idx]
            track_dy = df['y_m'][idx + 1] - df['y_m'][idx]
        else:
            track_dx = df['x_m'][idx] - df['x_m'][idx - 1]
            track_dy = df['y_m'][idx] - df['y_m'][idx - 1]
        track_angle = math.atan2(track_dy, track_dx)
        perp_angle = track_angle + math.pi / 2
        
        boundary_left_x = df['x_m'][idx] + df['w_tr_left_m'][idx] * math.cos(perp_angle)
        boundary_left_y = df['y_m'][idx] + df['w_tr_left_m'][idx] * math.sin(perp_angle)
        left_boundary.append((boundary_left_x, boundary_left_y))
        
        boundary_right_x = df['x_m'][idx] - df['w_tr_right_m'][idx] * math.cos(perp_angle)
        boundary_right_y = df['y_m'][idx] - df['w_tr_right_m'][idx] * math.sin(perp_angle)
        right_boundary.append((boundary_right_x, boundary_right_y))
    
    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)
    
    # Plot centerline and boundaries
    ax.plot(df['x_m'], df['y_m'], 'k--', label='Centerline')
    ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'r', label='Left Boundary')
    ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'b', label='Right Boundary')
    
    # Initialize list of Line2D objects for each path
    lines = [ax.plot([], [], lw=2, label=f'Episode {i*10}')[0] for i in range(1, len(paths)+1)]
    
    ax.set_title('Paths of Every 10th Episode')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    
    def animate(frame):
        for idx, line in enumerate(lines):
            path = np.array(paths[idx])
            if frame < len(path):
                line.set_data(path[:frame+1, 0], path[:frame+1, 1])
        return lines
    
    # Determine the maximum number of frames
    max_frames = max(len(path) for path in paths)
    
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=max_frames, interval=100, blit=True)
    
    if save:
        ani.save(save_path, writer='imagemagick')
        print(f"Multiple paths animation saved to {save_path}")
    else:
        plt.show()
    
    plt.close()



def create_final_animation_with_speed_linecollection(df, path, speeds, save=False, save_path='final_racing_animation.gif'):
    """
    Creates an animation of the final racing line with speed color map using LineCollection.
    
    Parameters:
    - df: DataFrame containing racetrack data.
    - path: List of [x, y] positions taken by the agent.
    - speeds: List of speeds corresponding to each position.
    - save: Whether to save the animation as a GIF.
    - save_path: Path to save the animation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate left and right boundaries (same as before)
    left_boundary = []
    right_boundary = []
    num_points = len(df)
    for idx in range(num_points):
        if idx < num_points - 1:
            track_dx = df['x_m'][idx + 1] - df['x_m'][idx]
            track_dy = df['y_m'][idx + 1] - df['y_m'][idx]
        else:
            track_dx = df['x_m'][idx] - df['x_m'][idx - 1]
            track_dy = df['y_m'][idx] - df['y_m'][idx - 1]
        track_angle = math.atan2(track_dy, track_dx)
        perp_angle = track_angle + math.pi / 2
        
        boundary_left_x = df['x_m'][idx] + df['w_tr_left_m'][idx] * math.cos(perp_angle)
        boundary_left_y = df['y_m'][idx] + df['w_tr_left_m'][idx] * math.sin(perp_angle)
        left_boundary.append((boundary_left_x, boundary_left_y))
        
        boundary_right_x = df['x_m'][idx] - df['w_tr_right_m'][idx] * math.cos(perp_angle)
        boundary_right_y = df['y_m'][idx] - df['w_tr_right_m'][idx] * math.sin(perp_angle)
        right_boundary.append((boundary_right_x, boundary_right_y))
    
    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)
    
    # Plot centerline and boundaries
    ax.plot(df['x_m'], df['y_m'], 'k--', label='Centerline')
    ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'r', label='Left Boundary')
    ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'b', label='Right Boundary')
    
    # Convert path and speeds to numpy arrays
    path = np.array(path)
    speeds = np.array(speeds)
    
    # Normalize speeds for color mapping
    norm = plt.Normalize(speeds.min(), speeds.max())
    cmap = plt.cm.viridis
    
    # Create segments for LineCollection
    points = path.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(speeds)
    lc.set_linewidth(2)
    
    ax.add_collection(lc)
    
    # Add a colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Speed (m/s)')
    
    # Plot start and end points
    ax.plot(path[0, 0], path[0, 1], 'go', label='Start')
    ax.plot(path[-1, 0], path[-1, 1], 'ro', label='End')
    
    ax.set_title('Final Learned Racing Line with Speed Color Map')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)
    
    if save:
        plt.savefig(save_path, dpi=300)
        print(f"Final racing line with speed color map saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    
create_final_animation_with_speed_linecollection(
    df, 
    final_path, 
    final_speeds, 
    save=True, 
    save_path=os.path.join(plot_directory, 'final_racing_animation_linecollection.png')
)
# ===========================
# 7. Create Animations After Training
# ===========================

# Run a final episode to collect the path and speeds
final_path, final_speeds = run_final_episode(env, policy_net)

# Create and save the final racing line animation with speed color map
create_final_animation_with_speed(
    df, 
    final_path, 
    final_speeds, 
    save=True, 
    save_path=os.path.join(plot_directory, 'final_racing_animation.gif')
)

# Create and save the multiple paths animation
create_multiple_paths_animation(
    df, 
    paths_to_store, 
    save=True, 
    save_path=os.path.join(plot_directory, 'multiple_paths_animation.gif')
)


# Update the training loop to store throttle and steering values for plotting



print("All animations have been created and saved.")
