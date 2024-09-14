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

# ===========================
# 1. Generate Sample Racetrack Data
# ===========================

def generate_circular_track(radius=100, num_points=360, w_tr_left=10, w_tr_right=10):
    """
    Generates a circular racetrack DataFrame for demonstration purposes.

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

# Generate sample track data
df = generate_circular_track()

# ===========================
# 2. Define Plotting Function
# ===========================

def plot_racetrack_and_path(df, path, episode_num, done, total_reward, save=False, save_dir='plots'):
    """
    Plots the racetrack boundaries and the agent's path with annotations.

    Parameters:
    - df: DataFrame containing racetrack data with columns 'x_m', 'y_m', 'w_tr_left_m', 'w_tr_right_m'.
    - path: List or array of [x, y] positions taken by the agent.
    - episode_num: Current episode number (for labeling).
    - done: Boolean indicating if the episode ended successfully or due to a failure.
    - total_reward: Total reward obtained in the episode.
    - save: Whether to save the plot as an image.
    - save_dir: Directory to save the plot images.
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
        # self.steering_values = np.linspace(-1.0, 1.0, num=9)  # [-1.0, -0.75, -0.5, ..., 0.5, 0.75, 1.0]
        # self.throttle_values = np.linspace(0.0, 1.0, num=5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

        self.steering_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.throttle_values = np.array([0.0, 0.5, 1.0])
        self.action_space = spaces.Discrete(len(self.steering_values) * len(self.throttle_values))
        self.action_space_list = [ (s, t) for s in self.steering_values for t in self.throttle_values ]
        
        # Observations: x, y, velocity, sin(orientation), cos(orientation), distance_left, distance_right
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        # Initialize state
        self.max_steps = max_steps
        self.reset()
    
    def reset(self):
        # Starting at the first point
        self.position = np.array([self.x[0], self.y[0]], dtype=np.float32)
        self.velocity = 0.0
        self.orientation = 0.0  # Facing along the track initially
        self.lap = 0
        self.done = False
        self.current_step = 0
        return self._get_obs()
    
    def step(self, action_idx):
        if self.done:
            # If episode is done, no further steps are processed
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
            return self._get_obs(), reward, self.done, {}
        
        # Check for lap completion
        if self._check_lap_completion():
            reward += 100.0
            self.done = True  # Terminate episode upon lap completion
            return self._get_obs(), reward, self.done, {}
        
        # Check max steps
        if self.current_step >= self.max_steps:
            self.done = True  # Terminate episode upon reaching max steps
        
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
    
    def _check_lap_completion(self):
        # Simple lap completion check
        lap_threshold = 5.0  # Distance threshold to start point
        distance = np.sqrt((self.position[0] - self.x[0])**2 + (self.position[1] - self.y[0])**2)
        if distance < lap_threshold and self.current_step > 100:
            self.lap += 1
            return True
        return False

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
# ===========================

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
num_episodes = 1000
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

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    done = False
    path = [env.position.copy()]  # Initialize path with the starting position
    
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()
        
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
    
    # Plot the path at specified frequency
    if episode % plot_frequency == 0 or episode == 1:
        plot_racetrack_and_path(
            df, 
            path, 
            episode_num=episode, 
            done=done, 
            total_reward=total_reward, 
            save=True, 
            save_dir=plot_directory
        )
    
    # Print progress
    print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

print("Training completed.")







# def test_environment(env):
#     state = env.reset()
#     done = False
#     step = 0
#     path = [env.position.copy()]
    
#     while not done and step < 100:
#         # Take random actions
#         action = env.action_space.sample()
#         next_state, reward, done, _ = env.step(action)
#         path.append(env.position.copy())
#         print(f"Step {step+1}: Reward = {reward:.2f}, Done = {done}")
#         step += 1
    
#     # Plot the path
#     plot_racetrack_and_path(df, path, episode_num='Test', done=done, total_reward=reward, save=False)
#     print(f"Test Episode ended at step {step} with total reward {reward:.2f}.")

# # Initialize environment
# test_env = RaceTrackEnv(df)

# # Run the test
# test_environment(test_env)
