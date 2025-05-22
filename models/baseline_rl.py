# Baseline Reinforcement Learning Models for Mars Habitat Resource Management

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import random
from collections import deque, namedtuple
import json
import sys
from tqdm import tqdm

# Add simulation directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.rl_environment import MarsHabitatRLEnvironment
from simulations.rl_formulation import MarsHabitatRLFormulation

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for discrete action spaces
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the DQN network
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Dimension of hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    """
    Deep Q-Network agent for discrete action spaces
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', lr=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 buffer_size=10000, batch_size=64, update_every=4, hidden_dim=128):
        """
        Initialize the DQN agent
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            device (str): Device to run the model on ('cpu' or 'cuda')
            lr (float): Learning rate
            gamma (float): Discount factor
            epsilon_start (float): Starting value of epsilon for epsilon-greedy policy
            epsilon_end (float): Minimum value of epsilon
            epsilon_decay (float): Decay rate of epsilon
            buffer_size (int): Size of replay buffer
            batch_size (int): Batch size for training
            update_every (int): How often to update the network
            hidden_dim (int): Dimension of hidden layers
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.hidden_dim = hidden_dim
        
        # Q-Networks
        self.qnetwork_local = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.qnetwork_target = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(action_dim, buffer_size, batch_size)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """
        Update agent's knowledge
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Save experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def act(self, state, eval_mode=False):
        """
        Returns actions for given state as per current policy
        
        Args:
            state: Current state
            eval_mode (bool): Whether to use evaluation mode (no exploration)
            
        Returns:
            int: Chosen action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon or eval_mode:
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))
    
    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples
        
        Args:
            experiences: Tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): Interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filename):
        """
        Save the model
        
        Args:
            filename (str): Filename to save the model
        """
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """
        Load the model
        
        Args:
            filename (str): Filename to load the model from
        """
        checkpoint = torch.load(filename)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class PPONetwork(nn.Module):
    """
    Actor-Critic network for PPO
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the PPO network
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Dimension of hidden layers
        """
        super(PPONetwork, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (action_probs, state_value)
        """
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
    
    def act(self, state):
        """
        Sample action from policy
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (action, action_log_prob, state_value)
        """
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob, state_value
    
    def evaluate(self, state, action):
        """
        Evaluate action given state
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            tuple: (action_log_probs, state_values, entropy)
        """
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return action_log_probs, state_value, entropy

class PPOAgent:
    """
    Proximal Policy Optimization agent
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, K_epochs=4, hidden_dim=128):
        """
        Initialize the PPO agent
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            device (str): Device to run the model on ('cpu' or 'cuda')
            lr (float): Learning rate
            gamma (float): Discount factor
            eps_clip (float): Clipping parameter for PPO
            K_epochs (int): Number of epochs to update policy
            hidden_dim (int): Dimension of hidden layers
        """
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = PPONetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = PPONetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        """
        Select action using the policy
        
        Args:
            state: Current state
            
        Returns:
            tuple: (action, action_log_prob, state_value)
        """
        state = torch.FloatTensor(state).to(self.device)
        return self.policy_old.act(state)
    
    def update(self, memory):
        """
        Update policy using the PPO algorithm
        
        Args:
            memory: Memory buffer containing trajectories
        """
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.tensor(memory.actions, dtype=torch.int64).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        old_state_values = torch.stack(memory.state_values).to(self.device).detach()
        
        # Calculate advantages
        advantages = rewards - old_state_values.squeeze()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def save(self, filename):
        """
        Save the model
        
        Args:
            filename (str): Filename to save the model
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        """
        Load the model
        
        Args:
            filename (str): Filename to load the model from
        """
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples
    """
    
    def __init__(self, action_size, buffer_size, batch_size, device='cpu'):
        """
        Initialize a ReplayBuffer object
        
        Args:
            action_size (int): Dimension of each action
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            device (str): Device to store the tensors on
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """
        Return the current size of internal memory
        
        Returns:
            int: Size of memory
        """
        return len(self.memory)

class PPOMemory:
    """
    Memory buffer for PPO
    """
    
    def __init__(self):
        """
        Initialize a PPOMemory object
        """
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        """
        Clear the memory
        """
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.is_terminals[:]

class MarsHabitatBaselineRL:
    """
    Baseline Reinforcement Learning models for Mars Habitat Resource Management
    """
    
    def __init__(self, data_dir):
        """
        Initialize the baseline RL models
        
        Args:
            data_dir (str): Directory containing data and configuration files
        """
        self.data_dir = data_dir
        self.models_dir = os.path.join(data_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create RL formulation
        self.formulation = MarsHabitatRLFormulation(data_dir)
        
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Training parameters
        self.training_params = {
            "dqn": {
                "lr": 1e-4,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "buffer_size": 10000,
                "batch_size": 64,
                "update_every": 4,
                "hidden_dim": 128
            },
            "ppo": {
                "lr": 3e-4,
                "gamma": 0.99,
                "eps_clip": 0.2,
                "K_epochs": 4,
                "hidden_dim": 128
            }
        }
        
        print(f"Mars Habitat Baseline RL initialized")
    
    def preprocess_state(self, state_dict):
        """
        Preprocess state dictionary into flat array
        
        Args:
            state_dict (dict): State dictionary from environment
            
        Returns:
            numpy.ndarray: Flattened state array
        """
        # Extract components from state dictionary
        time_array = state_dict['time']
        env_array = state_dict['environment']
        habitat_array = state_dict['habitat']
        subsystems_array = state_dict['subsystems']
        
        # Concatenate arrays
        state_array = np.concatenate([time_array, env_array, habitat_array, subsystems_array])
        
        return state_array
    
    def train_dqn(self, env_name, num_episodes=1000, max_steps=500, eval_interval=100, 
                  save_interval=100, render=False):
        """
        Train a DQN agent
        
        Args:
            env_name (str): Name of the environment action to train on
            num_episodes (int): Number of episodes to train for
            max_steps (int): Maximum steps per episode
            eval_interval (int): Interval to evaluate the agent
            save_interval (int): Interval to save the agent
            render (bool): Whether to render the environment
            
        Returns:
            tuple: (agent, scores)
        """
        print(f"Training DQN agent on {env_name}...")
        
        # Create environment
        env = self._create_environment(env_name)
        
        # Get state and action dimensions
        state_dim = self._get_state_dim(env)
        action_dim = self._get_action_dim(env, env_name)
        
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        
        # Create agent
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            **self.training_params["dqn"]
        )
        
        # Training loop
        scores = []
        eval_scores = []
        
        for i_episode in tqdm(range(1, num_episodes+1), desc="Episodes"):
            state = env.reset()
            state = self.preprocess_state(state)
            score = 0
            
            for t in range(max_steps):
                # Select action
                action = agent.act(state)
                
                # Convert action index to environment action
                env_action = self._convert_action_index_to_env_action(action, env_name)
                
                # Take action
                next_state, reward, done, _ = env.step(env_action)
                next_state = self.preprocess_state(next_state)
                
                # Store experience
                agent.step(state, action, reward, next_state, done)
                
                # Update state and score
                state = next_state
                score += reward
                
                if render:
                    env.render()
                
                if done:
                    break
            
            # Store score
            scores.append(score)
            
            # Evaluate agent
            if i_episode % eval_interval == 0:
                eval_score = self.evaluate_agent(agent, env, env_name, num_episodes=5)
                eval_scores.append(eval_score)
                print(f"Episode {i_episode}/{num_episodes} | Average Score: {np.mean(scores[-100:]):.2f} | Eval Score: {eval_score:.2f}")
            
            # Save agent
            if i_episode % save_interval == 0:
                agent.save(os.path.join(self.models_dir, f"dqn_{env_name}_{i_episode}.pth"))
        
        # Save final agent
        agent.save(os.path.join(self.models_dir, f"dqn_{env_name}_final.pth"))
        
        # Plot scores
        self._plot_scores(scores, eval_scores, eval_interval, "DQN", env_name)
        
        return agent, scores
    
    def train_ppo(self, env_name, num_episodes=1000, max_steps=500, update_interval=4000, 
                  eval_interval=100, save_interval=100, render=False):
        """
        Train a PPO agent
        
        Args:
            env_name (str): Name of the environment action to train on
            num_episodes (int): Number of episodes to train for
            max_steps (int): Maximum steps per episode
            update_interval (int): Interval to update the agent
            eval_interval (int): Interval to evaluate the agent
            save_interval (int): Interval to save the agent
            render (bool): Whether to render the environment
            
        Returns:
            tuple: (agent, scores)
        """
        print(f"Training PPO agent on {env_name}...")
        
        # Create environment
        env = self._create_environment(env_name)
        
        # Get state and action dimensions
        state_dim = self._get_state_dim(env)
        action_dim = self._get_action_dim(env, env_name)
        
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        
        # Create agent
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            **self.training_params["ppo"]
        )
        
        # Create memory
        memory = PPOMemory()
        
        # Training loop
        scores = []
        eval_scores = []
        time_steps = 0
        
        for i_episode in tqdm(range(1, num_episodes+1), desc="Episodes"):
            state = env.reset()
            state = self.preprocess_state(state)
            score = 0
            
            for t in range(max_steps):
                time_steps += 1
                
                # Select action
                action, action_log_prob, state_value = agent.select_action(state)
                
                # Convert action index to environment action
                env_action = self._convert_action_index_to_env_action(action, env_name)
                
                # Take action
                next_state, reward, done, _ = env.step(env_action)
                next_state = self.preprocess_state(next_state)
                
                # Store experience
                memory.states.append(torch.FloatTensor(state).to(agent.device))
                memory.actions.append(action)
                memory.logprobs.append(action_log_prob)
                memory.state_values.append(state_value)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                # Update state and score
                state = next_state
                score += reward
                
                if render:
                    env.render()
                
                # Update agent
                if time_steps % update_interval == 0:
                    agent.update(memory)
                    memory.clear()
                
                if done:
                    break
            
            # Store score
            scores.append(score)
            
            # Evaluate agent
            if i_episode % eval_interval == 0:
                eval_score = self.evaluate_agent(agent, env, env_name, num_episodes=5, agent_type="ppo")
                eval_scores.append(eval_score)
                print(f"Episode {i_episode}/{num_episodes} | Average Score: {np.mean(scores[-100:]):.2f} | Eval Score: {eval_score:.2f}")
            
            # Save agent
            if i_episode % save_interval == 0:
                agent.save(os.path.join(self.models_dir, f"ppo_{env_name}_{i_episode}.pth"))
        
        # Save final agent
        agent.save(os.path.join(self.models_dir, f"ppo_{env_name}_final.pth"))
        
        # Plot scores
        self._plot_scores(scores, eval_scores, eval_interval, "PPO", env_name)
        
        return agent, scores
    
    def evaluate_agent(self, agent, env, env_name, num_episodes=10, max_steps=500, agent_type="dqn"):
        """
        Evaluate an agent
        
        Args:
            agent: Agent to evaluate
            env: Environment to evaluate on
            env_name (str): Name of the environment action
            num_episodes (int): Number of episodes to evaluate for
            max_steps (int): Maximum steps per episode
            agent_type (str): Type of agent ('dqn' or 'ppo')
            
        Returns:
            float: Average score
        """
        scores = []
        
        for i_episode in range(num_episodes):
            state = env.reset()
            state = self.preprocess_state(state)
            score = 0
            
            for t in range(max_steps):
                # Select action
                if agent_type == "dqn":
                    action = agent.act(state, eval_mode=True)
                elif agent_type == "ppo":
                    action, _, _ = agent.select_action(state)
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")
                
                # Convert action index to environment action
                env_action = self._convert_action_index_to_env_action(action, env_name)
                
                # Take action
                next_state, reward, done, _ = env.step(env_action)
                next_state = self.preprocess_state(next_state)
                
                # Update state and score
                state = next_state
                score += reward
                
                if done:
                    break
            
            scores.append(score)
        
        return np.mean(scores)
    
    def _create_environment(self, env_name):
        """
        Create environment for training
        
        Args:
            env_name (str): Name of the environment action to train on
            
        Returns:
            gym.Env: Environment
        """
        # Create environment
        env = self.formulation.create_environment(difficulty='normal')
        
        return env
    
    def _get_state_dim(self, env):
        """
        Get state dimension
        
        Args:
            env: Environment
            
        Returns:
            int: State dimension
        """
        # Reset environment to get state
        state = env.reset()
        
        # Preprocess state
        state_array = self.preprocess_state(state)
        
        return len(state_array)
    
    def _get_action_dim(self, env, env_name):
        """
        Get action dimension
        
        Args:
            env: Environment
            env_name (str): Name of the environment action to train on
            
        Returns:
            int: Action dimension
        """
        if env_name == "isru_mode":
            return 4  # water, oxygen, both, off
        elif env_name == "maintenance_target":
            return 5  # power_system, life_support, isru, thermal_control, none
        else:
            raise ValueError(f"Unknown environment action: {env_name}")
    
    def _convert_action_index_to_env_action(self, action_index, env_name):
        """
        Convert action index to environment action
        
        Args:
            action_index (int): Action index
            env_name (str): Name of the environment action
            
        Returns:
            dict: Environment action
        """
        if env_name == "isru_mode":
            isru_modes = ['water', 'oxygen', 'both', 'off']
            return {
                'power_allocation': {
                    'life_support': 5.0,
                    'isru': 3.0,
                    'thermal_control': 2.0
                },
                'isru_mode': isru_modes[action_index],
                'maintenance_target': None
            }
        elif env_name == "maintenance_target":
            maintenance_targets = ['power_system', 'life_support', 'isru', 'thermal_control', None]
            return {
                'power_allocation': {
                    'life_support': 5.0,
                    'isru': 3.0,
                    'thermal_control': 2.0
                },
                'isru_mode': 'both',
                'maintenance_target': maintenance_targets[action_index]
            }
        else:
            raise ValueError(f"Unknown environment action: {env_name}")
    
    def _plot_scores(self, scores, eval_scores, eval_interval, algorithm, env_name):
        """
        Plot training and evaluation scores
        
        Args:
            scores (list): Training scores
            eval_scores (list): Evaluation scores
            eval_interval (int): Evaluation interval
            algorithm (str): Algorithm name
            env_name (str): Environment name
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot training scores
        plt.plot(np.arange(len(scores)), scores, label='Training Score')
        
        # Plot evaluation scores
        eval_episodes = np.arange(eval_interval, len(scores) + 1, eval_interval)
        plt.plot(eval_episodes, eval_scores, 'r-', label='Evaluation Score')
        
        # Add moving average
        window_size = min(100, len(scores))
        moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(scores)), moving_avg, 'g-', label=f'{window_size}-Episode Moving Avg')
        
        # Add labels and title
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title(f'{algorithm} Training on {env_name}')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(os.path.join(self.models_dir, f"{algorithm.lower()}_{env_name}_scores.png"), dpi=300)
        plt.close()
    
    def train_all_models(self, num_episodes=1000, max_steps=500):
        """
        Train all baseline models
        
        Args:
            num_episodes (int): Number of episodes to train for
            max_steps (int): Maximum steps per episode
            
        Returns:
            dict: Dictionary of trained agents
        """
        agents = {}
        
        # Train DQN for ISRU mode
        agents["dqn_isru_mode"], _ = self.train_dqn(
            env_name="isru_mode",
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        
        # Train DQN for maintenance target
        agents["dqn_maintenance_target"], _ = self.train_dqn(
            env_name="maintenance_target",
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        
        # Train PPO for ISRU mode
        agents["ppo_isru_mode"], _ = self.train_ppo(
            env_name="isru_mode",
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        
        # Train PPO for maintenance target
        agents["ppo_maintenance_target"], _ = self.train_ppo(
            env_name="maintenance_target",
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        
        return agents
    
    def save_training_config(self):
        """
        Save training configuration
        
        Returns:
            str: Path to saved configuration
        """
        config = {
            "training_params": self.training_params,
            "device": str(self.device),
            "random_seed": RANDOM_SEED
        }
        
        config_path = os.path.join(self.models_dir, "training_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Training configuration saved to {config_path}")
        return config_path

# Example usage
if __name__ == "__main__":
    # Create baseline RL models
    baseline_rl = MarsHabitatBaselineRL("/home/ubuntu/martian_habitat_pathfinder/data")
    
    # Save training configuration
    baseline_rl.save_training_config()
    
    # Train DQN for ISRU mode (shorter training for demonstration)
    agent, scores = baseline_rl.train_dqn(
        env_name="isru_mode",
        num_episodes=100,
        max_steps=200
    )
