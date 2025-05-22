# Advanced Reinforcement Learning Approaches for Mars Habitat Resource Management

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gym
import random
from collections import deque, namedtuple
import json
import sys
from tqdm import tqdm

# Add project directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.rl_environment import MarsHabitatRLEnvironment
from simulations.rl_formulation import MarsHabitatRLFormulation
from models.baseline_rl import MarsHabitatBaselineRL, ReplayBuffer

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class HierarchicalRLAgent:
    """
    Hierarchical Reinforcement Learning Agent for Mars Habitat Resource Management
    
    This agent implements a two-level hierarchy:
    - High-level policy: Selects goals or subtasks
    - Low-level policies: Execute specific subtasks
    """
    
    def __init__(self, state_dim, action_dims, device='cpu', lr=3e-4, gamma=0.99):
        """
        Initialize the Hierarchical RL agent
        
        Args:
            state_dim (int): Dimension of state space
            action_dims (dict): Dictionary of action dimensions for each subtask
            device (str): Device to run the model on ('cpu' or 'cuda')
            lr (float): Learning rate
            gamma (float): Discount factor
        """
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.device = device
        self.lr = lr
        self.gamma = gamma
        
        # Define subtasks
        self.subtasks = list(action_dims.keys())
        self.num_subtasks = len(self.subtasks)
        
        # High-level policy (meta-controller)
        self.meta_controller = MetaController(state_dim, self.num_subtasks, device, lr)
        
        # Low-level policies (controllers)
        self.controllers = {}
        for subtask, action_dim in action_dims.items():
            self.controllers[subtask] = SubtaskController(state_dim, action_dim, device, lr)
        
        # Current subtask
        self.current_subtask = None
        self.subtask_steps = 0
        self.max_subtask_steps = 10  # Maximum steps for a subtask before re-selection
        
        print(f"Hierarchical RL Agent initialized with {self.num_subtasks} subtasks: {self.subtasks}")
    
    def select_action(self, state):
        """
        Select action using the hierarchical policy
        
        Args:
            state: Current state
            
        Returns:
            tuple: (action, subtask)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Select or continue subtask
        if self.current_subtask is None or self.subtask_steps >= self.max_subtask_steps:
            # Select new subtask using meta-controller
            subtask_idx = self.meta_controller.select_action(state_tensor)
            self.current_subtask = self.subtasks[subtask_idx]
            self.subtask_steps = 0
        
        # Select action using current subtask controller
        action = self.controllers[self.current_subtask].select_action(state_tensor)
        
        # Increment subtask steps
        self.subtask_steps += 1
        
        return action, self.current_subtask
    
    def update(self, state, action, reward, next_state, done, subtask):
        """
        Update the hierarchical policy
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            subtask: Current subtask
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([float(done)]).to(self.device)
        
        # Update subtask controller
        self.controllers[subtask].update(state_tensor, action, reward_tensor, next_state_tensor, done_tensor)
        
        # Update meta-controller if subtask is completed or episode is done
        if self.subtask_steps >= self.max_subtask_steps or done:
            subtask_idx = self.subtasks.index(subtask)
            self.meta_controller.update(state_tensor, subtask_idx, reward_tensor, next_state_tensor, done_tensor)
    
    def save(self, directory):
        """
        Save the hierarchical policy
        
        Args:
            directory (str): Directory to save the policy
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save meta-controller
        self.meta_controller.save(os.path.join(directory, "meta_controller.pth"))
        
        # Save controllers
        for subtask, controller in self.controllers.items():
            controller.save(os.path.join(directory, f"controller_{subtask}.pth"))
    
    def load(self, directory):
        """
        Load the hierarchical policy
        
        Args:
            directory (str): Directory to load the policy from
        """
        # Load meta-controller
        self.meta_controller.load(os.path.join(directory, "meta_controller.pth"))
        
        # Load controllers
        for subtask, controller in self.controllers.items():
            controller.load(os.path.join(directory, f"controller_{subtask}.pth"))

class MetaController:
    """
    Meta-controller for selecting subtasks
    """
    
    def __init__(self, state_dim, num_subtasks, device='cpu', lr=3e-4):
        """
        Initialize the meta-controller
        
        Args:
            state_dim (int): Dimension of state space
            num_subtasks (int): Number of subtasks
            device (str): Device to run the model on ('cpu' or 'cuda')
            lr (float): Learning rate
        """
        self.state_dim = state_dim
        self.num_subtasks = num_subtasks
        self.device = device
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_subtasks)
        ).to(device)
        
        # Target Q-Network
        self.target_q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_subtasks)
        ).to(device)
        
        # Copy parameters to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001  # Soft update parameter
    
    def select_action(self, state):
        """
        Select subtask using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            int: Selected subtask index
        """
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.num_subtasks - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()
    
    def update(self, state, subtask, reward, next_state, done):
        """
        Update the meta-controller
        
        Args:
            state: Current state
            subtask: Selected subtask
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience in replay buffer
        self.replay_buffer.append((state, subtask, reward, next_state, done))
        
        # Check if enough samples are available
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample mini-batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, subtasks, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        subtasks = torch.tensor(subtasks, dtype=torch.long).to(self.device)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)
        
        # Compute Q-values
        q_values = self.q_network(states).gather(1, subtasks.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename):
        """
        Save the meta-controller
        
        Args:
            filename (str): Filename to save the meta-controller
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """
        Load the meta-controller
        
        Args:
            filename (str): Filename to load the meta-controller from
        """
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class SubtaskController:
    """
    Controller for executing subtasks
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', lr=3e-4):
        """
        Initialize the subtask controller
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            device (str): Device to run the model on ('cpu' or 'cuda')
            lr (float): Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Training parameters
        self.gamma = 0.99
    
    def select_action(self, state):
        """
        Select action using the actor network
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        with torch.no_grad():
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            return action.item()
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the subtask controller
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)
        
        # Compute advantage
        value = self.critic(state)
        next_value = self.critic(next_state)
        advantage = reward + (1 - done) * self.gamma * next_value - value
        
        # Update critic
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action_tensor)
        actor_loss = -log_prob * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def save(self, filename):
        """
        Save the subtask controller
        
        Args:
            filename (str): Filename to save the subtask controller
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        """
        Load the subtask controller
        
        Args:
            filename (str): Filename to load the subtask controller from
        """
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

class MultiAgentRLSystem:
    """
    Multi-Agent Reinforcement Learning System for Mars Habitat Resource Management
    
    This system implements multiple agents, each responsible for a different subsystem:
    - Power Agent: Manages power generation and distribution
    - Life Support Agent: Manages air, water, and food
    - ISRU Agent: Manages in-situ resource utilization
    - Maintenance Agent: Manages maintenance and repairs
    """
    
    def __init__(self, state_dim, action_dims, device='cpu', lr=3e-4, gamma=0.99):
        """
        Initialize the Multi-Agent RL system
        
        Args:
            state_dim (int): Dimension of state space
            action_dims (dict): Dictionary of action dimensions for each agent
            device (str): Device to run the model on ('cpu' or 'cuda')
            lr (float): Learning rate
            gamma (float): Discount factor
        """
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.device = device
        self.lr = lr
        self.gamma = gamma
        
        # Define agents
        self.agents = {}
        for agent_name, action_dim in action_dims.items():
            self.agents[agent_name] = Agent(agent_name, state_dim, action_dim, device, lr)
        
        # Communication module
        self.communication_module = CommunicationModule(list(action_dims.keys()), device)
        
        # Coordination module
        self.coordination_module = CoordinationModule(list(action_dims.keys()), device, lr)
        
        print(f"Multi-Agent RL System initialized with {len(self.agents)} agents: {list(self.agents.keys())}")
    
    def select_actions(self, state):
        """
        Select actions for all agents
        
        Args:
            state: Current state
            
        Returns:
            dict: Dictionary of actions for each agent
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get agent observations
        agent_observations = {}
        for agent_name, agent in self.agents.items():
            agent_observations[agent_name] = agent.get_observation(state_tensor)
        
        # Process communication
        messages = self.communication_module.process_communication(agent_observations)
        
        # Select actions
        actions = {}
        for agent_name, agent in self.agents.items():
            actions[agent_name] = agent.select_action(agent_observations[agent_name], messages)
        
        # Coordinate actions
        coordinated_actions = self.coordination_module.coordinate_actions(actions, state_tensor)
        
        return coordinated_actions
    
    def update(self, state, actions, reward, next_state, done):
        """
        Update all agents
        
        Args:
            state: Current state
            actions: Dictionary of actions for each agent
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([float(done)]).to(self.device)
        
        # Get agent observations
        agent_observations = {}
        next_agent_observations = {}
        for agent_name, agent in self.agents.items():
            agent_observations[agent_name] = agent.get_observation(state_tensor)
            next_agent_observations[agent_name] = agent.get_observation(next_state_tensor)
        
        # Process communication
        messages = self.communication_module.process_communication(agent_observations)
        next_messages = self.communication_module.process_communication(next_agent_observations)
        
        # Compute rewards for each agent
        agent_rewards = self.coordination_module.distribute_reward(reward_tensor, actions, state_tensor)
        
        # Update agents
        for agent_name, agent in self.agents.items():
            agent.update(
                agent_observations[agent_name],
                actions[agent_name],
                agent_rewards[agent_name],
                next_agent_observations[agent_name],
                done_tensor,
                messages,
                next_messages
            )
        
        # Update communication module
        self.communication_module.update(agent_observations, messages, agent_rewards, next_agent_observations, done_tensor)
        
        # Update coordination module
        self.coordination_module.update(state_tensor, actions, reward_tensor, next_state_tensor, done_tensor)
    
    def save(self, directory):
        """
        Save the multi-agent system
        
        Args:
            directory (str): Directory to save the system
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save agents
        for agent_name, agent in self.agents.items():
            agent.save(os.path.join(directory, f"agent_{agent_name}.pth"))
        
        # Save communication module
        self.communication_module.save(os.path.join(directory, "communication_module.pth"))
        
        # Save coordination module
        self.coordination_module.save(os.path.join(directory, "coordination_module.pth"))
    
    def load(self, directory):
        """
        Load the multi-agent system
        
        Args:
            directory (str): Directory to load the system from
        """
        # Load agents
        for agent_name, agent in self.agents.items():
            agent.load(os.path.join(directory, f"agent_{agent_name}.pth"))
        
        # Load communication module
        self.communication_module.load(os.path.join(directory, "communication_module.pth"))
        
        # Load coordination module
        self.coordination_module.load(os.path.join(directory, "coordination_module.pth"))

class Agent:
    """
    Individual agent in the multi-agent system
    """
    
    def __init__(self, name, state_dim, action_dim, device='cpu', lr=3e-4):
        """
        Initialize the agent
        
        Args:
            name (str): Agent name
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            device (str): Device to run the model on ('cpu' or 'cuda')
            lr (float): Learning rate
        """
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim + 64, 128),  # +64 for message embedding
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim + 64, 128),  # +64 for message embedding
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        
        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        ).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.message_optimizer = optim.Adam(self.message_encoder.parameters(), lr=lr)
        
        # Training parameters
        self.gamma = 0.99
    
    def get_observation(self, state):
        """
        Get agent-specific observation from state
        
        Args:
            state: Global state
            
        Returns:
            torch.Tensor: Agent-specific observation
        """
        # For simplicity, use the full state as observation
        # In a real implementation, this would extract relevant parts of the state
        return state
    
    def encode_message(self, observation):
        """
        Encode message from observation
        
        Args:
            observation: Agent-specific observation
            
        Returns:
            torch.Tensor: Encoded message
        """
        return self.message_encoder(observation)
    
    def select_action(self, observation, messages):
        """
        Select action using the actor network
        
        Args:
            observation: Agent-specific observation
            messages: Dictionary of messages from other agents
            
        Returns:
            int: Selected action
        """
        # Combine observation with messages
        message_vector = torch.cat(list(messages.values()), dim=0)
        message_vector = message_vector.mean(dim=0, keepdim=True)  # Average messages
        
        combined_input = torch.cat([observation, message_vector], dim=-1)
        
        # Select action
        with torch.no_grad():
            action_probs = self.actor(combined_input)
            dist = Categorical(action_probs)
            action = dist.sample()
            return action.item()
    
    def update(self, observation, action, reward, next_observation, done, messages, next_messages):
        """
        Update the agent
        
        Args:
            observation: Agent-specific observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
            messages: Dictionary of messages from other agents
            next_messages: Dictionary of next messages from other agents
        """
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)
        
        # Combine observation with messages
        message_vector = torch.cat(list(messages.values()), dim=0)
        message_vector = message_vector.mean(dim=0, keepdim=True)  # Average messages
        combined_input = torch.cat([observation, message_vector], dim=-1)
        
        next_message_vector = torch.cat(list(next_messages.values()), dim=0)
        next_message_vector = next_message_vector.mean(dim=0, keepdim=True)  # Average messages
        next_combined_input = torch.cat([next_observation, next_message_vector], dim=-1)
        
        # Compute advantage
        value = self.critic(combined_input)
        next_value = self.critic(next_combined_input)
        advantage = reward + (1 - done) * self.gamma * next_value - value
        
        # Update critic
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_probs = self.actor(combined_input)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action_tensor)
        actor_loss = -log_prob * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update message encoder
        message_loss = -advantage.detach().mean()  # Encourage messages that lead to higher advantage
        self.message_optimizer.zero_grad()
        message_loss.backward()
        self.message_optimizer.step()
    
    def save(self, filename):
        """
        Save the agent
        
        Args:
            filename (str): Filename to save the agent
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'message_encoder_state_dict': self.message_encoder.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'message_optimizer_state_dict': self.message_optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        """
        Load the agent
        
        Args:
            filename (str): Filename to load the agent from
        """
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.message_encoder.load_state_dict(checkpoint['message_encoder_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.message_optimizer.load_state_dict(checkpoint['message_optimizer_state_dict'])

class CommunicationModule:
    """
    Communication module for the multi-agent system
    """
    
    def __init__(self, agent_names, device='cpu'):
        """
        Initialize the communication module
        
        Args:
            agent_names (list): List of agent names
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.agent_names = agent_names
        self.device = device
        
        # Message attention network
        self.attention_network = nn.Sequential(
            nn.Linear(64, 64),  # 64 is the message dimension
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=0)
        ).to(device)
    
    def process_communication(self, agent_observations):
        """
        Process communication between agents
        
        Args:
            agent_observations (dict): Dictionary of agent observations
            
        Returns:
            dict: Dictionary of processed messages for each agent
        """
        # Encode messages
        messages = {}
        for agent_name, observation in agent_observations.items():
            # Create a dummy message encoder for demonstration
            # In a real implementation, this would use the agent's message encoder
            message_encoder = nn.Sequential(
                nn.Linear(observation.shape[-1], 64),
                nn.ReLU()
            ).to(self.device)
            messages[agent_name] = message_encoder(observation)
        
        # Process messages with attention
        processed_messages = {}
        for agent_name in self.agent_names:
            # Exclude agent's own message
            other_messages = {name: msg for name, msg in messages.items() if name != agent_name}
            
            if other_messages:
                # Apply attention
                message_stack = torch.stack(list(other_messages.values()), dim=0)
                attention_weights = self.attention_network(message_stack)
                weighted_messages = message_stack * attention_weights
                processed_messages[agent_name] = weighted_messages
            else:
                # No other messages, use zero vector
                processed_messages[agent_name] = torch.zeros(1, 64).to(self.device)
        
        return processed_messages
    
    def update(self, agent_observations, messages, rewards, next_agent_observations, done):
        """
        Update the communication module
        
        Args:
            agent_observations (dict): Dictionary of agent observations
            messages (dict): Dictionary of messages
            rewards (dict): Dictionary of rewards
            next_agent_observations (dict): Dictionary of next observations
            done (torch.Tensor): Whether episode is done
        """
        # In a real implementation, this would update the attention network
        # based on the effectiveness of communication
        pass
    
    def save(self, filename):
        """
        Save the communication module
        
        Args:
            filename (str): Filename to save the module
        """
        torch.save({
            'attention_network_state_dict': self.attention_network.state_dict()
        }, filename)
    
    def load(self, filename):
        """
        Load the communication module
        
        Args:
            filename (str): Filename to load the module from
        """
        checkpoint = torch.load(filename)
        self.attention_network.load_state_dict(checkpoint['attention_network_state_dict'])

class CoordinationModule:
    """
    Coordination module for the multi-agent system
    """
    
    def __init__(self, agent_names, device='cpu', lr=3e-4):
        """
        Initialize the coordination module
        
        Args:
            agent_names (list): List of agent names
            device (str): Device to run the model on ('cpu' or 'cuda')
            lr (float): Learning rate
        """
        self.agent_names = agent_names
        self.device = device
        
        # Coordination network
        self.coordination_network = nn.Sequential(
            nn.Linear(len(agent_names) * 10, 128),  # Assuming each agent has 10 possible actions
            nn.ReLU(),
            nn.Linear(128, len(agent_names) * 10),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Reward distribution network
        self.reward_network = nn.Sequential(
            nn.Linear(len(agent_names) + 1, len(agent_names)),  # +1 for global reward
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(list(self.coordination_network.parameters()) + 
                                   list(self.reward_network.parameters()), lr=lr)
    
    def coordinate_actions(self, actions, state):
        """
        Coordinate actions between agents
        
        Args:
            actions (dict): Dictionary of actions for each agent
            state: Global state
            
        Returns:
            dict: Dictionary of coordinated actions for each agent
        """
        # In a real implementation, this would adjust actions to ensure coordination
        # For simplicity, just return the original actions
        return actions
    
    def distribute_reward(self, global_reward, actions, state):
        """
        Distribute global reward among agents
        
        Args:
            global_reward (torch.Tensor): Global reward
            actions (dict): Dictionary of actions for each agent
            state: Global state
            
        Returns:
            dict: Dictionary of rewards for each agent
        """
        # Create action vector
        action_vector = torch.tensor([actions[name] for name in self.agent_names], dtype=torch.float32).to(self.device)
        
        # Combine with global reward
        combined_input = torch.cat([action_vector, global_reward], dim=0)
        
        # Compute reward distribution
        reward_distribution = self.reward_network(combined_input.unsqueeze(0)).squeeze(0)
        
        # Distribute rewards
        agent_rewards = {}
        for i, name in enumerate(self.agent_names):
            agent_rewards[name] = global_reward * reward_distribution[i]
        
        return agent_rewards
    
    def update(self, state, actions, reward, next_state, done):
        """
        Update the coordination module
        
        Args:
            state: Global state
            actions (dict): Dictionary of actions for each agent
            reward (torch.Tensor): Global reward
            next_state: Next global state
            done (torch.Tensor): Whether episode is done
        """
        # In a real implementation, this would update the coordination and reward networks
        # based on the effectiveness of coordination
        pass
    
    def save(self, filename):
        """
        Save the coordination module
        
        Args:
            filename (str): Filename to save the module
        """
        torch.save({
            'coordination_network_state_dict': self.coordination_network.state_dict(),
            'reward_network_state_dict': self.reward_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)
    
    def load(self, filename):
        """
        Load the coordination module
        
        Args:
            filename (str): Filename to load the module from
        """
        checkpoint = torch.load(filename)
        self.coordination_network.load_state_dict(checkpoint['coordination_network_state_dict'])
        self.reward_network.load_state_dict(checkpoint['reward_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class MarsHabitatAdvancedRL:
    """
    Advanced Reinforcement Learning approaches for Mars Habitat Resource Management
    """
    
    def __init__(self, data_dir):
        """
        Initialize the advanced RL approaches
        
        Args:
            data_dir (str): Directory containing data and configuration files
        """
        self.data_dir = data_dir
        self.models_dir = os.path.join(data_dir, "models", "advanced")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create RL formulation
        self.formulation = MarsHabitatRLFormulation(data_dir)
        
        # Create baseline RL for comparison
        self.baseline_rl = MarsHabitatBaselineRL(data_dir)
        
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Training parameters
        self.training_params = {
            "hrl": {
                "lr": 3e-4,
                "gamma": 0.99,
                "max_subtask_steps": 10
            },
            "marl": {
                "lr": 3e-4,
                "gamma": 0.99
            }
        }
        
        print(f"Mars Habitat Advanced RL initialized")
    
    def preprocess_state(self, state_dict):
        """
        Preprocess state dictionary into flat array
        
        Args:
            state_dict (dict): State dictionary from environment
            
        Returns:
            numpy.ndarray: Flattened state array
        """
        # Use the same preprocessing as baseline RL
        return self.baseline_rl.preprocess_state(state_dict)
    
    def train_hierarchical_rl(self, num_episodes=1000, max_steps=500, eval_interval=100, 
                             save_interval=100, render=False):
        """
        Train a Hierarchical RL agent
        
        Args:
            num_episodes (int): Number of episodes to train for
            max_steps (int): Maximum steps per episode
            eval_interval (int): Interval to evaluate the agent
            save_interval (int): Interval to save the agent
            render (bool): Whether to render the environment
            
        Returns:
            tuple: (agent, scores)
        """
        print(f"Training Hierarchical RL agent...")
        
        # Create environment
        env = self.formulation.create_environment(difficulty='normal')
        
        # Get state dimension
        state_dim = self._get_state_dim(env)
        
        # Define subtasks and their action dimensions
        action_dims = {
            "isru_mode": 4,  # water, oxygen, both, off
            "maintenance_target": 5  # power_system, life_support, isru, thermal_control, none
        }
        
        print(f"State dimension: {state_dim}, Action dimensions: {action_dims}")
        
        # Create agent
        agent = HierarchicalRLAgent(
            state_dim=state_dim,
            action_dims=action_dims,
            device=self.device,
            **self.training_params["hrl"]
        )
        
        # Training loop
        scores = []
        eval_scores = []
        
        for i_episode in tqdm(range(1, num_episodes+1), desc="Episodes"):
            state = env.reset()
            state_array = self.preprocess_state(state)
            score = 0
            
            for t in range(max_steps):
                # Select action
                action, subtask = agent.select_action(state_array)
                
                # Convert action to environment action
                env_action = self._convert_action_to_env_action(action, subtask)
                
                # Take action
                next_state, reward, done, _ = env.step(env_action)
                next_state_array = self.preprocess_state(next_state)
                
                # Update agent
                agent.update(state_array, action, reward, next_state_array, done, subtask)
                
                # Update state and score
                state = next_state
                state_array = next_state_array
                score += reward
                
                if render:
                    env.render()
                
                if done:
                    break
            
            # Store score
            scores.append(score)
            
            # Evaluate agent
            if i_episode % eval_interval == 0:
                eval_score = self.evaluate_hierarchical_rl(agent, env, num_episodes=5)
                eval_scores.append(eval_score)
                print(f"Episode {i_episode}/{num_episodes} | Average Score: {np.mean(scores[-100:]):.2f} | Eval Score: {eval_score:.2f}")
            
            # Save agent
            if i_episode % save_interval == 0:
                agent.save(os.path.join(self.models_dir, f"hrl_{i_episode}"))
        
        # Save final agent
        agent.save(os.path.join(self.models_dir, "hrl_final"))
        
        # Plot scores
        self._plot_scores(scores, eval_scores, eval_interval, "HRL")
        
        return agent, scores
    
    def train_multi_agent_rl(self, num_episodes=1000, max_steps=500, eval_interval=100, 
                            save_interval=100, render=False):
        """
        Train a Multi-Agent RL system
        
        Args:
            num_episodes (int): Number of episodes to train for
            max_steps (int): Maximum steps per episode
            eval_interval (int): Interval to evaluate the system
            save_interval (int): Interval to save the system
            render (bool): Whether to render the environment
            
        Returns:
            tuple: (system, scores)
        """
        print(f"Training Multi-Agent RL system...")
        
        # Create environment
        env = self.formulation.create_environment(difficulty='normal')
        
        # Get state dimension
        state_dim = self._get_state_dim(env)
        
        # Define agents and their action dimensions
        action_dims = {
            "power_agent": 4,  # Different power allocation strategies
            "life_support_agent": 3,  # Different life support modes
            "isru_agent": 4,  # water, oxygen, both, off
            "maintenance_agent": 5  # power_system, life_support, isru, thermal_control, none
        }
        
        print(f"State dimension: {state_dim}, Action dimensions: {action_dims}")
        
        # Create system
        system = MultiAgentRLSystem(
            state_dim=state_dim,
            action_dims=action_dims,
            device=self.device,
            **self.training_params["marl"]
        )
        
        # Training loop
        scores = []
        eval_scores = []
        
        for i_episode in tqdm(range(1, num_episodes+1), desc="Episodes"):
            state = env.reset()
            state_array = self.preprocess_state(state)
            score = 0
            
            for t in range(max_steps):
                # Select actions
                actions = system.select_actions(state_array)
                
                # Convert actions to environment action
                env_action = self._convert_marl_actions_to_env_action(actions)
                
                # Take action
                next_state, reward, done, _ = env.step(env_action)
                next_state_array = self.preprocess_state(next_state)
                
                # Update system
                system.update(state_array, actions, reward, next_state_array, done)
                
                # Update state and score
                state = next_state
                state_array = next_state_array
                score += reward
                
                if render:
                    env.render()
                
                if done:
                    break
            
            # Store score
            scores.append(score)
            
            # Evaluate system
            if i_episode % eval_interval == 0:
                eval_score = self.evaluate_multi_agent_rl(system, env, num_episodes=5)
                eval_scores.append(eval_score)
                print(f"Episode {i_episode}/{num_episodes} | Average Score: {np.mean(scores[-100:]):.2f} | Eval Score: {eval_score:.2f}")
            
            # Save system
            if i_episode % save_interval == 0:
                system.save(os.path.join(self.models_dir, f"marl_{i_episode}"))
        
        # Save final system
        system.save(os.path.join(self.models_dir, "marl_final"))
        
        # Plot scores
        self._plot_scores(scores, eval_scores, eval_interval, "MARL")
        
        return system, scores
    
    def evaluate_hierarchical_rl(self, agent, env, num_episodes=10, max_steps=500):
        """
        Evaluate a Hierarchical RL agent
        
        Args:
            agent: Agent to evaluate
            env: Environment to evaluate on
            num_episodes (int): Number of episodes to evaluate for
            max_steps (int): Maximum steps per episode
            
        Returns:
            float: Average score
        """
        scores = []
        
        for i_episode in range(num_episodes):
            state = env.reset()
            state_array = self.preprocess_state(state)
            score = 0
            
            for t in range(max_steps):
                # Select action
                action, subtask = agent.select_action(state_array)
                
                # Convert action to environment action
                env_action = self._convert_action_to_env_action(action, subtask)
                
                # Take action
                next_state, reward, done, _ = env.step(env_action)
                next_state_array = self.preprocess_state(next_state)
                
                # Update state and score
                state = next_state
                state_array = next_state_array
                score += reward
                
                if done:
                    break
            
            scores.append(score)
        
        return np.mean(scores)
    
    def evaluate_multi_agent_rl(self, system, env, num_episodes=10, max_steps=500):
        """
        Evaluate a Multi-Agent RL system
        
        Args:
            system: System to evaluate
            env: Environment to evaluate on
            num_episodes (int): Number of episodes to evaluate for
            max_steps (int): Maximum steps per episode
            
        Returns:
            float: Average score
        """
        scores = []
        
        for i_episode in range(num_episodes):
            state = env.reset()
            state_array = self.preprocess_state(state)
            score = 0
            
            for t in range(max_steps):
                # Select actions
                actions = system.select_actions(state_array)
                
                # Convert actions to environment action
                env_action = self._convert_marl_actions_to_env_action(actions)
                
                # Take action
                next_state, reward, done, _ = env.step(env_action)
                next_state_array = self.preprocess_state(next_state)
                
                # Update state and score
                state = next_state
                state_array = next_state_array
                score += reward
                
                if done:
                    break
            
            scores.append(score)
        
        return np.mean(scores)
    
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
    
    def _convert_action_to_env_action(self, action, subtask):
        """
        Convert action to environment action
        
        Args:
            action (int): Action index
            subtask (str): Subtask name
            
        Returns:
            dict: Environment action
        """
        if subtask == "isru_mode":
            isru_modes = ['water', 'oxygen', 'both', 'off']
            return {
                'power_allocation': {
                    'life_support': 5.0,
                    'isru': 3.0,
                    'thermal_control': 2.0
                },
                'isru_mode': isru_modes[action],
                'maintenance_target': None
            }
        elif subtask == "maintenance_target":
            maintenance_targets = ['power_system', 'life_support', 'isru', 'thermal_control', None]
            return {
                'power_allocation': {
                    'life_support': 5.0,
                    'isru': 3.0,
                    'thermal_control': 2.0
                },
                'isru_mode': 'both',
                'maintenance_target': maintenance_targets[action]
            }
        else:
            raise ValueError(f"Unknown subtask: {subtask}")
    
    def _convert_marl_actions_to_env_action(self, actions):
        """
        Convert multi-agent actions to environment action
        
        Args:
            actions (dict): Dictionary of actions for each agent
            
        Returns:
            dict: Environment action
        """
        # Power allocation strategies
        power_strategies = [
            {'life_support': 7.0, 'isru': 2.0, 'thermal_control': 1.0},  # Life support priority
            {'life_support': 4.0, 'isru': 5.0, 'thermal_control': 1.0},  # ISRU priority
            {'life_support': 4.0, 'isru': 2.0, 'thermal_control': 4.0},  # Thermal priority
            {'life_support': 4.0, 'isru': 3.0, 'thermal_control': 3.0}   # Balanced
        ]
        
        # Life support modes
        life_support_modes = [
            'minimal',  # Minimal life support
            'standard',  # Standard life support
            'comfort'    # Comfort-oriented life support
        ]
        
        # ISRU modes
        isru_modes = ['water', 'oxygen', 'both', 'off']
        
        # Maintenance targets
        maintenance_targets = ['power_system', 'life_support', 'isru', 'thermal_control', None]
        
        # Combine actions
        power_allocation = power_strategies[actions['power_agent']]
        isru_mode = isru_modes[actions['isru_agent']]
        maintenance_target = maintenance_targets[actions['maintenance_agent']]
        
        # Create environment action
        env_action = {
            'power_allocation': power_allocation,
            'isru_mode': isru_mode
        }
        
        # Add maintenance target if not None
        if maintenance_target:
            env_action['maintenance_target'] = maintenance_target
        
        return env_action
    
    def _plot_scores(self, scores, eval_scores, eval_interval, algorithm):
        """
        Plot training and evaluation scores
        
        Args:
            scores (list): Training scores
            eval_scores (list): Evaluation scores
            eval_interval (int): Evaluation interval
            algorithm (str): Algorithm name
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
        plt.title(f'{algorithm} Training')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(os.path.join(self.models_dir, f"{algorithm.lower()}_scores.png"), dpi=300)
        plt.close()
    
    def compare_approaches(self, num_episodes=10, max_steps=500):
        """
        Compare different RL approaches
        
        Args:
            num_episodes (int): Number of episodes to evaluate for
            max_steps (int): Maximum steps per episode
            
        Returns:
            dict: Dictionary of scores for each approach
        """
        print(f"Comparing RL approaches...")
        
        # Create environment
        env = self.formulation.create_environment(difficulty='normal')
        
        # Load models
        dqn_agent = self.baseline_rl.load_model("dqn_isru_mode_final.pth")
        ppo_agent = self.baseline_rl.load_model("ppo_isru_mode_final.pth")
        
        hrl_agent = HierarchicalRLAgent(
            state_dim=self._get_state_dim(env),
            action_dims={"isru_mode": 4, "maintenance_target": 5},
            device=self.device,
            **self.training_params["hrl"]
        )
        hrl_agent.load(os.path.join(self.models_dir, "hrl_final"))
        
        marl_system = MultiAgentRLSystem(
            state_dim=self._get_state_dim(env),
            action_dims={
                "power_agent": 4,
                "life_support_agent": 3,
                "isru_agent": 4,
                "maintenance_agent": 5
            },
            device=self.device,
            **self.training_params["marl"]
        )
        marl_system.load(os.path.join(self.models_dir, "marl_final"))
        
        # Evaluate models
        dqn_scores = self.baseline_rl.evaluate_agent(dqn_agent, env, "isru_mode", num_episodes, max_steps, "dqn")
        ppo_scores = self.baseline_rl.evaluate_agent(ppo_agent, env, "isru_mode", num_episodes, max_steps, "ppo")
        hrl_scores = self.evaluate_hierarchical_rl(hrl_agent, env, num_episodes, max_steps)
        marl_scores = self.evaluate_multi_agent_rl(marl_system, env, num_episodes, max_steps)
        
        # Compile results
        results = {
            "DQN": dqn_scores,
            "PPO": ppo_scores,
            "HRL": hrl_scores,
            "MARL": marl_scores
        }
        
        # Plot comparison
        self._plot_comparison(results)
        
        return results
    
    def _plot_comparison(self, results):
        """
        Plot comparison of different approaches
        
        Args:
            results (dict): Dictionary of scores for each approach
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot scores
        plt.bar(results.keys(), [score for score in results.values()])
        
        # Add labels and title
        plt.xlabel('Approach')
        plt.ylabel('Average Score')
        plt.title('Comparison of RL Approaches')
        plt.grid(True, axis='y')
        
        # Add value labels
        for i, (approach, score) in enumerate(results.items()):
            plt.text(i, score + 0.1, f'{score:.2f}', ha='center')
        
        # Save figure
        plt.savefig(os.path.join(self.models_dir, "comparison.png"), dpi=300)
        plt.close()
    
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
    # Create advanced RL approaches
    advanced_rl = MarsHabitatAdvancedRL("/home/ubuntu/martian_habitat_pathfinder/data")
    
    # Save training configuration
    advanced_rl.save_training_config()
    
    # Train Hierarchical RL agent (shorter training for demonstration)
    agent, scores = advanced_rl.train_hierarchical_rl(
        num_episodes=100,
        max_steps=200
    )
