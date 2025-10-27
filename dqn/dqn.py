import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import torch 
import torch.nn as nn
import torch.optim as optim
import sys


class dqn(nn.Module):

    # DQN class
    def __init__(self, input_dims, n_actions) -> None:
        
        super(dqn, self).__init__()
        self.input_dims = input_dims
        self.nn_conv1 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 8, stride = 4)
        self.nn_conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 4, stride = 2)
        self.nn_conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1)

        self.linear1 = nn.Linear(in_features = 7 * 7 * 32  , out_features = 512) 
        self.linear2 = nn.Linear(in_features = 512  , out_features = 256)
        self.linear3 = nn.Linear(in_features = 256  , out_features = 64)
        self.linear4 = nn.Linear(in_features = 64  , out_features = 32)
        self.linear5 = nn.Linear(in_features = 32  , out_features = n_actions)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x, **kwargs):

        x = torch.as_tensor(x)
        x = self.nn_conv1(x)
        x = nn.functional.dropout(x, p = 0.1)
        x = self.relu(x)
        x = self.nn_conv2(x)
        x = nn.functional.dropout(x, p = 0.1)
        x = self.relu(x)
        x = self.nn_conv3(x)
        x = nn.functional.dropout(x, p = 0.1)
        x = self.relu(x)

        # Flatten first
        x = torch.flatten(x, start_dim=1)
        
        x = self.linear1(x)
        x = nn.functional.dropout(x, p = 0.1)
        x = self.relu(x)
        x = self.linear2(x)
        x = nn.functional.dropout(x, p = 0.1)
        x = self.relu(x)
        x = self.linear3(x)
        x = nn.functional.dropout(x, p = 0.1)
        x = self.relu(x)
        x = self.linear4(x)
        x = nn.functional.dropout(x, p = 0.1)
        x = self.relu(x)
        

        logits = self.linear5(x) 

        return logits
    
# Memory Class for the DQN
class Memory:
    
    def __init__(self,input_dims,max_mem_size = 20000):

        """ Init a memory class and memory variables """
        
        super(Memory).__init__()

        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.new_state_mem = np.zeros((self.mem_size,*input_dims),dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype= np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype= np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype = bool)


    def store_transition(self, state, action, reward, state_neu, done):
        """ Storing trnasitions in a Matrix and overrighting some"""
        
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_mem[index] = state_neu
        self.reward_memory [index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr +=1
    
class Agents(nn.Module):
    
    def __init__(self, gamma, lr, batch_size, n_actions, input_dims, epsilon, eps_end=0.1, eps_dec = 0.9999, max_mem_size = 20000):
        """ Defining an agent class"""

        super(Agents, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gamma= gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space  = [i for i in range(n_actions)]
        self.batch_size = batch_size

        self.memory = Memory(input_dims,max_mem_size)

        self.Q_eval = dqn(input_dims, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.Q_eval.parameters(),lr=self.lr,eps=1e-7)
        self.loss_fn = nn.MSELoss()
    
    def choose_action(self, observation):
        
        if np.random.random() > self.epsilon:
            state = np.array([observation],dtype=np.float32)
            state = torch.tensor(state).to(self.device)
            actions = self.Q_eval(state)
            return torch.argmax(actions).item()
        else:
            return np.random.choice(self.action_space)

    def learn(self, target_res , terminated):
        
        if self.memory.mem_cntr < self.batch_size:
            return

        max_mem = min(self.memory.mem_cntr, self.memory.mem_size)
        batch = np.random.choice(max_mem,self.batch_size, replace= False)
        batch_index = np.arange(self.batch_size,dtype = np.int32)

        state_batch =  torch.tensor(self.memory.state_memory[batch]).to(self.device)
        new_state_batch = torch.tensor(self.memory.new_state_mem[batch]).to(self.device)
        reward_batch = torch.tensor(self.memory.reward_memory[batch]).to(self.device)
        terminal_batch = torch.tensor(self.memory.terminal_memory[batch]).to(self.device)

        action_batch = self.memory.action_memory[batch]

        q_eval = self.Q_eval(state_batch)[batch_index,action_batch]
        q_next = target_res(new_state_batch)
        

        q_target = reward_batch + (1 - terminated) * self.gamma * torch.max(q_next, dim=1)[0]

        self.optimizer.zero_grad()
        
        loss = self.loss_fn(q_target, q_eval).to(self.device)

        loss.backward()
        self.optimizer.step()

    


        