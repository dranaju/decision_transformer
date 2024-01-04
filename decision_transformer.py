import gymnasium as gym 
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import numpy as np
from types import SimpleNamespace
import cv2
from torchvision.transforms import v2
import torch.optim as optim
# from transformers import AutoModel

transform_obs = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32)
    ])

def normalizer(data, normalizer):
    """ normalize data"""
    data_normalized = (data - normalizer.min)/(normalizer.max - normalizer.min)

    return data_normalized 

def img_to_tensor(frame):
    frame = transform_obs(frame)
    assert frame.shape == (3, 84, 84), f'frame shape is {frame.shape}'
    return frame

class ReplayBuffer:
    def __init__(
            self, 
            buffer_size= 1000, 
            batch_size= 64, 
            action_shape= (3,),
            obs_shape= (3, 84, 84),
            state_normalizer= SimpleNamespace(min=0, max=255), 
            action_normalizer= SimpleNamespace(min=-1, max=1), 
            device = 'cpu',
            observation_type = 'pixel'
            ):

        # self.memory             = deque(maxlen=buffer_size)
        self.buffer_size        = buffer_size
        self.batch_size         = batch_size
        self.device             = device
        self.state_normalizer   = state_normalizer
        self.action_normalizer  = action_normalizer
        self.experience         = namedtuple(
            'experience', field_names=['state', 'action', 'reward', 'next_state', 'done']
            )
        self.observation_type = observation_type

        self.states = np.empty((buffer_size, *obs_shape), dtype=np.float32)
        self.next_states = np.empty((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.empty((buffer_size, *action_shape), dtype=np.float32)
        self.rewards = np.empty((buffer_size, 1), dtype=np.float32)
        self.dones = np.empty((buffer_size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
    
    def add(self, state, action, reward, next_state, done):

        state       = normalizer(state, self.state_normalizer)
        action      = normalizer(action, self.action_normalizer)
        next_state  = normalizer(next_state, self.state_normalizer)

        if observation_type == 'pixel':
            state = transform_obs(state)
            next_state = transform_obs(next_state)

        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.buffer_size
        self.full = self.full or self.idx == 0

    def sample(self):

        indexs_sampled = np.random.randint(0, self.buffer_size if self.full else self.idx, size= self.batch_size)

        states = self.states[indexs_sampled]
        actions = self.actions[indexs_sampled]
        rewards = self.rewards[indexs_sampled]
        next_states = self.next_states[indexs_sampled]
        dones = self.dones[indexs_sampled]

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)


        return (states, actions, rewards, next_states, dones)
        

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion, device):
        super(TransformerBlock, self).__init__()
        self.device = device
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout).to(self.device)
        self.norm1 = nn.LayerNorm(embed_dim).to(self.device)
        self.norm2 = nn.LayerNorm(embed_dim).to(self.device)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion*embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_dim, embed_dim)
            ).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, attn_mask=mask)[0]
        x = self.dropout(self.norm1(attention + value))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out
    
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, num_heads, depth, forward_expansion, dropout, device):
        super(DecisionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        # self.state_embedding = nn.Sequential(
        #     nn.Conv2d(3, 32, 4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, 4, stride=2),
        #     nn.ReLU(),
        #     nn.Linear(256*2*2, embed_dim),
        #     nn.LayerNorm(embed_dim)
        # ).to(device)
        self.state_embedding_conv1 = nn.Conv2d(3, 32, 4, stride=2).to(self.device)
        self.state_embedding_conv2 = nn.Conv2d(32, 64, 4, stride=2).to(self.device)
        self.state_embedding_conv3 = nn.Conv2d(64, 128, 4, stride=2).to(self.device)
        self.state_embedding_conv4 = nn.Conv2d(128, 256, 4, stride=2).to(self.device)
        self.state_embedding_fc = nn.Linear(256 * 3 * 3, embed_dim).to(self.device)
        self.state_embedding_norm = nn.LayerNorm(embed_dim).to(self.device)


        self.action_embedding = nn.Linear(*action_dim, embed_dim).to(self.device)
        self.return_embedding = nn.Linear(1, embed_dim).to(self.device)
        self.positional_embedding = nn.Embedding(100, embed_dim).to(self.device)
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads, 
                dropout, 
                forward_expansion,
                self.device) for _ in range(depth)
        ])

        self.fc_out = nn.Linear(embed_dim, *action_dim).to(self.device)

    def forward(self, state, action, reward, desired_reward):
        # N, _ = state.shape
        # state_embedding = self.state_embedding(state)

        x = self.state_embedding_conv1(state)
        x = F.relu(x)
        x = self.state_embedding_conv2(x)
        x = F.relu(x)
        x = self.state_embedding_conv3(x)
        x = F.relu(x)
        x = self.state_embedding_conv4(x)
        # print(x.shape)
        x = F.relu(x)
        # x = x.view(x.size(0), -1)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # x = x.detach()
        # print(x.shape)

        x = self.state_embedding_fc(x.view(x.size(0), -1))
        state_embedding = self.state_embedding_norm(x)

        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        # print('acion', action.shape)
        action_embedding = self.action_embedding(action)
        sequence = state_embedding + action_embedding + self.positional_embedding
        # x = self.positional_embedding.repeat(N, 1, 1)
        for layer in self.layers:
            sequence = layer(sequence, sequence, sequence, mask=None)
        x = self.fc_out(sequence)
        return x
    
    def update(self, memory, optimizer):
        states, actions, rewards, next_states, dones = memory.sample()
        # print('states', states.shape)
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            optimizer.zero_grad()
            action_pred = self.forward(state, action, rewards, 0)
            # print('action_pred', action_pred.shape)
            # print('action', actions.shape)
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            loss = F.mse_loss(action_pred, action)
            loss.backward()
            optimizer.step()


# Hyperparameters ----------------------------------------------------------------------------------------------------
buffer_size = 10000
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('----')
print(f'Device used: {device}')
print('----')


env = gym.make("CarRacing-v2", continuous=True, render_mode = "human")
print("Observation space: ", env.observation_space)
observation_shape = (3, 84, 84)
observation_type = 'pixel'
# observation_type = 'raw_sensor'
print("Action space: ", env.action_space)
action_shape = env.action_space.shape

state_normalizer    = SimpleNamespace(min=0, max=255)
action_normalizer   = SimpleNamespace(min=-1, max=1)

replay_buffer = ReplayBuffer(
    buffer_size, 
    batch_size, 
    action_shape,
    observation_shape,
    state_normalizer, 
    action_normalizer, 
    device,
    observation_type
    )

observation = env.reset()[0]
observation = cv2.resize(observation[0:84, 0:96], (84, 84), interpolation = cv2.INTER_AREA)
# --------------------------------------------------------------------------------------------------------------------

# transformer load ---------------------------------------------------------------------------------------------------
embed_dim = 128
num_heads = 4
depth = 3
forward_expansion = 4
dropout = 0.1
decision_model = DecisionTransformer(observation_shape, action_shape, embed_dim, num_heads, depth, forward_expansion, dropout, device)

# Optimizer
decision_optimizer = optim.Adam(decision_model.parameters(), lr=0.0001)


# --------------------------------------------------------------------------------------------------------------------
for step in range(15):

    env.render()

    action = env.action_space.sample() 
    next_observation, reward, done, _, _ = env.step(action)
    next_observation = cv2.resize(
            next_observation[0:84, 0:96], (84, 84), interpolation = cv2.INTER_AREA
            )
    
    # print('action', action.shape)

    replay_buffer.add(observation, action, reward, next_observation, done)
    print('step', step)

    observation = next_observation

    if step > batch_size:
        decision_model.update(replay_buffer, decision_optimizer)

    if done:
        observation = env.reset()[0]
        observation = cv2.resize(
            observation[0:84, 0:96], (84, 84), interpolation = cv2.INTER_AREA
            )


# o, a, r, o_prime, done =replay_buffer.sample()
# print(o_prime.shape)
# print(a.shape)

# from PIL import Image
# import torchvision

# img = Image.open('/home/dranaju/Downloads/vae_dataset/0_0.png')
# img = np.array(img)
# print(img.shape)
# tr = torchvision.transforms.v2.ToImage()
# print(tr(img).shape)

env.close()