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
import sys
import math
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
        self.time_steps = np.empty((buffer_size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.full_batch = False
    
    def add(self, state, action, reward, next_state, done, timestep):

        if self.observation_type == 'pixel':
            state       = normalizer(state, self.state_normalizer)
            action      = normalizer(action, self.action_normalizer)
            next_state  = normalizer(next_state, self.state_normalizer)
            state = transform_obs(state)
            next_state = transform_obs(next_state)

        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.dones[self.idx], done)
        np.copyto(self.time_steps[self.idx], timestep)

        self.idx = (self.idx + 1) % self.buffer_size
        self.full = self.full or self.idx == 0
        self.full_batch = self.full_batch or self.idx == self.batch_size

    def sample(self, option='random'):

        if option == 'random':
            indexs_sampled = np.random.choice(self.buffer_size if self.full else self.idx, 
                                            size= self.batch_size if self.full_batch else self.idx,
                                            replace=False)
        elif option == 'sequential':
            random_idx = np.random.randint(0, self.buffer_size if self.full else self.idx)
            if random_idx < self.batch_size:
                random_idx = self.batch_size
            indexs_sampled = np.arange(random_idx - self.batch_size, random_idx)
        
        # print('indexs_sampled', indexs_sampled)


        states = self.states[indexs_sampled]
        actions = self.actions[indexs_sampled]
        rewards = self.rewards[indexs_sampled]
        next_states = self.next_states[indexs_sampled]
        dones = self.dones[indexs_sampled]
        time_steps = self.time_steps[indexs_sampled].reshape(-1)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)
        time_steps = torch.from_numpy(time_steps).long().to(self.device)


        return (states, actions, rewards, next_states, dones, time_steps)
        

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
    
class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, device= 'cuda'):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T
        self.device = device

        self.q_net = nn.Linear(h_dim, h_dim).to(self.device)
        self.k_net = nn.Linear(h_dim, h_dim).to(self.device)
        self.v_net = nn.Linear(h_dim, h_dim).to(self.device)

        self.proj_net = nn.Linear(h_dim, h_dim).to(self.device)

        self.att_drop = nn.Dropout(drop_p).to(self.device)
        self.proj_drop = nn.Dropout(drop_p).to(self.device)

        ones = torch.ones((max_T, max_T)).to(self.device)
        mask = torch.tril(ones).view(1, 1, max_T, max_T).to(self.device)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, device= 'cuda'):
        super().__init__()
        self.device = device
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p, device=self.device)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            ).to(self.device)
        self.ln1 = nn.LayerNorm(h_dim).to(self.device)
        self.ln2 = nn.LayerNorm(h_dim).to(self.device)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x
    
class DecisionTransformer(nn.Module):
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            embed_dim, 
            num_heads, 
            depth, 
            forward_expansion, 
            dropout, device, 
            max_step=4096, 
            observation_type='pixel',
            batch_size=1,
            context_length=1
            ):
        super(DecisionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.observation_type = observation_type
        self.batch_size = batch_size
        self.context_length = context_length
        self.action_dim = action_dim

        if self.observation_type == 'pixel':
            self.state_embedding_conv1 = nn.Conv2d(3, 32, 4, stride=2).to(self.device)
            self.state_embedding_conv2 = nn.Conv2d(32, 64, 4, stride=2).to(self.device)
            self.state_embedding_conv3 = nn.Conv2d(64, 128, 4, stride=2).to(self.device)
            self.state_embedding_conv4 = nn.Conv2d(128, 256, 4, stride=2).to(self.device)
            self.state_embedding_fc = nn.Linear(256 * 3 * 3, embed_dim).to(self.device)
            self.state_embedding_norm = nn.LayerNorm(embed_dim).to(self.device)
        else:
            self.state_embedding = nn.Linear(*state_dim, embed_dim).to(self.device)

        # embedding
        self.action_embedding = nn.Linear(*action_dim, embed_dim).to(self.device)
        self.return_embedding = nn.Linear(1, embed_dim).to(self.device)
        self.time_embedding = nn.Embedding(max_step, embed_dim).to(self.device)
        self.layer_norm_embedding = nn.LayerNorm(embed_dim).to(self.device)   
        # self.layers = nn.ModuleList([
        #     TransformerBlock(
        #         embed_dim,
        #         num_heads, 
        #         dropout, 
        #         forward_expansion,
        #         self.device) for _ in range(depth)
        # ])
        blocks = [Block(embed_dim, 3*self.context_length, num_heads, dropout, device=self.device) for _ in range (depth)]
        self.transformer = nn.Sequential(*blocks)
    

        # prediction
        use_action_tanh = True 
        print('use_action_tanh', action_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(embed_dim, *action_dim)] + [nn.Tanh()] if use_action_tanh else [])
            ).to(self.device)
        self.predict_reward = nn.Linear(embed_dim, 1).to(self.device)
        self.predict_state = nn.Linear(embed_dim, *state_dim).to(self.device)
        # self.fc_out = nn.Linear(embed_dim, *action_dim).to(self.device)

        # prediction


    def forward(self, state, action, reward, timestep):
        B, T, _ = state.shape

        # print('time_step', timestep.shape)
        # # print('time_step', timestep)
        # print('state', state.shape)
        # # print('state', state)
        # print('action', action.shape)
        # print('reward', reward.shape)

        

        if self.observation_type == 'pixel':
            x = self.state_embedding_conv1(state)
            x = F.relu(x)
            x = self.state_embedding_conv2(x)
            x = F.relu(x)
            x = self.state_embedding_conv3(x)
            x = F.relu(x)
            x = self.state_embedding_conv4(x)
            x = F.relu(x)
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            x = self.state_embedding_fc(x.view(x.size(0), -1))
            state_embedding = self.state_embedding_norm(x) + self.time_embedding(timestep)
        else:
            state_embedding = self.state_embedding(state) + self.time_embedding(timestep)
        

        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        action_embedding = self.action_embedding(action) + self.time_embedding(timestep)

        return_embedding = self.return_embedding(reward) + self.time_embedding(timestep)

        

        # sequence = state_embedding + action_embedding + return_embedding
        # sequence = self.layer_norm_embedding(sequence)

        # for layer in self.layers:
        #     sequence = layer(sequence, sequence, sequence, mask=None)

        h = torch.stack(
            (return_embedding, state_embedding, action_embedding), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.embed_dim)

        h = self.layer_norm_embedding(h)

        # transformer and prediction
        h = self.transformer(h)

        h = h.reshape(B, T, 3, self.embed_dim).permute(0, 2, 1, 3)

        return_preds = self.predict_reward(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1])  # predict action given r, s

        

        return state_preds, action_preds, return_preds
    
    def update(self, memory, optimizer, model):
        # a = [memory.sample() for _ in range(10)]
        # a = torch.cat(a)
        # 

        model.train()
        
        samples = [memory.sample(option='sequential') for _ in range(self.batch_size)]
        states, actions, rewards, next_states,dones, timesteps = zip(*samples)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)
        timesteps = torch.stack(timesteps)

        # states, actions, rewards, next_states, dones, timesteps = [memory.sample() for _ in range(10)]

        # print('states', states.shape)
        # print('time_steps', timesteps.shape)
        # for state, action, reward, next_state, done, timestep in zip(states, 
        #                                                    actions, 
        #                                                    rewards, 
        #                                                    next_states, 
        #                                                    dones, 
        #                                                    timesteps):

        _, action_preds, _ = self.forward(states, actions, rewards, timesteps)

        action_preds = action_preds.view(-1, *self.action_dim)
        actions = actions.view(-1, *self.action_dim)

        # traj_mask = torch.randint(0, 2, size=(64, 20), dtype=torch.float).to(self.device)0]

        action_loss = F.mse_loss(action_preds, actions, reduction='mean')

        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # print('action_loss', action_loss)

        # sys.exit('stop')


# Hyperparameters ----------------------------------------------------------------------------------------------------
buffer_size = 10000
batch_size_memory = 20
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('----')
print(f'Device used: {device}')
print('----')


# env = gym.make("CarRacing-v2", continuous=True, render_mode = "human")
env = gym.make("Pendulum-v1", render_mode = "human")
# env = gym.make("Pendulum-v1")
# observation_type = 'pixel'
observation_type = 'raw_sensor'

print("Observation space: ", env.observation_space)

if observation_type == 'pixel':
    observation_shape = (3, 84, 84)
else:
    observation_shape = env.observation_space.shape

print("Action space: ", env.action_space)
action_shape = env.action_space.shape
# print('action_shape', action_shape)

state_normalizer    = SimpleNamespace(min=0, max=255)
action_normalizer   = SimpleNamespace(min=-1, max=1)

replay_buffer = ReplayBuffer(
    buffer_size, 
    batch_size_memory, 
    action_shape,
    observation_shape,
    state_normalizer, 
    action_normalizer, 
    device,
    observation_type
    )

if observation_type == 'pixel':
    observation = env.reset()[0]
    observation = cv2.resize(observation[0:84, 0:96], (84, 84), interpolation = cv2.INTER_AREA)
else:
    observation = env.reset()[0]
# --------------------------------------------------------------------------------------------------------------------

# transformer load ---------------------------------------------------------------------------------------------------
embed_dim = 128
num_heads = 4
depth = 3
forward_expansion = 4
dropout = 0.1
decision_model = DecisionTransformer(observation_shape, 
                                     action_shape, 
                                     embed_dim, 
                                     num_heads, 
                                     depth, 
                                     forward_expansion, 
                                     dropout, 
                                     device, 
                                     observation_type=observation_type,
                                     batch_size=batch_size,
                                     context_length=batch_size_memory)

# Optimizer
decision_optimizer = optim.Adam(decision_model.parameters(), 
                                lr=1e-4)


# --------------------------------------------------------------------------------------------------------------------
step = 0
max_step = 500
update_model_flag = False

for _ in range(100*max_step):

    env.render()

    decision_model.eval()
    with torch.no_grad():
        if step == 0:
            observations = [torch.from_numpy(observation).float().to(device) for _ in range(batch_size_memory)]
            observations = torch.stack(observations).unsqueeze(0)
            actions = [torch.from_numpy(np.array([0])).float().to(device) for _ in range(batch_size_memory)]
            actions = torch.stack(actions).unsqueeze(0)
            rewards = [torch.from_numpy(np.array([0])).float().to(device) for _ in range(batch_size_memory)]
            rewards = torch.stack(rewards).unsqueeze(0)
            timesteps = [torch.from_numpy(np.array([i])).long().to(device) for i in range(batch_size_memory)]
            timesteps = torch.cat(timesteps).unsqueeze(0)
            _, action_preds, _ = decision_model(observations, actions, rewards, torch.from_numpy(np.array([0])).long().to(device))
            action_t = action_preds[0, step].detach()
        else:
            timesteps = torch.cat([timesteps[0, 1:], torch.from_numpy(np.array([step+batch_size_memory-1])).long().to(device)]).unsqueeze(0)
            observations = torch.cat([observations[0, 1:], torch.from_numpy(observation).float().to(device).unsqueeze(0)]).unsqueeze(0)
            actions = torch.cat([actions[0, 1:], action_t.to(device).unsqueeze(0)]).unsqueeze(0)
            rewards = torch.cat([rewards[0, 1:], torch.from_numpy(np.array([reward])).float().to(device).unsqueeze(0)]).unsqueeze(0)
            _, action_preds, _ = decision_model(observations, actions, rewards, timesteps)
            action_t = action_preds[0, batch_size_memory-1].detach()

        action = action_t.cpu().numpy()

    

    # action = env.action_space.sample()
    # print('action', action)
    next_observation, reward, done, _, _ = env.step(action*2)
    # print('reward', reward)
    # print('reward_type', type(reward))
    # print('reward_shape', reward.shape)
    # sys,exit('stop')

    if observation_type == 'pixel':
        next_observation = cv2.resize(
                next_observation[0:84, 0:96], (84, 84), interpolation = cv2.INTER_AREA
                )
        
    if step == 0:
        [replay_buffer.add(state=observation,
                        action=action , 
                        reward=reward, 
                        next_state=next_observation, 
                        done=done, 
                        timestep=i) for i in range(batch_size_memory)]
    else:
        replay_buffer.add(state=observation,
                        action=action , 
                        reward=reward, 
                        next_state=next_observation, 
                        done=done, 
                        timestep=step+batch_size_memory-1)
    
    print('step', step)
    step += 1

    observation = next_observation

    if replay_buffer.idx > batch_size*batch_size_memory or update_model_flag:
        update_model_flag = True
        print('update')
        decision_model.update(replay_buffer, decision_optimizer, decision_model)

    if done or step == max_step:
        step = 0
        observation = env.reset()[0]
        if observation_type == 'pixel':
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