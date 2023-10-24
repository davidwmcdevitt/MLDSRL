# -*- coding: utf-8 -*-
"""pong-baseline-v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1twjVzobZi6ItYzUTGwrSIvLwJNIej0-s
"""

!nvidia-smi

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import os
from torch.utils.data import DataLoader, TensorDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.empty_cache()

def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(float).ravel(), [80,80])

env = gym.make("Pong-v0")

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 10 * 10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

        # Apply He initialization
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')

        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 10 * 10)  # Flatten the tensor
        x = F.relu(self.fc1(x))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.fc3(x)

        return F.softmax(x, dim=1)

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 10 * 10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

        # Apply He initialization
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')

        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 10 * 10)  # Flatten the tensor
        x = F.relu(self.fc1(x))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.fc3(x)

        return F.softmax(x, dim=1)

def get_action(policy_net, state, prior, method):
  if method == "random":
    action = np.random.choice(2) + 2
    probs = np.array([[0.5, 0.5]])
  if method == "gradient":
    screen = torch.tensor(preprocess(state)).unsqueeze(0).float()
    screen = screen.to(device)
    prior = prior.to(device)
    screen = screen - prior
    probs = policy_net(screen.unsqueeze(0))
    prior = screen

    try:
        probs = probs.cpu().detach().numpy()
        action = np.random.choice(2, p=probs) + 2
    except ValueError:
        action = np.random.choice(2) + 2

  return action, probs, prior

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = PolicyNetwork().to(device)

optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)

gamma = 0.99

policy_net

value_net = ValueNetwork().to(device)

value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)

value_net

longest_episode = 0
longest_episode_length = 0
longest_episode_path = ""
episode_lengths = []
loss_list = []
rolling_rewards = []
rolling_rewards_avg = []
rolling_time_avg = []
rolling_loss = []
value_list = []
val_avg = []
rolling_val = []

episode =  0
count = 0
cooldown = 0

for episode in range(10000):

    state, info = env.reset()
    t = 0

    obs_history = []
    action_history = []
    episode_rewards = []
    saved_log_probs = []
    state_hist = []
    action_hist = []

    terminated, truncated = False, False

    prior = torch.zeros((1, 80, 80))

    while not terminated:
        state_hist.append(torch.FloatTensor(preprocess(state)))
        t+=1'''
        if (episode) % 25 == 0:
            action, probs, prior = get_action(policy_net, state, prior, "random")
        else:
        '''
        action, probs, prior = get_action(policy_net, state, prior, "gradient")

        observation, reward, terminated, truncated, info = env.step(action)

        obs_history.append(observation)
        episode_rewards.append(reward)
        action_history.append(action)
        saved_log_probs.append(torch.tensor(np.log(probs)[0][action-2]))
        print(probs)
        print(probs[0])
        print(np.log(probs)[0][action-2])
        action_hist.append(action)


    rolling_rewards.append(sum(episode_rewards))
    rolling_rewards_avg.append(np.mean(rolling_rewards[-100:]))
    discounted_rewards = []
    R = 0
    for i, r in enumerate(reversed(episode_rewards)):
        #r = 10 * r + 1
        #if episode_rewards[i] != 0: R = 0
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.tensor(discounted_rewards)

    value_optimizer.zero_grad()
    criterion = nn.MSELoss()
    advantages = []
    '''
    for i, j in zip(state_hist,discounted_rewards):
        state_input = i.unsqueeze(0).unsqueeze(0)
        state_input = state_input.to(device)
        reward_input = j.to(device)
        value_pred = value_net(state_input)
        value_loss = criterion(value_pred, reward_input)
        value_loss.backward()
        value_optimizer.step()
        advantages.append(reward_input - value_pred)
    '''
    batch_size = 32

    state_hist_tensor = torch.stack(state_hist)
    discounted_rewards_tensor = torch.Tensor(discounted_rewards)

    dataset = TensorDataset(state_hist_tensor, discounted_rewards_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    value_losses = []

    for batch_idx, (batch_states, batch_rewards) in enumerate(dataloader):
        batch_states = batch_states.to(device)
        batch_states = batch_states.unsqueeze(1)
        batch_rewards = batch_rewards.to(device)

        value_pred = value_net(batch_states).squeeze(1)
        value_loss = criterion(value_pred, batch_rewards)

        value_losses.append(value_loss)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        adv = batch_rewards - value_pred.detach()
        adv = adv.to('cpu')

        advantages.append(adv)


    val_avg.append(sum(value_losses)/len(value_losses))

    policy_loss = -torch.mean(torch.tensor(saved_log_probs, requires_grad=True) * torch.cat(advantages))

    optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1)
    optimizer.step()

    episode_lengths.append(t)

    loss_list.append(policy_loss.cpu().detach().numpy())
    rolling_loss.append(np.mean(loss_list[-100:]))
    rolling_time_avg.append(np.mean(episode_lengths[-100:]))


    if (episode + 1) % 100 == 0:
        print("-"*50)
        print(f"Episode {episode+1} finished after {t} time steps.")
        print(f"Episode {episode+1} average time per 100: {np.mean(episode_lengths[-100:])} time steps.")
        print(f"Episode {episode + 1} rolling loss: {np.mean(loss_list[-100:])}")
        print(f"Episode {episode + 1} discounted rewards: {torch.mean(discounted_rewards)}")
        print(f"Episode {episode + 1} saved log probs: {np.mean(saved_log_probs)}")
        print(f"Episode {episode + 1} Action Means: {np.mean(action_history)}")
        print(f"Episode {episode + 1} Reward Rolling Average per 100: {np.mean(rolling_rewards[-100:])}")
        print("-"*50)

plt.figure(figsize=(10, 6))
plt.plot(episode_lengths, marker='o')

plt.title('Episode Lengths Over Time')
plt.xlabel('Episode')
plt.ylabel('Length')

plt.savefig('results/pong_duration.png', dpi=300, format='png', bbox_inches='tight')

plt.show()

plt.figure(figsize=(10, 6))
plt.plot(rolling_loss, marker='o')

plt.title('Average Loss Over Time')
plt.xlabel('Episode')
plt.ylabel('Loss (per 100)')
plt.savefig('results/pong_loss.png', dpi=300, format='png', bbox_inches='tight')

plt.show()

loss_list

plt.figure(figsize=(10, 6))
plt.plot(rolling_rewards_avg, marker='o')

plt.title('Avg Reward Over Time')
plt.xlabel('Episode')
plt.ylabel('Average Reward (per 100)')
plt.savefig('results/pong_avg_reward.png', dpi=300, format='png', bbox_inches='tight')

plt.show()

plt.figure(figsize=(10, 6))
plt.plot(rolling_time_avg, marker='o')

plt.title('Avg Lifespan Over Time')
plt.xlabel('Episode')
plt.ylabel('Average Lifespan (per 100)')
plt.savefig('results/pong_avg_life.png', dpi=300, format='png', bbox_inches='tight')

plt.show()

val_chart = torch.stack(val_avg).to('cpu').detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(val_chart, marker='o')

plt.title('Value Loss Over Time')
plt.xlabel('Episode')
plt.ylabel('Average Loss per 100')
plt.savefig('results/pong_rolling_val_loss.png', dpi=300, format='png', bbox_inches='tight')

plt.show()

rolling_val = np.convolve(val_chart, np.ones(100)/100, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(rolling_val, marker='o')

plt.title('Value Loss Over Time')
plt.xlabel('Episode')
plt.ylabel('Average Loss per 100')
plt.savefig('results/pong_rolling_val_loss.png', dpi=300, format='png', bbox_inches='tight')

plt.show()