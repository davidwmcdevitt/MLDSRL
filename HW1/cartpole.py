# -*- coding: utf-8 -*-

!nvidia-smi

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        self.fc = nn.Sequential()

        # First fully connected layer
        self.fc1 = nn.Linear(4, 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc.add_module("fc1", self.fc1)
        self.fc.add_module("relu1", nn.ReLU())

        # Second fully connected layer
        self.fc2 = nn.Linear(128, 2)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc.add_module("fc2", self.fc2)

    def forward(self, x):
        logits = self.fc(x)
        logits -= torch.max(logits, dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
        return F.softmax(logits, dim=-1)

def get_action(policy_net, state, method):
  if method == "random":
    action = env.action_space.sample()
  if method == "gradient":
    try:
        probs = policy_net(torch.tensor(state).unsqueeze(0).float().to(device))
    except:
        probs = policy_net(torch.tensor(state[0]).unsqueeze(0).float().to(device) )
    try:
      action = np.random.choice(2, p=probs.cpu().detach().numpy()[0])
    except:
      policy_net = PolicyNetwork()
      action = 0

  return action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = PolicyNetwork().to(device)
#policy_net = PolicyNetwork()
optimizer = optim.RMSprop(policy_net.parameters(), lr=0.001)

gamma = 0.95


policy_net

env = gym.make("CartPole-v0")

max = 195

longest_episode = 0
longest_episode_length = 0
longest_episode_path = ""
episode_lengths = []
loss_list = []
loss_avg = []
length_avg = []

episode =  0
count = 0
cooldown = 0

print("Training...")
while True:

    state = env.reset()
    episode_rewards = []
    saved_log_probs = []

    episode += 1

    cooldown -=1

    for t in range(1, 1000):

        action = get_action(policy_net, state, method = "gradient")
        state, reward, done, truncated, info = env.step(action)

        saved_log_probs.append(torch.log(policy_net(torch.from_numpy(state).float().unsqueeze(0).to(device))[0][action]))
        episode_rewards.append(reward)

        if done or truncated:
            #print(f"Episode {episode + 1} finished after {t} time steps.")
            episode_lengths.append(t)
            length_avg.append(np.mean(episode_lengths[-100:]))

            if t > longest_episode_length:
                longest_episode_length = t
                longest_episode = episode

            break

    discounted_rewards = []
    R = 0
    for r in reversed(episode_rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    loss = -torch.mean(torch.mul(torch.stack(saved_log_probs), discounted_rewards.to(device)))
    loss_list.append(loss.cpu().detach().numpy())
    loss_avg.append(np.mean(loss_list[-100:]))
    #print(f"Episode {episode + 1} loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1)
    optimizer.step()

    if t >= max and cooldown <= 0:

      count +=1

      if count >= 10:

        cooldown = 10

        print(f"Testing on episode {episode +1}")

        test_lengths = []

        for test_count in range(100):

          state = env.reset()

          for t in range(1, 1000):
            action = get_action(policy_net, state, method = "gradient")
            state, reward, done, truncated, info = env.step(action)

            if done or truncated:
                #print(f"Episode {episode + 1} finished after {t} time steps.")
                test_lengths.append(t)
                break

        test_average = sum(test_lengths) / 100
        print(f"Test on episode {episode +1}: Average per 100 = {test_average}")

        if test_average >= 195:

          state = env.reset()
          frames = []
          for t in range(1, 1000):
            action = get_action(policy_net, state, method = "gradient")
            state, reward, done, truncated, info = env.step(action)

            if done or truncated:
              break

          break
    else:
        count = 0


env.close()

plt.figure(figsize=(10, 6))
plt.plot(length_avg, marker='o')

plt.title('Episode Lengths Over Time')
plt.xlabel('Episode')
plt.ylabel('Average Length per 100')

plt.show()

plt.savefig('results/cartpole_rolling_duration.png', dpi=300, format='png', bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(loss_avg, marker='o')

plt.title('Loss Over Time')
plt.xlabel('Episode')
plt.ylabel('Average Loss per 10')

plt.show()

plt.savefig('results/cartpole_rolling_loss.png', dpi=300, format='png', bbox_inches='tight')
