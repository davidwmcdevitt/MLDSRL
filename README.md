# MLDSRL

```
git clone https://github.com/davidwmcdevitt/MLDSRL
```
_____________________________________________

HW3 Submission

_____________________________________________

1. CartPole Submission
   
```
python MLDSRL/HW3/cartpole_dqn.py
```
![CartPole-1](./HW3/results/cartpole_lengths.png)
![CartPole-2](./HW3/results/cartpole_loss.png)
![CartPole-3](./HW3/results/cartpole_max_q.png)

500 Episodes Rollout Test:
Mean: 199.37
SD: 5.82

2. Ms. Pacman Submission

```
python MLDSRL/HW2/mspacman_dqn.py
```
![Pong-1](./HW3/results/pong_avg_life.png)
![Pong-2](./HW3/results/pong_avg_reward.png)
![Pong-3](./HW3/results/pong_loss.png)
![Pong-4](./HW3/results/pong_rolling_val_loss.png)
_____________________________________________

HW2 Submission

_____________________________________________
   
1. CartPole Submission
   
```
python MLDSRL/HW2/cartpole_base.py
```
![CartPole-1](./HW2/results/cartpole_rolling_dur.png)
![CartPole-2](./HW2/results/cartpole_policy_loss.png)
![CartPole-3](./HW2/results/cartpole_value_loss.png)

CartPole trains until a streak of 10 consecutive episodes with a duration greater than or equal to 195 steps is acheived. Algorithm is tested on 100 episodes. if The average value of those 100 episodes is greater than or equal to 195, then the problem has been solved. 

2. Pong Submission

```
python MLDSRL/HW2/pong_baseline_v2.py
```
![Pong-1](./HW2/results/pong_avg_life.png)
![Pong-2](./HW2/results/pong_avg_reward.png)
![Pong-3](./HW2/results/pong_loss.png)
![Pong-4](./HW2/results/pong_rolling_val_loss.png)


_____________________________________________

HW1 Submission

_____________________________________________

1. CartPole Submission
   
```
python MLDSRL/HW1/cartpole.py
```

![CartPole-1](./HW1/results/cartpole_rolling_duration.png)
![CartPole-2](./HW1/results/cartpole_rolling_loss.png)

CartPole trains until a streak of 10 consecutive episodes with a duration greater than or equal to 195 steps is acheived. Algorithm is tested on 100 episodes. if The average value of those 100 episodes is greater than or equal to 195, then the problem has been solved. 

2. Pong Submission

```
python MLDSRL/HW1/pong.py
```
![Pong-1](./HW1/results/pong_avg_life.png)
![Pong-2](./HW1/results/pong_avg_reward.png)
![Pong-3](./HW1/results/pong_loss.png)
