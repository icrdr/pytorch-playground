# %%
from agent import DQNAgent
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import wandb
from net import ConvNet, Net


HEIGHT = 50
WIDTH = 50
env = gym.make('CartPole-v1')
wandb.init(project="CartPole-v1")
# env.reset()
# agent = Agent(50, 50, env.action_space.n)
# plt.imshow(agent.get_screen(env))
# plt.show()


def get_screen_state(env):
    global last_screen
    screen = env.render(mode='rgb_array')
    screen = np.asarray(screen, dtype=np.float32) / 255
    screen = resize(screen, (HEIGHT, WIDTH))
    current_screen = screen.transpose(2, 0, 1)

    if(last_screen is None):
        last_screen = current_screen

    state = current_screen - last_screen
    last_screen = current_screen
    return state


num_epochs = 3000
# agent = DQNAgent(ConvNet, (WIDTH, HEIGHT, env.action_space.n), env.action_space.n)
agent = DQNAgent(Net, (4, env.action_space.n), env.action_space.n)
for i_epoch in range(num_epochs):
    state = env.reset()
    last_screen = None
    # state = get_state(env)
    loss_list = []
    total_reward = 0

    for i_step in range(1000):
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        agent.memory.add(state, action, next_state, reward, done)

        loss = agent.learn()
        if loss is not None:
            loss_list.append(loss)
        state = next_state
        total_reward += reward

        if done:
            loss_avg = math.nan if len(loss_list) == 0 else np.mean(np.asarray(loss_list))
            wandb.log({'reward': total_reward, 'loss': loss_avg, 'eps': agent.eps})
            print("Epoch {} reward:{} loss:{} eps:{}"
                  .format(i_epoch, total_reward, format(loss_avg, '.3f'), format(agent.eps, '.2f')))
            break
env.close()

# %%
