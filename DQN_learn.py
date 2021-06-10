# %%
from agent import DQNAgent
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import wandb
from net import ConvNet, Net

# LunarLander-v2
# CartPole-v1
HEIGHT = 100
WIDTH = 100
env = gym.make('LunarLander-v2')
wandb.init(project="LunarLander-v2")
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


# agent = DQNAgent(ConvNet,
#                  (WIDTH, HEIGHT, env.action_space.n),
#                  env.action_space.n,
#                  batch_size=64,
#                  eps_end=0.1,
#                  eps_decay=500)
agent = DQNAgent(Net,
                 (env.observation_space.shape[0], env.action_space.n),
                 env.action_space.n,
                 batch_size=64,
                 eps_end=0.1,
                 eps_decay=500)
num_epochs = 5000
max_time = 500
for i_epoch in range(num_epochs):
    state = env.reset()
    last_screen = None
    # state = get_screen_state(env)
    loss_list = []
    total_reward = 0

    for i_step in range(max_time):
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        # next_state = get_screen_state(env)
        agent.memory.add(state, action, next_state, reward, done)
        if(i_step >= max_time - 1):
            done = True
            reward = -100

        loss = agent.learn()
        if loss is not None:
            loss_list.append(loss)
        state = next_state
        total_reward += reward

        if done:
            loss_avg = math.nan if len(loss_list) == 0 else np.mean(np.asarray(loss_list))
            wandb.log({'reward': total_reward, 'loss': loss_avg, 'eps': agent.eps})
            print("Epoch {} reward:{} loss:{} eps:{}"
                  .format(i_epoch, format(total_reward, '.2f'), format(loss_avg, '.3f'), format(agent.eps, '.2f')))
            break
env.close()

# %%
