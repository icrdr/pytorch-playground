import minerl
import gym
import logging
logging.basicConfig(level=logging.DEBUG)
env = gym.make('MineRLNavigateDense-v0')
print('v')

obs = env.reset()
done = False
net_reward = 0

while not done:
    action = env.action_space.noop()
    print(action)
    action['camera'] = [0, 0.03 * obs["compassAngle"]]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(action)
    env.render()
    net_reward += reward
    print("Total reward: ", net_reward)
