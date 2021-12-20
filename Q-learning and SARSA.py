import gym
import numpy as np
from matplotlib import pyplot as plt


class Arguments:
    def __init__(self):
        self.env = None
        self.obs_n = None
        self.act_n = None
        self.agent = None

        # Set your parameters here
        # 一些参数的设定
        self.episodes = 400
        self.max_step = 400
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon =0.1


class QLearningAgent:
    def __init__(self, args):
        self.obs_n = args.obs_n
        self.act_n = args.act_n  # 动作维度，表示有几个动作可选
        self.lr = args.lr  # 学习率
        self.gamma = args.gamma  # reward的衰减值
        self.epsilon = args.epsilon  # 按一定概率随机选动作
        self.Q = np.zeros((args.obs_n, args.act_n))  # 初始化Q表

    def select_action(self, obs, if_train=True):
        # Implement your code here
        # ...
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  #根据table的Q值选动作
            Q_list = self.Q[obs, :]  # 从Q表中选取状态(或观察值)对应的那一行
            maxQ = np.max(Q_list)  # 获取这一行最大的Q值
            action_list = np.where(Q_list == maxQ)[0]  # np.where找出最大值所在的位置
            action = np.random.choice(action_list)  # 选取最大值对应的动作
        else:
            action = np.random.choice(self.act_n)  #有一定概率随机探索选取一个动作
        return action

    def update(self, transition):
        obs, action, reward, next_obs, done = transition
        # Implement your code here
        # ...
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 如果到达终止状态， 没有下一个状态了，直接把奖励赋值给target_Q
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)


class SARSAAgent:
    def __init__(self, args):
        self.obs_n = args.obs_n
        self.act_n = args.act_n  # 动作的维度
        self.lr = args.lr  # 学习率
        self.gamma = args.gamma  # 折扣因子，reward的衰减率
        self.epsilon = args.epsilon  # 按一定概率随机选取动作
        self.Q = np.zeros((args.obs_n, args.act_n))  # 创建一个Q表格
    
    def select_action(self, obs, if_train=True):
        # Implement your code here
        # ...
        # 根据table的Q值选动作
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            Q_list = self.Q[obs, :]  # 从Q表中选取状态(或观察值)对应的那一行
            maxQ = np.max(Q_list)  # 获取这一行最大的Q值
            action_list = np.where(Q_list == maxQ)[0]  # np.where找出最大值所在的位置
            action = np.random.choice(action_list)  # 选取最大值对应的动作
        else:
            action = np.random.choice(self.act_n)  # 随机选取一个动作
        return action

    def update(self, transition):
        obs, action, reward, next_obs, next_action, done = transition
        # Implement your code here
        # ...
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 如果到达终止状态， 没有下一个状态了，直接把奖励赋值给target_Q
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]
        self.Q[obs, action] = predict_Q + self.lr * (target_Q - predict_Q)


def q_learning_train(args):
    env = args.env
    agent = args.agent
    episodes = args.episodes
    max_step = args.max_step
    rewards = []
    mean_100ep_reward = []
    for episode in range(episodes):
        episode_reward = 0  # 记录一个episode获得的总奖励
        # Implement your code here
        # ...
        obs = env.reset()  # 重置环境，重新开始新的一轮
        for t in range(max_step):
            # Implement your code here
            # ...
            action = agent.select_action(obs)  # 选取一个动作
            next_obs, reward, done, _ = env.step(action)  
            agent.update((obs, action, reward, next_obs, done))
            obs = next_obs
            episode_reward += reward
            if done: break
        print(f'Episode {episode}\t Step {t}\t Reward {episode_reward}')
        rewards.append(episode_reward)
        if len(rewards) < 100:
            mean_100ep_reward.append(np.mean(rewards))
        else:
            mean_100ep_reward.append(np.mean(rewards[-100:]))
    return mean_100ep_reward


def sarsa_train(args):
    env = args.env
    agent = args.agent
    episodes = args.episodes
    max_step = args.max_step
    rewards = []
    mean_100ep_reward = []
    for episode in range(episodes):
        episode_reward = 0  # 记录一个episode获得的总奖励
        # Implement your code here
        # ...
        obs = env.reset()  # 重置环境，重新开始新的一轮
        action = agent.select_action(obs)  # 根据算法选取一个动作
        for t in range(max_step):
            # Implement your code here
            # ...
            next_obs, reward, done, info = env.step(action)  # 将action作用于环境并得到反馈
            next_action = agent.select_action(next_obs)  # 根据下一状态，获取下一动作
            # 训练SARSA算法，更新Q表格
            agent.update((obs, action, reward, next_obs, next_action, done))
            action = next_action
            obs = next_obs  # 存储上一次观测值
            episode_reward += reward
            if done: break
        print(f'Episode {episode}\t Step {t}\t Reward {episode_reward}')
        rewards.append(episode_reward)
        if len(rewards) < 100:
            mean_100ep_reward.append(np.mean(rewards))
        else:
            mean_100ep_reward.append(np.mean(rewards[-100:]))
    return mean_100ep_reward


def q_learning_test(args):
    # Implement your code here
    # ...
    env = args.env
    agent = args.agent
    total_reward = 0
    obs = env.reset()
    while True:
        Q_list = agent.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)    # 获取下一个动作
        next_obs, reward, done, _ = env.step(action)  # 得到奖励reward和done
        total_reward += reward
        obs = next_obs
        env.render()  # 输出渲染
        if done:
            break


def sarsa_test(args):
    # Implement your code here
    # ...
    total_reward = 0
    env = args.env
    agent = args.agent
    obs = env.reset()
    while True:
        Q_list = agent.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        obs = next_obs
        env.render()
        if done:
            break
    return total_reward


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    q_learning_args = Arguments()
    env = gym.make("CliffWalking-v0")
    q_learning_args.env = env
    q_learning_args.obs_n = env.observation_space.n
    q_learning_args.act_n = env.action_space.n
    q_learning_args.agent = QLearningAgent(q_learning_args)
    

    sarsa_args = Arguments()
    env = gym.make("CliffWalking-v0")
    sarsa_args.env = env
    sarsa_args.obs_n = env.observation_space.n
    sarsa_args.act_n = env.action_space.n
    sarsa_args.agent = SARSAAgent(sarsa_args)

    q_learning_rewards = q_learning_train(q_learning_args)
    sarsa_rewards = sarsa_train(sarsa_args)

    q_learning_test(q_learning_args)
    sarsa_test(sarsa_args)

    plt.plot(range(q_learning_args.episodes), q_learning_rewards, label='Q Learning')
    plt.plot(range(q_learning_args.episodes), sarsa_rewards, label='SARSA')
    plt.legend()
    plt.show()
