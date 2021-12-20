import gym
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import gamma


class CliffWalking:
    def __init__(self):
        self.actions = (0, 1, 2, 3) # 可以采取的动作
        # 地形
        self.rewards = [[-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, -1],
                        [-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, -1],
                        [-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, -1],
                        [-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  0]]

    # 这个函数是前后左右行走
    def step(self, i, j, a):
        i_ = 0
        j_ = 0
        if a == 0:
            i_ = i-1 if i > 0 else 0
            j_ = j
        elif a == 1:
            i_ = i
            j_ = j+1 if j < 11 else j
        elif a == 2:
            i_ = i+1 if i < 3 else i
            j_ = j
        elif a == 3:
            i_ = i
            j_ = j-1 if j > 0 else j
        return i_, j_, self.rewards[i_][j_]




# 策略迭代
class PolicyIteration:
    def __init__(self, env):
        self.env = env
        # 初始化，将策略进行随机化
        self.PI = np.array([[np.random.choice((0,1,2,3)) for j in range(12)] for i in range(4)])
        # 初始化，值函数也随机化
        self.V = np.array([[np.random.random() for j in range(12)] for i in range(4)])
        # 需要定义最后到达终点的值函数为0
        self.V[-1][-1] = 0

    # 策略评估函数
    def policy_valuation(self):
        # 一些参数的设置
        threshold = 1e-10
        gamma = 0.9
        while True:
            new_value_table = np.copy(self.V)   # 赋值生成一个新的值列表
            # 循环，遍历所有的状态
            for i in range(4):
                for j in range(12):
                    action = self.PI[i][j]  # 返回当前策略当前状态下对应的动作
                    next_i, next_j, reward = env.step(i, j, action)  # 返回当前状态下执行动作得到的下一状态和奖励
                    self.V[i][j] = reward + gamma * new_value_table[next_i][next_j]  # 计算策略下的状态价值
            # 如果两个更新之间的差值小于阈值，则退出循环
            if np.sum((np.fabs(new_value_table - self.V))) <= threshold:
                break
        return self.V
    
    # 策略提升函数
    def policy_improvement(self):
        # 参数的设定
        gamma = 0.9
        # 循环，遍历所有的状态
        for i in range(4):
            for j in range(12):
                # 创建列表存储当前状态下执行不同动作的价值
                action_table = np.zeros(4)
                # 循环，遍历所有的动作
                for action in range(4):
                    next_i, next_j, reward = env.step(i, j, action) # 返回当前状态执行动作得到的下一状态及奖励
                    action_table[action] = reward + gamma * self.V[next_i][next_j]  # 计算当前状态下执行该动作获得的奖励
                # 策略提升，选取获取奖励最大的动作更新策略
                self.PI[i][j] = np.argmax(action_table)
        return self.PI
    
    
    def learn(self):
        # Implement your code here
        # ...
        # 循环，交错调用策略评估和策略提升函数
        while True:
            last_policy = np.copy(self.PI)
            self.policy_valuation()
            self.policy_improvement()
            # 如果前后两次的策略没有更新，表示已经收敛，所以退出循环
            if(np.all(last_policy == self.PI)):
                print('策略迭代结束，策略迭代最优策略如下')
                break            
        print(self.PI)
    

# 值迭代
class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.PI = np.array([[np.random.choice((0,1,2,3)) for j in range(12)] for i in range(4)])
        self.V = np.array([[np.random.random() for j in range(12)] for i in range(4)])
        self.V[-1][-1] = 0

    def learn(self):
        # Implement your code here
        # ...
        # 一些基本参数的设置
        threshold = 1e-20
        gamma = 0.9
        # 不断循环
        while True:
            # 创建每次迭代更新的状态价值表
            new_value_table = np.copy(self.V)
            # 循环，遍历所有状态
            for i in range(4):
                for j in range(12):
                    # 创建空的动作价值列表
                    action_value = np.zeros(4)
                    # 循环，遍历所有动作
                    for action in range(4):
                        # 返回当前状态-动作下一步的状态和奖励
                        next_x, next_y, reward = env.step(i, j, action)
                        # 计算动作的累积期望奖励
                        action_value[action] = reward + gamma * new_value_table[next_x][next_y]
                    self.V[i][j] = max(action_value)  # 更新状态值表
                    self.PI[i][j] = np.argmax(action_value)  # 记录当前最佳策略
            # 价值表前后两次更新之差小于阈值时停止循环
            if np.sum((np.fabs(new_value_table - self.V))) <= threshold:
                print('值迭代结束, 值迭代最优策略如下')
                break
        print(self.PI)



if __name__ == '__main__':
    np.random.seed(0)
    env = CliffWalking()
    
    PI = PolicyIteration(env)
    PI.learn()

    go = {0:'↑', 1:'→', 2:'↓', 3:'←'}
    start_pos_x = 3
    start_pos_y = 0
    list = []
    while True:
        st = PI.PI[start_pos_x][start_pos_y]
        list.append(go[st])
        start_pos_x, start_pos_y, re = env.step(start_pos_x, start_pos_y, st)
        if(start_pos_x ==3 and start_pos_y==11):
            break
    print('策略迭代路线图如下')
    print(list)
    
    VI = ValueIteration(env)
    VI.learn()
    
    go = {0:'↑', 1:'→', 2:'↓', 3:'←'}
    start_pos_x = 3
    start_pos_y = 0
    list = []
    while True:
        st = VI.PI[start_pos_x][start_pos_y]
        list.append(go[st])
        start_pos_x, start_pos_y, re = env.step(start_pos_x, start_pos_y, st)
        if(start_pos_x ==3 and start_pos_y==11):
            break
    print('值迭代路线图如下')
    print(list)
