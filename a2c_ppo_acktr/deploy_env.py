import gym
from gym import error, spaces, utils
from gym.spaces import Box,MultiDiscrete
from gym.utils import seeding
import pandas as pd
import numpy as np
import torch
import math

NODE_OVER_UTILIZED_THRESHOLD = 0.8
NODE_UNDER_UTILIZED_THRESHOLD = 0.2
NODE_OVER_UTILIZED_PENALTY = 1
NODE_UNDER_UTILIZED_PENALTY = 1
POD_UNDER_REQUESTED_PENALTY = 1
MIGRATION_COST_PENALTY = 0.2

df = pd.read_csv('/data/rise_monitor_usage_data.csv')


def getNextData(step,pod_num):
    '''
        - getNextData返回下一个5min内的请求
        - 下一个5min[cur_step*300,(cur_step+1)*300)
        - 忽略所有释放资源的请求
    '''
    request = pd.DataFrame(columns=df.columns)
    counter = 0
    step = int(step)
    for t in range(step, step+3000, 300):
        selected_df = df[(df['start_time']>=t) & (df['start_time']<(t+300))]
        selected_df = selected_df.assign(start_time=t)
        added = selected_df.groupby('pod_id').mean().reset_index()
        request = request.append(added)
        for i in range(pod_num):
            if i not in added['pod_id'].values:
                request_i = df[(df['start_time']<t) & (df['pod_id']==i)]
                request_add = request_i.sort_values('start_time').iloc[len(request_i)-1]
                request_add['start_time'] = t
                request.loc[len(request)] = request_add
    
    assert len(request) == pod_num*10, f'{request}'
    return request

def hasNextData(step):
    request = df[(df['start_time']>=step) & (df['start_time']<=(step+3000))]
    return len(request)!=0

class Pod():
    def __init__(self, index, node_id):
        self.index = index
        self.current_node = node_id

class Node():
    def __init__(self, node_index):
        self.index = node_index
        self.pods = set()
    

class Cluster():
    def __init__(self,nodes_num,pods_num,init_data):
        self.nodes_num = nodes_num
        self.pods_num = pods_num
        self.nodes = [Node(i) for i in range(nodes_num)]
        self.pods = [Pod(i,-1) for i in range(pods_num)]
        def initPod(row):
            pod_id = int(row['pod_id'])
            node_id = int(row['node_id'])
            self.pods[pod_id].current_node = node_id
            self.nodes[node_id].pods.add(pod_id)
        init_data.apply(initPod,axis=1)
    
    def reset(self, nodes_num,pods_num,init_data):
        self.__init__(nodes_num,pods_num,init_data)

    def handle_migration(self, action):
        if action.shape==(self.pods_num,1):
            action = action.squeeze(-1)
        cost = 0
        for pod_index, pod_action in enumerate(action):
            assert pod_action>=0 and pod_action<self.nodes_num,f'{pod_action}' 
            if self.pods[pod_index].current_node==pod_action:
                continue

            from_node = self.pods[pod_index].current_node
            self.nodes[from_node].pods.remove(pod_index)
            self.nodes[pod_action].pods.add(pod_index)
            self.pods[pod_index].current_node = pod_action
            cost+=1
        return cost

    def describe(self,step):
        ret=[]
        step = int(step)
        for node in self.nodes:
            for t in range(step,step+3000,300):
                node_data = []
                for i in range(self.pods_num):
                    node_data.append([0.0,0.0])
                for pod_index in node.pods:
                    pod_data = self.data[(self.data['start_time']==t) & (self.data['pod_id']==pod_index)]
                    assert len(pod_data)==1, f'{pod_data},{self.data},{t}'
                    node_data[pod_index] = [float(pod_data['used_cpu'].values),float(pod_data['used_mem'].values)]
                ret.append(node_data)
        ret_np = np.array(ret)
        assert ret_np.shape==(10*self.nodes_num,self.pods_num,2)
        return ret_np
        
    def setData(self,data):
        self.data = data

class DeployEnvMonitorData(gym.Env):
    def __init__(self):
        super(DeployEnvMonitorData, self).__init__()
        self.init_data = pd.read_csv('/data/rise_monitor_init_data.csv')
        data = pd.read_csv('/data/rise_monitor_usage_data.csv')
        nodes_num = len(self.init_data['node_id'].unique())
        pods_num = len(self.init_data['pod_id'].unique())
        self.t = self.init_data['start_time'][0]
        self.start = self.t
        self.nodes_num = nodes_num
        self.pods_num = pods_num
        self.cluster = Cluster(nodes_num,pods_num,self.init_data)
        self.observation_space = Box(0,1,[nodes_num*10,pods_num,2])
        self.action_space = MultiDiscrete([nodes_num for i in range(pods_num)])

    '''
        以step作为请求序列的开始请求，
        cluster reset到初始化的阶段，
        t reset到step后第一个请求分配的请求
    '''
    def reset(self):
        self.t = self.init_data['start_time'][0]
        self.cluster.reset(self.nodes_num,self.pods_num,self.init_data)
        self.handle_next_request()
        state = self.cluster.describe(self.t)
        return state

    def termination(self):
        if not hasNextData(self.t):
            return True
        # TODO: 没有考虑当前资源已经分配不了的情况
        # TODO: 如果内存不够会存在无法handle下一个请求的情况
        return False

    '''
        执行迁移action
        action是一个元素为[from,target,to]的列表
    '''
    def _step(self,actions):
        if len(actions)==0:
            return 0
        return self.cluster.handle_migration(actions)

    '''
        handle下一个五分钟内的请求
        - getNextData获取下一个五分钟内的请求
        - scheduleRequest做请求的调度，返回pod固定时刻的request data序列
    '''
    def handle_next_request(self):
        request = getNextData(self.t, self.pods_num)
        self.cluster.setData(request)

    '''
        1. 执行迁移action
        2. 计算迁移后的即时reward
        3. getNextData handle下一个五分钟内的请求，更改cluster状态
    '''
    def step(self, actions):
        # print(migration_cost)
        self.t += 3000
        self.handle_next_request()
        migration_cost = self._step(actions)
        reward = self.reward(migration_cost)
        state = self.cluster.describe(self.t)
        done = self.termination()
        return state, reward, done, {'episode':{'r':reward}}

    def reward(self,migration_cost):
        state = self.cluster.describe(self.t)
        # 1. 资源考量：当前时间段内node上remain_cpu/total_cpu（或mem）<20%的次数或>80%的次数
        over = 0
        under = 0
        idle_num = 0
        over_num = 0
        for node in state:
            # assigned_cpu/total_cpu
            cpu_util = node[:,0].sum()
            mem_util = node[:,1].sum()
            if cpu_util==0 and mem_util==0:
                idle_num += 1
            if (not cpu_util==0 and cpu_util < NODE_UNDER_UTILIZED_THRESHOLD) or \
                (cpu_util > NODE_OVER_UTILIZED_THRESHOLD) or \
                (not mem_util==0 and mem_util < NODE_UNDER_UTILIZED_THRESHOLD) or \
                (mem_util > NODE_OVER_UTILIZED_THRESHOLD):
                over_num += 1
        resource_penalty = 1-2/(math.exp(idle_num/len(state))+1)+1/(over_num/len(state)+1)
        # print(f'resource_penalty: {resource_penalty}\n')

        # 2. 性能考量

        # 3. 互补性考量

        # 4. 迁移成本考量：被迁移的pod的request_cpu和request_mem累加
        migration_penalty = 1/(pow(1.1,migration_cost)+1)
        if reward>4:
            import pdb;pdb.set_trace()
        # print(f'migration_penalty: {migration_penalty}\n')
        return resource_penalty+migration_penalty
        
    def get_attr(self, attr_name):
        # request是还没有handle的下一阶段的请求
        if attr_name == 'obs':
            return self.cluster.describe(self.t)
        elif attr_name == 'req_step':
            return self.t
        return None