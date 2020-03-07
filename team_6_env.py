import numpy as np
import json

class Env(object):
    """[summary]
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self,
        dataset,
        reward_function,
        test=False):
        self.test = test
        self.dataset = np.array(self.dataset_progress(dataset))
        self.reward_function = reward_function
        self.action_space = 1
        self.observation_space = 15
        self.index = 0

    def step(self, action):
        
        reward = 0
        state = self.state[self.time]
        state_next = self.state[self.time + 1]

        outcome = (self.time + 1 == len(self.state) - 1)
        reward = self.reward_function.compute(
            state_PFRatio = state[10],
            state_next_PFRatio = state_next[10],
            agent_tidal_ratio = action,
            history_tidal_ratio = self.history_tidal_ratio[self.time],
            outcome = outcome,
            death_90d = self.death_90d[self.time],
            icu_stay_17d = self.icu_stay_17d[self.time])
        # if(outcome):
        #     print("Index: " + str(self.index))
        self.time += 1


        return self.normalized(state_next), reward, outcome

    def reset(self):
        self.time = 0
        self.state = []
        self.state = np.concatenate([self.dataset[self.index][:,1:7], self.dataset[self.index][:,8:17]], axis=-1)
        self.history_tidal_ratio = self.dataset[self.index][:,7]
        self.icu_stay_17d =  self.dataset[self.index][:,17]
        self.death_90d = self.dataset[self.index][:,18]
        self.index = np.random.randint(0,len(self.dataset)-1)
        self.state[self.time]
        return self.normalized(self.state[self.time])

    def dataset_progress(self, dataset):
        dataset_pro = []
        state_temp = []
        for index, state in enumerate(dataset):
            if(index - 1 >= 0):
                if(dataset[index - 1][0] != dataset[index][0]):
                    dataset_pro.append(state_temp)
                    state_temp = []
            state_temp.append(state.tolist())
        dataset_pro.append(state_temp)
        return dataset_pro
    def normalized(self, v):
        # return v
        norm=np.linalg.norm(v, ord=1)
        if norm==0:
            norm=np.finfo(v.dtype).eps
        return v/norm

"""
[
dataset[1:6] + dataset[8:16]
0 'icustay_id', int

1'admission_age', int
2'gender', 0 1
3'weight', float
4'height', 
5'BMI',
6'PBW', 

7'tidalratio', 

8'peep', 
9'pco2', 
10'pao2', 
11'fio2', 
12'pao2fio2', 
13'wbc',
14'so2', 
15'plateaupressure', 
16'heartrate', 

17'icu_stay_17d', 
18 'death_90d'
]


0'admission_age', int
1'gender', 0 1
2'weight', float
3'height', 
4'BMI',
5'PBW', 
6'peep', 
7'pco2', 
8'pao2', 
9'fio2', 
10'pao2fio2', 
11'wbc',
12'so2', 
13'plateaupressure', 
14'heartrate', 

"""