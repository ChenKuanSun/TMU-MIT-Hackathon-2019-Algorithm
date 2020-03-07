class Reward_Function(object):
    """[summary]
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self, index):
        self.index = index
        self.compute = getattr(self, "function_" + str(index))

    def function_0(self, 
            state_PFRatio, 
            state_next_PFRatio, 
            agent_tidal_ratio, 
            history_tidal_ratio, 
            outcome,
            death_90d,
            icu_stay_17d):
        """[summary]
        
        Arguments:
            state_PFRatio {[type]} -- [description]
            state_next_PFRatio {[type]} -- [description]
            agent_tidal_ratio {[type]} -- [description]
            history_tidal_ratio {[type]} -- [description]
            outcome {[type]} -- [description]
            death_90d {[type]} -- [description]
            icu_stay_17d {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        reward = 0

        # if(state_next_PFRatio > state_PFRatio):
        #     reward += 5
        print(agent_tidal_ratio[0], '\t', history_tidal_ratio)
        delta_action = abs(agent_tidal_ratio[0] - history_tidal_ratio) / 17 * 10

        if(death_90d):
            delta_action /= 2
            
        reward -= delta_action
        
        if(outcome):
            if(death_90d):
                reward -= 50
            else:
                reward += 50
                if(not icu_stay_17d):
                    reward += 50
        return reward
