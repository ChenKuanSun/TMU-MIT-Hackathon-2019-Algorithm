import numpy as np
import tensorflow as tf
import os
import team_6_env
import team_6_reward

MAX_EPISODE = 10000

LEARNING_RATE_ACTOR = 0.0001 
LEARNING_RATE_CRITIC = 0.001
N_STATE_FEATURES = 15
ACTION_BOUND = [1, 18]

GAMMA = 0.9

class Actor(object):
    # Policy Gradients
    def __init__(self, session, n_state_features, action_bound, learning_rate):
        self.session = session
        
        self.state = tf.placeholder(tf.float32, [1, n_state_features], "state")

        self.action = tf.placeholder(tf.float32, None, name="action")

        # TD_error
        # From Critic
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  

        layer_1 = tf.layers.dense(
                inputs=self.state,
                # number of hidden units
                units=45,  
                activation=tf.nn.relu,
                # weights
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1), 
                # biases
                bias_initializer=tf.constant_initializer(0.1),  
                name='layer_1'
            )

        mutual = tf.layers.dense(
                inputs=layer_1,
                # number of hidden units
                units=1,  
                activation=tf.nn.tanh,
                # weights
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1), 
                # biases 
                bias_initializer=tf.constant_initializer(0.1),  
                name='mutual'
            )

        sigma = tf.layers.dense(
                inputs=layer_1,
                # output units
                units=1,  
                # get action probabilities
                activation=tf.nn.softplus,  
                # weights
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),  
                # biases
                bias_initializer=tf.constant_initializer(1.0),  
                name='sigma'
            )

        global_step = tf.Variable(0, trainable=False)

        self.mutual, self.sigma = tf.squeeze(mutual * 2), tf.squeeze(sigma + 0.1)

        self.normal_dist = tf.distributions.Normal(self.mutual, self.sigma)

        # clip Action 4~8 ?
        self.action = tf.clip_by_value(
            self.normal_dist.sample(1),
            action_bound[0],
            action_bound[1])

        with tf.name_scope('expected_value'):
            # loss without advantage
            log_probability = self.normal_dist.log_prob(
                self.action)  
            # advantage (TD_error) guided loss
            self.expected_value = log_probability * self.td_error
            # Add cross entropy cost to encourage exploration
            self.expected_value += 0.01 * self.normal_dist.entropy()

        with tf.name_scope('train'):
            # min(value) = max(-value)
            self.train_op = tf.train.AdamOptimizer(
                learning_rate).minimize(-self.expected_value, global_step)  

    def learn(self, state, action, td_error):
        state = state[np.newaxis, :]
        feed_dict = {self.state: state, self.action: action, self.td_error: td_error}
        _, expected_value = self.session.run([self.train_op, self.expected_value], feed_dict)
        return expected_value

    def choose_action(self, state):
        state = state[np.newaxis, :]
        # get probabilities for all actions
        return self.session.run(self.action, {self.state: state})


class Critic(object):
    # Q learning
    def __init__(self, session, n_state_features, learning_rate):
        self.session = session
        
        with tf.name_scope('inputs'):
            self.state = tf.placeholder(tf.float32, [1, n_state_features], "state")
            self.value_next = tf.placeholder(tf.float32, [1, 1], name="value_next")
            self.reward = tf.placeholder(tf.float32, name='reward')

        with tf.variable_scope('Critic'):
            layer_1 = tf.layers.dense(
                inputs=self.state,
                # number of hidden units
                units=45,  
                activation=tf.nn.relu,
                # weights
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),  
                # biases
                bias_initializer=tf.constant_initializer(0.1),  
                name='layer_1'
            )

            self.value = tf.layers.dense(
                inputs=layer_1,
                # output units
                units=1,  
                activation=None,
                # weights
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),  
                # biases
                bias_initializer=tf.constant_initializer(0.1),  
                name='value'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.reward + GAMMA * self.value_next - self.value)
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def learn(self, state, reward, state_next):
        state, state_next = state, state_next

        value_next = self.session.run(self.value, {self.state: state_next[np.newaxis, :]})
        td_error, _ = self.session.run([self.td_error, self.train_op],
                                    {self.state: state[np.newaxis, :], self.value_next: value_next, self.reward: reward})
        return td_error
    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.session, './model_save2/params')

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.session, './model_save3/params', write_meta_graph=False)


session = tf.Session()
dataset = np.load('./dataset/ards_all_data_ver1.npy')
reward_function_0 = team_6_reward.Reward_Function(0)

env = team_6_env.Env(
    dataset = dataset,
    reward_function = reward_function_0
    )

actor = Actor(session=session, 
            n_state_features=N_STATE_FEATURES, 
            action_bound=ACTION_BOUND, 
            learning_rate=LEARNING_RATE_ACTOR)

critic = Critic(session=session, 
            n_state_features=N_STATE_FEATURES, 
            learning_rate=LEARNING_RATE_CRITIC)


session.run(tf.global_variables_initializer())

# critic.load()
tf.summary.FileWriter("logs/", session.graph)

for i_episode in range(MAX_EPISODE):
    state = env.reset()
    
    time = 0
    episode_reward_temp = []
    if( i_episode % 3000 == 0 ):
        LEARNING_RATE_ACTOR *=  0.01
        LEARNING_RATE_CRITIC *=  0.01
    action_s = []
    while True:
        action = actor.choose_action(state)

        state_next, reward, done = env.step(action)
        # normalize
        # reward /= 10

        td_error = critic.learn(state, reward, state_next)  
        actor.learn(state, action, td_error) 

        state = state_next
        time += 1
        action_s.append(action)
        episode_reward_temp.append(reward)
        if done:
            print('Reward: %i' % sum(episode_reward_temp), '\tAVG Action: %i' % (sum(action_s)/ len(action_s)), '\tMax Action: %i' % (max(action_s)), '\tmin Action: %i' % (min(action_s)))
            episode_reward_temp = []
            # print(sum(action_s)/ len(action_s))
            break
critic.save()