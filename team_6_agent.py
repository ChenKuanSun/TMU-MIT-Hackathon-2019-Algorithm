import tensorflow as tf
import numpy as np

MAX_EPISODES = 6000
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CITIC = 0.0005
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
BATCH_SIZE = 72

MEMORY_CAPACITY = 2880

class  DDPG(object):
    def __init__(self, action_dim, state_dim, action_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + action_dim + 1), dtype=np.float32)

        self.pointer = 0
        self.session = tf.Session()

        self.action_dim, self.state_dim, self.action_bound = action_dim, state_dim, action_bound

        self.state = tf.placeholder(tf.float32, [None, state_dim], 'state')
        self.state_next = tf.placeholder(tf.float32, [None, state_dim], 'state_next')

        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')

        with tf.variable_scope('Actor'):
            self.actor_eval = self._build_actor_net(self.state, scope='eval', trainable=True)
            actor_target = self._build_actor_net(self.state_next, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.actor_eval = a in memory when calculating q_eval_value for td_error,
            # otherwise the self.actor_eval is from Actor when updating Actor
            q_eval_value = self._build_critic_net(self.state, self.actor_eval, scope='eval', trainable=True)
            q_target_value = self._build_critic_net(self.state_next, actor_target, scope='target', trainable=False)

        # networks parameters
        self.actor_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.actor_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.critic_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.critic_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(target_params, (1 - TAU) * target_params + TAU * eval_params)
                             for target_params, eval_params in zip(self.actor_target_params + self.critic_target_params, self.actor_eval_params + self.critic_eval_params)]

        q_target = self.reward + GAMMA * q_target_value
        # in the feed_dic for the td_error, the self.actor_eval should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q_eval_value)
        self.critic_train = tf.train.AdamOptimizer(LEARNING_RATE_CITIC).minimize(td_error, var_list=self.critic_eval_params)
        
        actor_loss = -tf.reduce_mean(q_eval_value)    # maximize the q_eval_value
        self.actor_train = tf.train.AdamOptimizer(LEARNING_RATE_ACTOR).minimize(actor_loss, var_list=self.actor_eval_params)

        self.session.run(tf.global_variables_initializer())

    def choose_action(self, state):
        return self.session.run(self.actor_eval, {self.state: state[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.session.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_time = self.memory[indices, :]
        batch_state = batch_time[:, :self.state_dim]
        batach_action = batch_time[:, self.state_dim: self.state_dim + self.action_dim]
        batch_reward = batch_time[:, -self.state_dim - 1: -self.state_dim]
        batch_state_next = batch_time[:, -self.state_dim:]

        self.session.run(self.actor_train, {self.state: batch_state})
        self.session.run(self.critic_train, {self.state: batch_state, self.actor_eval: batach_action, self.reward: batch_reward, self.state_next: batch_state_next})

    def store_transition(self, state, action, reward, state_next):
        transition = np.hstack((state, action, [reward], state_next))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_actor_net(self, state, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(state, 20, activation=tf.nn.relu, name='layer_1', trainable=trainable)
            action = tf.layers.dense(net, self.action_dim, activation=tf.nn.tanh, name='action', trainable=trainable)
            return tf.multiply(action, self.action_bound, name='scaled_a')

    def _build_critic_net(self, state, action, scope, trainable):
        with tf.variable_scope(scope):
            n_layer_1 = 20
            weight_state_1 = tf.get_variable('weight_state_1', [self.state_dim, n_layer_1], trainable=trainable)
            weight_action_1 = tf.get_variable('weight_action_1', [self.action_dim, n_layer_1], trainable=trainable)
            bias_1 = tf.get_variable('bias_1', [1, n_layer_1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(state, weight_state_1) + tf.matmul(action, weight_action_1) + bias_1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(state,action)


    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.session, './model_save/params')

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.session, './model_save/params', write_meta_graph=False)

if __name__ == "__main__":
    import gym
    import time
    RENDER = False
    ENV_NAME = 'Pendulum-v0'
    MAX_EP_STEPS = 10000
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    print(s_dim,a_dim, a_bound)
    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -50:RENDER = True
                break
    print('Running time: ', time.time() - t1)