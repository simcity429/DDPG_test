import gym
import tensorflow as tf
import numpy as np
from random import shuffle
from sklearn.preprocessing import StandardScaler

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def assign_target(sess, main_network_param, target_network_param):
    assign_op = []
    for main_var, target_var in zip(main_network_param, target_network_param):
        assign_op.append(target_var.assign(main_var.value()))
    return sess.run(assign_op)

class Actor:
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.input, out = self.create_actor('actor')
        self.target_input, self.target_out = self.create_actor('target_actor')
        self.main_network_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.target_network_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

        self.action_gradient = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='action_gradient')
        tmp = tf.gradients(self.out, self.main_network_param, -self.action_gradient)
        self.actor_gradient = list(map(lambda x: tf.div(x, self.batch_size), tmp))
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.action_gradient, self.main_network_param))


    def create_actor(self, name):
        with tf.variable_scope(name):
            state_input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='state_input')
            out = tf.layers.dense(inputs=state_input, units=100, activation='elu')
            out = tf.layers.batch_normalization(out)
            out = tf.layers.dense(inputs=out, units=100, activation='elu')
            out = tf.layers.dense(inputs=out, units=self.action_dim, activation='tanh', use_bias=False,
                                  kernel_initializer=tf.initializers.random_uniform(minval=-0.003, maxval=0.003))
        return state_input, out

    def update_target(self):
        copy_op = []
        tau = self.tau
        for main_var, target_var in zip(self.main_network_param, self.target_network_param):
            copy_op.append(target_var.assign(tf.multiply(main_var.value(), tau) + tf.multiply(target_var.value(), 1 - tau)))
        return self.sess.run(copy_op)

    def train(self, input, action_gradient):
        return self.sess.run(self.optimize, feed_dict={self.input: input, self.action_gradient: action_gradient})

    def predict(self, input):
        return self.sess.run(self.out, feed_dict={self.input: input})

    def predict_target(self, input):
        return self.sess.run(self.target_out, feed_dict={self.target_input: input})

class Critic:
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.action_input, self.state_input, self.out = self.create_critic('critic')
        self.target_action_input, self.target_state_input, self.target_out = self.create_critic('target_critic')

        self.main_network_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        self.target_network_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')

        self.predicted_q = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='predicted_q')

        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.predicted_q, self.out)))
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action_input)



    def create_critic(self, name):
        with tf.variable_scope(name):
            action_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='action_input')
            state_input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='state_input')
            process_state = tf.layers.dense(inputs=state_input, units=100, activation='elu')
            process_state = tf.layers.batch_normalization(process_state)
            process_state = tf.layers.dense(inputs=process_state, units=100, activation='elu')
            process_action = tf.layers.dense(inputs=action_input, units=100, activation='elu')
            process_action = tf.layers.batch_normalization(process_action)
            out = tf.layers.dense(inputs=tf.add(process_state, process_action), units=50, activation='elu')
            out = tf.layers.dense(inputs=out, units=1, kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
        return action_input, state_input, out

    def update_target(self):
        copy_op = []
        tau = self.tau
        for main_var, target_var in zip(self.main_network_param, self.target_network_param):
            copy_op.append(target_var.assign(tf.multiply(main_var.value(), tau) + tf.multiply(target_var.value(), 1 - tau)))
        return self.sess.run(copy_op)

    def train(self, action_input, state_input, predicted_q):
        return self.sess.run(self.optimize, feed_dict={self.action_input: action_input, self.state_input: state_input, self.predicted_q: predicted_q})

    def action_gradient(self, action_input, state_input):
        return self.sess.run(self.action_grads, feed_dict={self.action_input: action_input, self.state_input: state_input})

    def predict(self, action_input, state_input):
        return self.sess.run(self.out, feed_dict={self.action_input: action_input, self.state_input: state_input})

    def predict_target(self, action_input, state_input):
        return self.sess.run(self.target_out, feed_dict={self.target_action_input: action_input, self.target_state_input:state_input})

class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.tau = 0.1
        self.batch_size = 50
        self.sess = tf.Session()
        self.actor = Actor(self.sess, state_dim, action_dim, self.actor_lr, self.tau, self.batch_size)
        self.critic = Critic(self.sess, state_dim, action_dim, self.critic_lr, self.tau, self.batch_size)
        self.sess.run(tf.global_variables_initializer())
        assign_target(self.sess, self.actor.main_network_param, self.actor.target_network_param)
        assign_target(self.sess, self.critic.main_network_param, self.critic.target_network_param)
        self.MAX_LENGTH = 20000
        self.memory = []


    def get_action(self, state, ep):
        limit = 100
        if ep < limit and ep is not -1:
            action = np.random.normal(0.3, 1.5) + 0.5
            if action > 1:
                action = 1
            elif action < -1:
                action = -1
            print('real action: ', action)
            return [action]
        c = 0
        if ep == -1:
            c = 0
        elif ep < 1000:
            ep /= 100
            c = 1 - (ep/10)
        state = np.reshape(state, [1, -1])
        action = self.actor.predict_target(state)[0]
        print('real action: ', action)
        noise = np.random.normal(0, 1.5*c)
        action += noise
        if action > 1:
            action[0] = 1
        elif action < -1:
            action[0] = -1
        return action


    def store_history(self, history):
        if len(self.memory) < self.MAX_LENGTH:
            self.memory.append(history)
        else:
            self.memory.pop(0)
            self.memory.append(history)
        return

    def return_batch(self, batch_size):
        ret = []
        shuffle(self.memory)
        ret.extend(self.memory[:batch_size])
        return ret

    def train(self, ep):
        limit = 100
        batch_sample = self.return_batch(self.batch_size)
        state_batch = []
        action_batch = []
        q_batch = []
        for i in range(len(batch_sample)):
            state_batch.append(batch_sample[i][0])
            action_batch.append([batch_sample[i][1]])
            q_batch.append([batch_sample[i][2]])
        state_batch = np.asarray(state_batch)
        action_batch = np.asarray(action_batch)
        q_batch = np.asarray(q_batch)
        if (ep>limit):
            print('action training')
            actor_grads = self.critic.action_gradient(action_batch, state_batch)
            self.actor.train(state_batch, actor_grads)
        self.critic.train(action_batch, state_batch, q_batch)
        return

    def update_target(self):
        self.actor.update_target()
        self.critic.update_target()
        return





if __name__ == "__main__":
    EPISODE = 100000
    gamma = 0.99
    limit = 100
    G_dividor = 10
    state_dim = 2
    action_dim = 1
    resume = False
    env = gym.make('MountainCarContinuous-v0')
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    action_examples = np.array([env.action_space.sample() for x in range(10000)])
    scaler_s = StandardScaler()
    scaler_a = StandardScaler()
    scaler_s.fit(observation_examples)
    scaler_a.fit(action_examples)
    agent = DDPG(state_dim, action_dim)
    saver = tf.train.Saver(tf.global_variables())
    if resume:
        saver.restore(agent.session, './DDPG.ckpt')
        print('successfully restored')
    for ep in range(EPISODE):
        ep_state = []
        ep_action = []
        ep_reward = []
        ep_G = []
        step = 0
        total_reward = 0
        s = env.reset()
        s = np.reshape(s, (-1, 2))
        s = list(scaler_s.transform(s)[0])
        while True:
            step += 1
            if len(agent.memory) > 2000:
                agent.train(ep)
            env.render()
            prev_s = s
            a = agent.get_action(s, ep)
            s, r, d, _ = env.step(a)
            if r > 50:
                print('success')
                r = 250
            total_reward += r
            s = np.reshape(s, (-1, 2))
            s = list(scaler_s.transform(s)[0])
            a = scaler_a.transform(np.reshape(a, (-1, 1)))[0]

            ep_state.append(prev_s)
            ep_action.append(a)
            ep_reward.append(r)
            if d:
                running_reward = 0
                for reward in reversed(ep_reward):
                    running_reward = gamma*running_reward + reward
                    ep_G.insert(0, running_reward)
                for i in range(len(ep_G)):
                    history = []
                    history.append(ep_state[i])
                    history.append(ep_action[i][0])
                    history.append(ep_G[i]/G_dividor)
                    agent.store_history(history)
                print('update target')
                agent.update_target()
                print('ep %d'%ep, 'total_reward: ', total_reward)
                if (ep % 5 == 0):
                    saver.save(agent.session, './DDPG.ckpt')
                break

