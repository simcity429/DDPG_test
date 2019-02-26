import gym
import tensorflow as tf
import numpy as np
from random import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from sklearn.preprocessing import StandardScaler
from scipy.stats import truncnorm

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

class DDPG:
    def __init__(self):
        self.build_model()
        self.build_target()
        self.MAX_LENGTH = 20000
        self.memory = []

    def build_actor(self):
        actor_state_input = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        actor_dense_1 = tf.layers.dense(inputs=actor_state_input, units=32, activation=tf.nn.elu)


    def build_model(self):
        state_input_c = tf.placeholder(tf.float32, shape=[None, 2], name="state_input_c")
        reward_and_next_q = tf.placeholder(tf.float32, shape=[None, 1], name='reward_and_next_q')
        action_input = tf.placeholder(tf.float32, shape=[None, 1], name='action_input')
        with tf.variable_scope('critic'):
            action_dense_1 = tf.layers.dense(inputs=action_input, units=32, activation=tf.nn.elu, name='dense_1')
            state_dense = tf.layers.dense(inputs=state_input_c, units=32, activation=tf.nn.elu, name='dense_2')
            concat = tf.concat([action_dense_1, state_dense], 1)
            q_1 = tf.layers.dense(inputs= concat, units=1, name='dense_3')

        with tf.variable_scope("actor"):
            tmp = tf.layers.dense(inputs=state_dense, units=16, activation=tf.nn.elu, name='tmp')
            action = tf.layers.dense(inputs=tmp, units=1, activation=tf.nn.tanh, name='action')

        with tf.variable_scope('actor', reuse=True):
            tmp = tf.layers.dense(inputs=state_dense, units=16, activation=tf.nn.elu, name='tmp')
            debug = tf.layers.dense(inputs=tmp, units=1, name='action')

        with tf.variable_scope('critic', reuse=True):
            action_dense_2 = tf.layers.dense(inputs=action, units=32, activation=tf.nn.elu, name='dense_1', reuse=True)
            concat = tf.concat([action_dense_2, state_dense], 1)
            q_2 = tf.layers.dense(inputs=concat, units=1, name='dense_3', reuse=True)

        loss_a = tf.reduce_mean(-q_2)
        loss_c = tf.reduce_mean(tf.square(q_1 - reward_and_next_q))

        self.debug = debug

        actor_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_a,
                                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))
        critic_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_c,
                                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'))

        self.loss_a = loss_a
        self.loss_c = loss_c

        self.action = action

        self.action_input = action_input
        self.state_input_c = state_input_c
        self.reward_and_next_q = reward_and_next_q
        self.q_1 = q_1
        self.q_2 = q_2

        self.actor_opt = actor_opt
        self.critic_opt = critic_opt
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

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
        sess = self.session
        state = np.reshape(state, [1, -1])
        action = sess.run(self.action, feed_dict={self.state_input_c: state})[0]
        print('real action: ', action)
        noise = np.random.normal(0, 1.5*c)
        action += noise
        if action > 1:
            action[0] = 1
        elif action < -1:
            action[0] = -1
        return action

    def get_q(self, state, action, flag):
        sess = self.session
        state = np.reshape(state, [1, -1])
        if flag:
            action = np.reshape(action, [1, -1])
        else:
            action = sess.run(self.action, feed_dict={self.state_input_c: state})
        return sess.run(self.q_t, feed_dict={self.state_input_t: state, self.action_input_t: action})[0]

    def store_history(self, history):
        if len(self.memory) < self.MAX_LENGTH:
            self.memory.append(history)
        else:
            self.memory.pop(0)
            self.memory.append(history)

    def return_batch(self, batch_size):
        ret = []
        shuffle(self.memory)
        ret.extend(self.memory[:batch_size])
        return ret

    def train(self, ep):
        limit = 100
        sess = self.session
        batch_sample = self.return_batch(150)
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
            sess.run(self.actor_opt, feed_dict={self.state_input_c: state_batch})
        sess.run(self.critic_opt, feed_dict={self.state_input_c: state_batch, self.action_input:action_batch, self.reward_and_next_q:q_batch})
        print(sess.run([self.loss_a, self.loss_c], feed_dict={self.state_input_c: state_batch, self.action_input:action_batch, self.reward_and_next_q:q_batch}))
        print('debug: ', sess.run([self.debug], feed_dict={self.state_input_c: state_batch})[0][:5])


    def build_target(self):
        state_input_t = tf.placeholder(tf.float32, shape=[None, 2], name="state_input_t")
        action_input_t = tf.placeholder(tf.float32, shape=[None, 1], name='action_input_t')
        with tf.variable_scope('target'):
            action_dense_1 = tf.layers.dense(inputs=action_input_t, units=32, activation=tf.nn.elu, name='dense_1')
            state_dense = tf.layers.dense(inputs=state_input_t, units=32, activation=tf.nn.elu, name='dense_2')
            concat = tf.concat([action_dense_1, state_dense], 1)
            q_t = tf.layers.dense(inputs= concat, units=1, name='dense_3')

        self.state_input_t = state_input_t
        self.action_input_t = action_input_t
        self.q_t = q_t
        self.session.run(tf.global_variables_initializer())
        self.update_target()


    def update_target(self):
        tau = 0.5
        copy_op = []
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        for critic_var, target_var in zip(critic_vars, target_vars):
            copy_op.append(target_var.assign(tf.multiply(critic_var.value(), tau) + tf.multiply(target_var.value(), 1 - tau)))

        self.session.run(copy_op)




if __name__ == "__main__":
    EPISODE = 100000
    gamma = 0.99
    limit = 100
    G_dividor = 10
    resume = False
    env = gym.make('MountainCarContinuous-v0')
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    action_examples = np.array([env.action_space.sample() for x in range(10000)])
    scaler_s = StandardScaler()
    scaler_a = StandardScaler()
    scaler_s.fit(observation_examples)
    scaler_a.fit(action_examples)
    agent = DDPG()
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

