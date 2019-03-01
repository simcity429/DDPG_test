import gym
import tensorflow as tf
import numpy as np
from random import shuffle
from sklearn.preprocessing import StandardScaler

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def update_target(sess, tau, main_network_param, target_network_param):
    copy_op = []
    for main_var, target_var in zip(main_network_param, target_network_param):
        copy_op.append(target_var.assign(tf.multiply(main_var.value(), tau) + tf.multiply(target_var.value(), 1 - tau)))
    return sess.run(copy_op)

def assign_target(sess, main_network_param, target_network_param):
    assign_op = []
    for main_var, target_var in zip(main_network_param, target_network_param):
        assign_op.append(target_var.assign(main_var.value()))
    return sess.run(assign_op)

class DDPG:
    def __init__(self, action_dim, state_dim):
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        self.tau = 0.1
        self.batch_size = 50
        self.action_dim, self.state_dim = action_dim, state_dim

        self.actor_state_input, self.actor_out, self.critic_state_input, self.critic_action_input, \
        self.action_gradient, self.q = self.build_model('main')

        self.target_actor_state_input, self.target_actor_out, self.target_critic_state_input, \
        self.target_critic_action_input, _, self.target_q = self.build_model('target')

        self.sess = tf.keras.backend.get_session()


        self.main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_main') \
                         + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_main')
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target') \
                           + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_target')


        self.predicted_q = tf.keras.layers.Input(shape=[1])
        self.action_gradient_input = tf.keras.layers.Input(shape=[1])

        tmp = tf.gradients(self.actor_out, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_main'), -self.action_gradient_input)
        self.actor_gradient = list(map(lambda x: tf.div(x, self.batch_size), tmp))
        grads_and_vars = zip(self.actor_gradient, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_main'))


        self.loss_c = tf.keras.backend.mean(tf.keras.backend.square(self.predicted_q - self.q))

        self.actor_opt = tf.train.AdamOptimizer(learning_rate=self.actor_lr).apply_gradients(grads_and_vars)
        self.critic_opt = tf.train.AdamOptimizer(learning_rate=self.critic_lr).minimize(self.loss_c, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_main'))

        self.MAX_LENGTH = 20000
        self.memory = []

    def get_action_grad(self, state, action):
        return self.sess.run(self.action_gradient, feed_dict={self.critic_state_input: state, self.critic_action_input:action})





    def build_model(self, name):
        initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        with tf.variable_scope('actor'+'_'+name):
            actor_state_input = tf.keras.layers.Input(shape=[self.state_dim])
            out = tf.keras.layers.Dense(units=100, activation='elu')(actor_state_input)
            out = tf.keras.layers.BatchNormalization()(out)
            out = tf.keras.layers.Dense(units=100, activation='elu')(out)
            actor_out = tf.keras.layers.Dense(units=self.action_dim, activation='tanh', use_bias=False, kernel_initializer=initializer)(out)

        with tf.variable_scope('critic'+'_'+name):
            critic_state_input = tf.keras.layers.Input(shape=[self.state_dim])
            critic_action_input = tf.keras.layers.Input(shape=[self.action_dim])

            process_action = tf.keras.layers.Dense(units=100, activation='elu', name='1')(critic_action_input)
            process_action = tf.keras.layers.BatchNormalization(name='2')(process_action)

            process_state = tf.keras.layers.Dense(units=100, activation='elu', name='3')(critic_state_input)
            process_state = tf.keras.layers.BatchNormalization(name='4')(process_state)
            process_state = tf.keras.layers.Dense(units=100, activation='elu', name='5')(process_state)

            q = tf.keras.layers.Dense(units=1, name='7', kernel_initializer=initializer)\
                (tf.keras.layers.Dense(units=50, activation='elu', name='6')(tf.add(process_action, process_state)))
            action_gradient = tf.gradients(q, critic_action_input)

        return actor_state_input, actor_out, critic_state_input, critic_action_input, action_gradient, q

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

    def get_action(self, state, ep):
        limit = 20
        if ep < limit and ep is not -1:
            action = np.random.normal(0.3, 1.5) + 0.5
            if action > 1:
                action = 1.0
            elif action < -1:
                action = -1.0
            print('real action: ', action)
            return [action]
        c = 0
        if ep == -1:
            c = 0
        elif ep < 1000:
            ep /= 100
            c = 1 - (ep / 10)
        state = np.reshape(state, [1, -1])
        action = self.actor_predict_target(state)[0]
        print('real action: ', action)
        noise = np.random.normal(0, 0.5 * c)
        action += noise
        if action > 1:
            action[0] = 1.0
        elif action < -1:
            action[0] = -1.0
        return action

    def actor_predict_target(self, state):
        return self.sess.run(self.target_actor_out, feed_dict={self.target_actor_state_input: state})

    def train(self, ep):
        limit = 20
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
            action_grad = self.get_action_grad(state_batch, action_batch)[0]
            self.actor_train(state_batch, action_grad)
        self.critic_train(state_batch, action_batch, q_batch)
        return

    def actor_train(self, state, action_grad):
        return self.sess.run(self.actor_opt, feed_dict={self.actor_state_input: state, self.action_gradient_input:action_grad})

    def critic_train(self, state, action, q_batch):
        return self.sess.run(self.critic_opt, feed_dict={self.critic_state_input: state, self.critic_action_input: action, self.predicted_q:q_batch})

    def update_target(self):
        return update_target(self.sess, self.tau, self.main_vars, self.target_vars)







if __name__ == "__main__":
    mode = 'play'
    EPISODE = 100000
    gamma = 0.99
    limit = 20
    G_dividor = 10
    state_dim = 2
    action_dim = 1
    resume = True
    env = gym.make('MountainCarContinuous-v0')
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    action_examples = np.array([env.action_space.sample() for x in range(10000)])
    scaler_s = StandardScaler()
    scaler_a = StandardScaler()
    scaler_s.fit(observation_examples)
    scaler_a.fit(action_examples)
    agent = DDPG(action_dim, state_dim)
    agent.sess.run(tf.global_variables_initializer())
    assign_target(agent.sess, agent.main_vars, agent.target_vars)
    saver = tf.train.Saver(tf.global_variables())
    if resume:
        saver.restore(agent.sess, './DDPG_united/DDPG.ckpt')
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
            if len(agent.memory) > 2000 and mode == 'train':
                agent.train(ep)
            env.render()
            prev_s = s
            if mode=="train":
                a = agent.get_action(s, ep)
            else:
                a = agent.get_action(s, -1)
            s, r, d, _ = env.step(a)
            if r > 50:
                print('success')
                r = 100
            total_reward += r
            s = np.reshape(s, (-1, 2))
            s = list(scaler_s.transform(s)[0])
            a = scaler_a.transform(np.reshape(a, (-1, 1)))[0]

            ep_state.append(prev_s)
            ep_action.append(a)
            ep_reward.append(r)
            if d and mode == 'train':
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
                agent.train(ep)
                print('update target')
                agent.update_target()
                print('ep %d'%ep, ' total_reward: ', total_reward)
                if (ep % 5 == 0):
                    saver.save(agent.sess, './DDPG_united/DDPG.ckpt')
                break
            elif d:
                print('ep %d'%ep, ' toral_reward: ', total_reward)
                break