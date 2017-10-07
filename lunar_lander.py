# -*- coding: utf-8 -*-

######################################################
# Advanced Methods in ML course - Spring 2017
#
# Edited on Tue Jun 18 2017
# @author: morgeva
#
# Submission IDs: 200831618, 038354213
#
######################################################


###########################
# Imports

import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import cPickle as pickle


###########################
# Agent model

class Agent(object):
    def __init__(self, env_d):
        self.env_d = env_d
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.starter_learning_rate = 0.01
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 100, 0.94)

        if env_d == 'LunarLander-v2':
            self.iu, self.hu1, self.hu2, self.ou = 8, 16, 12, 4
        else: # env_d = 'CartPole-v0'
            self.iu, self.hu, self.ou = 4, 10, 2

        # Input observations
        self.observations = tf.placeholder(tf.float32, [None, self.iu], name='observation')

        # Network variables
        self.W1 = tf.Variable(tf.random_normal([self.iu, self.hu1], stddev=0.35), name='W1')
        self.b1 = tf.Variable(tf.random_normal([self.hu1], stddev=0.35), name='b1')
        self.W2 = tf.Variable(tf.random_normal([self.hu1, self.hu2], stddev=0.35), name='W2')
        self.b2 = tf.Variable(tf.random_normal([self.hu2], stddev=0.35), name='b2')
        self.W3 = tf.Variable(tf.random_normal([self.hu2, self.ou], stddev=0.35), name='W3')
        self.b3 = tf.Variable(tf.random_normal([self.ou], stddev=0.35), name='b3')

        self.h1 = tf.nn.tanh(tf.add(tf.matmul(self.observations, self.W1), self.b1), name="h1")
        self.h2 = tf.nn.tanh(tf.add(tf.matmul(self.h1, self.W2), self.b2), name="h2")
        self.y = tf.nn.softmax(tf.add(tf.matmul(self.h2, self.W3), self.b3), name="y")

        # Gradients calculation
        self.actions = tf.placeholder(tf.int32, [None], name="y_max")
        self.rewards = tf.placeholder(tf.float32, [None], name="rewards")

        self.agent_vars = tf.trainable_variables()
        self.agent_vars_num = len(self.agent_vars)

        self.action_masks = tf.one_hot(self.actions, self.ou, name="action_masks")
        self.loglik = tf.reduce_sum(tf.multiply(tf.log(self.y), self.action_masks), axis=1)
        self.episode_loss = tf.negative(tf.reduce_sum(tf.multiply(self.loglik, self.rewards)))
        self.episode_grads = tf.gradients(self.episode_loss, self.agent_vars, name="episode_grads")

        # Policy learning
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.W1_grad = tf.placeholder(tf.float32, name="W1_grad")
        self.b1_grad = tf.placeholder(tf.float32, name="b1_grad")
        self.W2_grad = tf.placeholder(tf.float32, name="W2_grad")
        self.b2_grad = tf.placeholder(tf.float32, name="b2_grad")
        self.W3_grad = tf.placeholder(tf.float32, name="W3_grad")
        self.b3_grad = tf.placeholder(tf.float32, name="b3_grad")
        batch_grads = [self.W1_grad, self.b1_grad, self.W2_grad, self.b2_grad, self.W3_grad, self.b3_grad]
        self.optimizer_update = self.optimizer.apply_gradients(zip(batch_grads, self.agent_vars), global_step=self.global_step)

        # Parameters loading (for evaluation)
        self.W1_val = tf.placeholder(tf.float32, [self.iu, self.hu1], name="W1_val")
        self.b1_val = tf.placeholder(tf.float32, [self.hu1], name="b1_val")
        self.W2_val = tf.placeholder(tf.float32, [self.hu1, self.hu2], name="W2_val")
        self.b2_val = tf.placeholder(tf.float32, [self.hu2], name="b2_val")
        self.W3_val = tf.placeholder(tf.float32, [self.hu2, self.ou], name="W3_val")
        self.b3_val = tf.placeholder(tf.float32, [self.ou], name="b3_val")

        self.W1_assign = self.W1.assign(self.W1_val)
        self.b1_assign = self.b1.assign(self.b1_val)
        self.W2_assign = self.W2.assign(self.W2_val)
        self.b2_assign = self.b2.assign(self.b2_val)
        self.W3_assign = self.W3.assign(self.W3_val)
        self.b3_assign = self.b3.assign(self.b3_val)

        self.init = tf.global_variables_initializer()


###########################
# Helper functions

def store_params(sess, agent, fname):
    agent_vars = sess.run(agent.agent_vars)
    with open(fname, 'wb') as fp:
        pickle.dump(agent_vars, fp)


def load_params(sess, agent, fname):
    with open(fname, 'rb') as fp:
        agent_vars = pickle.load(fp)
        sess.run([agent.W1_assign, agent.b1_assign, agent.W2_assign, agent.b2_assign, agent.W3_assign, agent.b3_assign],
                 feed_dict={agent.W1_val: agent_vars[0], agent.b1_val: agent_vars[1], agent.W2_val: agent_vars[2],
                            agent.b2_val: agent_vars[3], agent.W3_val: agent_vars[4], agent.b3_val: agent_vars[5]})


def cumulative_rewards(rwrds, gamma):
    T = len(rwrds)
    cum_rewards = np.zeros_like(rwrds)
    cum_sum = 0.0
    for t in reversed(xrange(0,T)):
        cum_sum = cum_sum * gamma + rwrds[t]
        cum_rewards[t] = cum_sum
    return cum_rewards


###########################
# Execution procedures

def test(env, agent, params_fname, total_episodes, batch_size, display_size):
    with tf.Session() as sess:
        sess.run(agent.init)
        load_params(sess, agent, params_fname)

        reward_sum = 0
        reward_avg = 0
        episode_idx = 0
        obsrv = env.reset()
        while episode_idx < total_episodes:
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(agent.y, feed_dict={agent.observations: np.reshape(obsrv, [1, agent.iu])})
            action_probs_fix = np.asarray(action_probs[0], dtype="float64")
            action_probs_fix /= np.sum(action_probs_fix)
            action = np.argmax(np.random.multinomial(1, action_probs_fix))

            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)
            reward_sum += reward

            if done:
                episode_idx += 1
                reward_avg = float(reward_avg * (episode_idx - 1) + reward_sum) / float(episode_idx)

                if episode_idx % (display_size * batch_size) == 0:
                    print "episode %d avg.reward %.2f" % (episode_idx, reward_avg)

                reward_sum = 0
                obsrv = env.reset()


def train(env, agent, logf, params_fname, total_episodes, batch_size, display_size, gamma):
    with tf.Session() as sess:
        sess.run(agent.init)
        #writer = tf.summary.FileWriter('./tflogs', sess.graph)
        grads_buffer = sess.run(agent.agent_vars)
        for i in range(agent.agent_vars_num):
            grads_buffer[i] *= 0.0

        reward_sum = 0
        T_sum = 0
        num_rests = 0

        best_avg_reward = -np.inf
        episode_idx = 0
        obsrv = env.reset()
        obsrvs, acts, rwrds = [], [], []
        print "starter learning rate: {} gamma: {} batch size: {}".format(agent.starter_learning_rate, gamma, batch_size)
        while episode_idx < total_episodes:
            obsrvs.append(obsrv)
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(agent.y, feed_dict={agent.observations: np.reshape(obsrv, [1, agent.iu])})
            action_probs_fix = np.asarray(action_probs[0], dtype="float64")
            action_probs_fix /= np.sum(action_probs_fix)
            action = np.argmax(np.random.multinomial(1, action_probs_fix))

            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)
            reward_sum += reward
            acts.append(action)
            rwrds.append(reward)

            if done:
                episode_idx += 1
                T_sum += len(obsrvs)
                if reward == 100:
                    num_rests += 1

                # computing the episode gradients
                cum_rewards = cumulative_rewards(rwrds, gamma)
                cum_rewards -= np.mean(cum_rewards)
                cum_rewards /= np.std(cum_rewards)
                grads = sess.run(agent.episode_grads, feed_dict={agent.observations: np.vstack(obsrvs),
                                                                 agent.actions: acts,
                                                                 agent.rewards: cum_rewards})
                for i in range(agent.agent_vars_num):
                    grads_buffer[i] += grads[i]

                # update the agent parameters every batch
                if episode_idx % batch_size == 0:
                    for i in range(agent.agent_vars_num):
                        grads_buffer[i] /= float(batch_size)

                    _ = sess.run(agent.optimizer_update, feed_dict={agent.W1_grad: grads_buffer[0], agent.b1_grad: grads_buffer[1],
                                                                    agent.W2_grad: grads_buffer[2], agent.b2_grad: grads_buffer[3],
                                                                    agent.W3_grad: grads_buffer[4], agent.b3_grad: grads_buffer[5]})
                    for i in range(agent.agent_vars_num):
                        grads_buffer[i] *= 0.0

                    # check performance
                    batch_avg_reward = float(reward_sum)/float(batch_size)
                    batch_avg_T = float(T_sum)/float(batch_size)
                    rests_prcnt = float(num_rests)/float(batch_size)
                    #logf.write("{},{},{},{}\n".format(episode_idx, batch_avg_T, rests_prcnt, batch_avg_reward))
                    if batch_avg_reward >= best_avg_reward:
                        best_avg_reward = batch_avg_reward
                        store_params(sess, agent, params_fname)
                        print "episode %d avg.T %.2f pcnt.rests %.2f avg.reward %.4f - best so far!" % (episode_idx, batch_avg_T, rests_prcnt, batch_avg_reward)

                    reward_sum = 0
                    T_sum = 0
                    num_rests = 0

                    if episode_idx % (display_size * batch_size) == 0:
                        print "episode %d avg.T %.2f pcnt.rests %.2f avg.reward %.4f" % (episode_idx, batch_avg_T,
                                                                                         rests_prcnt, batch_avg_reward)

                obsrv = env.reset()
                obsrvs, acts, rwrds = [], [], []

        # writer.close()


###########################
# Main

def main(argv):
    # execution parameters
    env_d = 'LunarLander-v2'
    # env_d = 'CartPole-v0'
    is_test = False
    total_episodes = 30000      # stopping criteria
    batch_size = 20             # batch size
    display_size = 5            # display every X batches
    gamma = 0.99                # discount parameter

    # logging
    #log_fname = '{}.log'.format(argv[1])
    #params_fname = '{}_ws.p'.format(argv[1])
    params_fname = 'ws.p'

    # initialization
    agent = Agent(env_d)
    env = gym.make(env_d)
    #env = wrappers.Monitor(env, './LunarLander-experiment-1', force=True)

    # execution
    if is_test:
        test(env, agent, params_fname, total_episodes, batch_size, display_size)
    else:
        #logf = open(log_fname, 'w')
        #train(env, agent, logf, params_fname, total_episodes, batch_size, display_size, gamma)
        train(env, agent, None, params_fname, total_episodes, batch_size, display_size, gamma)
        #logf.close()


if __name__ == '__main__':
    tf.app.run()
