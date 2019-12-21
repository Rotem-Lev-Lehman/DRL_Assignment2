import time
import gym
import numpy as np
import tensorflow as tf
import collections
from ModifiedTensorBoard import ModifiedTensorBoard

env = gym.make('CartPole-v1')
np.random.seed(1)
actor_critic = True
baseline = True
start_time = time.time()
rewards_num = 0
satisfying_average = 475


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(name, int(time.time())))

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            #self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.output = tf.add(tf.matmul(self.Z1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            #self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.output = tf.add(tf.matmul(self.Z1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.value = self.output
            #self.value = tf.squeeze(tf.nn.relu(self.output))
            # Loss with negative log probability
            #self.loss = tf.reduce_mean(self.output * self.R_t)
            self.loss = tf.reduce_mean(tf.square(self.R_t - self.value))  # loss = mse
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def set_weights(self, other):
        self.W1 = other.W1
        self.b1 = other.b1
        self.W2 = other.W2
        self.b2 = other.b2


# Define hyper parameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate = 0.0004

render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate)
value = ValueNetwork(state_size, learning_rate)


# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0

    for episode in range(max_episodes):
        policy.tensorboard.step = episode
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []

        for step in range(max_steps):
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
            episode_rewards[episode] += reward

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                else:
                    average_rewards = np.mean(episode_rewards[:episode+1])
                policy.tensorboard.update_stats(last_100_average_reward=average_rewards, reward=episode_rewards[episode])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                rewards_num += episode_rewards[episode]
                if episode > 98 and average_rewards > satisfying_average:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break

            state = next_state

        if solved:
            break

        # Compute Rt for each time-step t and update the network's weights
        # update the weights of the two networks if we are using actor critic
        for t, transition in enumerate(episode_transitions):
            if actor_critic:
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
                if transition.done:
                    next_state_value = 0
                else:
                    next_state_value = sess.run(value.value, {value.state: transition.next_state})
                target_for_value = total_discounted_return
                delta = target_for_value - sess.run(value.value, {value.state: transition.state})
                value_dict = {value.state: transition.state, value.R_t: delta}
                _, value_loss = sess.run([value.optimizer, value.loss], value_dict)
                feed_dict = {policy.state: transition.state, policy.R_t: delta, policy.action: transition.action}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                policy.tensorboard.update_stats(loss=loss)
            else:
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt
                if baseline:
                    total_discounted_return -= sess.run(value.value, {value.state: transition.state})
                    value_dict = {value.state: transition.state, value.R_t: total_discounted_return}
                    _, value_loss = sess.run([value.optimizer, value.loss], value_dict)
                feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return, policy.action: transition.action}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                policy.tensorboard.update_stats(loss=loss)

print()
total_time = time.time()-start_time
policy.tensorboard.update_stats(Overall_time=total_time, Time_per_reward=total_time/rewards_num, Time_per_episode=total_time/episode)
