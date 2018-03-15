"""
 -------------------------------------------------------------------------
 A test-implementation of the IMPALA RL algorithm, (c) deepmind; Feb. 2018

 IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
 https://arxiv.org/abs/1802.01561

 - Special async off-policy way of distributing the RL algo between:
   o explorers: only act in their own envs and store each
     episode in a global buffer (no learning); they use a local copy (mu) of the main policy (pi)
     only synched at the beginning of each episode. Thus, mu may be behind pi.
   o learners: pull batches from the global buffer and apply the learning algo to
     the main policy (pi).
 - Because exploring (and the collected experiences) is off-policy, they introduce a trick - v-traces -
   to adjust the vanilla PG update for this off-policy case.

 created: 2018/03/09 in PyCharm
 (c) 2018 Sven - ducandu research GmbH
 -------------------------------------------------------------------------
"""

import tensorflow as tf
import argparse

# the Env
from env import Env
# a clever discounted-return function using scipy's lfilter
from test_discounted_return import discount


# A simple ca 250-lines IMPALA distributed tf prototype model that runs on the local machine
# - Uses n central learners (which are function as parameter servers) and
#   m explorer agents that collect experience from a simple Env taken from Barto&Sutton "completed 2017 draft"
#   Chapter 13 (policy gradient methods).
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learner-hosts", type=str, required=True,
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("-e", "--explorer-hosts", type=str, required=True,
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("-j", "--job-name", type=str, default="worker", help="One of 'learner', 'explorer'")
    parser.add_argument("-t", "--task-index", type=int, default=0, help="Index of task within the job")
    parser.add_argument("-s", "--steps-per-explorer", type=int, default=1000000,
                        help="Max. steps an explorer should make in its env (in total before stopping).")
    parser.add_argument("-m", "--max-steps-per-episode", type=int, default=100,
                        help="Max. steps an explorer should make in one episode.")
    parser.add_argument("-b", "--buffer-size", type=int, default=1000,
                        help="Number of time steps to store at any time for each explorer.")
    parser.add_argument("--learn-batch-size", type=int, default=64,
                        help="Size of a batch (number of episodes) to pull randomly from the main buffer "
                             "for each learner iteration.")
    parser.add_argument("-f", "--upload-frequency", type=int, default=4,
                        help="Every how many episodes does an explorer upload its local buffer of episodes "
                             "to the learners?")
    parser.add_argument("--num-hidden", type=int, default=10,
                        help="Number of hidden nodes.")
    parser.add_argument("-g", "--gamma", type=float, default=0.97,
                        help="The discount factor gamma (default 0.9).")
    parser.add_argument("-a", "--learning-rate", type=float, default=0.001,
                        help="The learning rate (alpha) to use for optimizing the cost.")
    args = parser.parse_args()
    learner_hosts = args.learner_hosts.split(",")
    explorer_hosts = args.explorer_hosts.split(",")

    # Create a cluster from the given hosts (learners and explorers).
    cluster = tf.train.ClusterSpec({"learner": learner_hosts, "explorer": explorer_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=args.job_name, task_index=args.task_index)

    # simple 1 hidden layer, feed-forward network
    num_inputs = 1  # input is always [1.0] -> blind env
    num_hidden = args.num_hidden
    num_out = 2 + 1  # left and right actions + value function
    # 1 disc. return, 1 action (0=left, 1=right),
    len_buffer_record = 2

    # the global main policy: pi (live on the learner(s))
    # - explorers sync their own policies (mu) with this at the beginning of an episode
    with tf.device(tf.train.replica_device_setter(ps_tasks=len(learner_hosts),
                                                  ps_device="/job:learner",
                                                  worker_device="/job:explorer/task:0")
                   ):
        weights_1_pi = tf.Variable(tf.truncated_normal(shape=(num_inputs, num_hidden)), name="pi-W1")
        biases_1_pi = tf.Variable(tf.zeros(shape=(num_hidden,)), name="pi-b1")
        weights_2_pi = tf.Variable(tf.truncated_normal(shape=(num_hidden, num_out)), name="pi-W2")
        biases_2_pi = tf.Variable(tf.zeros(shape=(num_out,)), name="pi-b2")

    # main experience buffer (on central learner)
    # - create experience buffer in both explorer and learner, but host it on learner
    # - each explorer writes to a certain chunk in round robin fashion
    size_main_experience_buffer = args.buffer_size * len(explorer_hosts)
    with tf.device("/job:learner/task:0"):
        main_experience_buffer = tf.Variable(tf.zeros([size_main_experience_buffer,
                                                       len_buffer_record]),
                                             name="global-episode-buffer")
        global_step = tf.train.get_or_create_global_step()

    if args.job_name == "explorer":
        # local policy (mu) -> all zero; will be sync'd with pi anyway at start of each episode
        weights_1_mu = tf.Variable(tf.zeros(shape=(num_inputs, num_hidden)), name="mu-W1")
        biases_1_mu = tf.Variable(tf.zeros(shape=(num_hidden,)), name="mu-b1")
        weights_2_mu = tf.Variable(tf.zeros(shape=(num_hidden, num_out)), name="mu-W2")
        biases_2_mu = tf.Variable(tf.zeros(shape=(num_out,)), name="mu-b2")

        # ops that sync from the main policy (pi) (using locking)
        # - must fetch these after a reset of the env (before querying the first action in each episode)
        sync_ops = [
            tf.assign(weights_1_mu, weights_1_pi, name="sync-W1"),
            tf.assign(biases_1_mu, biases_1_pi, name="sync-b1"),
            tf.assign(weights_2_mu, weights_2_pi, name="sync-W2"),
            tf.assign(biases_2_mu, biases_2_pi, name="sync-b2")
        ]
        # number of actions (log-action-probs) to sample
        num_action_samples = tf.placeholder(dtype=tf.int32, shape=())
        # Buffer to store n (capacity) episodes of experiences (round-robin).
        # This one gets inserted into the learners global memory after each m (upload-frequency) episodes.
        # rank0=episode, rank1=step in episode, rank2=[action(0=left, 1=right), reward, mu(a)]
        experience_buffer = tf.Variable(tf.zeros([args.buffer_size, len_buffer_record]), name="episode-buffer")
        experience_buffer_idx = tf.placeholder(dtype=tf.int32, shape=())
        # In case we would like to use LSTM -> need to pass the initial internal state to learner
        # so it can replay the episode through pi (instead of mu).
        # init_internal_buffer = tf.Variable(tf.zeros([args.buffer_size, num_internal_state]),
        #                                   name="init-internal-buffer")

        # upload op (from local experience buffer to global one)
        start = args.task_index * args.buffer_size
        stop = start + args.buffer_size
        experience_upload = tf.assign(main_experience_buffer[start:stop, :], experience_buffer)

        # forward pass -> let local explorer handle this (as it's needed right here for querying actions)
        hidden_out = tf.add(tf.matmul(tf.ones(shape=(num_action_samples, num_inputs)), weights_1_mu), biases_1_mu)
        logits = tf.add(tf.matmul(hidden_out, weights_2_mu), biases_2_mu)
        action_prob = tf.nn.softmax(logits[:, :2])  # first two outputs are action logits
        action_distr = tf.distributions.Bernoulli(probs=action_prob[:, 1])
        actions = action_distr.sample(sample_shape=num_action_samples)

        # store incoming step (a, R) in local experience buffer
        a_in = tf.placeholder(dtype=tf.float32, shape=(None,), name="a-in")  # 0.0=left, 1.0=right
        returns_in = tf.placeholder(dtype=tf.float32, shape=(None,), name="returns-in")  # None=timesteps in the episode
        # concat returns and log_aps within each timestep
        episode = tf.concat([tf.expand_dims(a_in, 1), tf.expand_dims(returns_in, 1)], 1)
        episode_len = tf.shape(episode)[0]
        stop = experience_buffer_idx + episode_len
        # don't have to lock as the only one that's ever touching the local buffer is ourselves
        add_episode = tf.cond(stop <= args.buffer_size,
                              # true fn
                              lambda: tf.group(tf.assign(experience_buffer[experience_buffer_idx:stop],
                                                         episode, use_locking=False)),
                              # false fn
                              lambda: tf.group(
                                  tf.assign(experience_buffer[experience_buffer_idx:],
                                            episode[:args.buffer_size-experience_buffer_idx], use_locking=False),
                                  tf.assign(experience_buffer[:episode_len-(args.buffer_size-experience_buffer_idx)],
                                            episode[args.buffer_size-experience_buffer_idx:], use_locking=False)
                              )
                              )

        # create our own private env
        env = Env()

        total_steps = 0
        num_episodes = 0

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(args.task_index == 0),
                                               hooks=[]) as mon_sess:
            while total_steps < args.steps_per_explorer:
                rs = []  # discounted accum. rewards over one episode
                as_ = []  # the actual actions taken
                # reset the env
                env.reset()
                episode_steps = 0
                buffer_idx = 0
                # update our mu with pi from learner
                fetches = mon_sess.run(sync_ops)

                while episode_steps < args.max_steps_per_episode and total_steps < args.steps_per_explorer:
                    s = env.state
                    a = mon_sess.run(actions, feed_dict={num_action_samples: 1})
                    a = a[0][0]  # a=0 for 'left', a=1 for 'right'
                    as_.append(a)
                    r, is_terminal = env.execute(a)
                    rs.append(r)

                    print("{:03d} explorer {}: s={} action {} (1=right) s'={} is-term={}".
                          format(total_steps, args.task_index, s, a, env.state, is_terminal))
                    total_steps += 1
                    episode_steps += 1

                    if is_terminal:
                        # calculate discounted accumulated rewards (returns)
                        returns = discount(rs, args.gamma)

                        # add episode to our buffer
                        mon_sess.run([add_episode],
                                     feed_dict={returns_in: returns, a_in: as_, experience_buffer_idx: buffer_idx})
                        env.reset()
                        buffer_idx = (buffer_idx + len(rs)) % args.buffer_size
                        rs = []
                        as_ = []
                        episode_steps = 0
                        num_episodes += 1
                        fetches = sync_ops
                        if num_episodes % args.upload_frequency == 0:
                            fetches.append(experience_upload)
                        fetches = mon_sess.run(fetches)
        print("Explorer {} is done!".format(args.task_index))

    # - every learner iteration, it samples randomly from the main buffer and learns
    else:
        # build the pi-network (similar to mu-network above)
        hidden_out = tf.add(tf.matmul(tf.ones(shape=(args.learn_batch_size, num_inputs)), weights_1_pi), biases_1_pi)
        logits = tf.add(tf.matmul(hidden_out, weights_2_pi), biases_2_pi)
        action_right_prob = tf.nn.softmax(logits[:, :2])[:, 1]  # first two outputs are action logits (1=right)
        avg_action_right_prob = tf.reduce_mean(action_right_prob)
        values = logits[:, 2:3]  # last output is the state-value

        # for now: do simple REINFORCE (add v-trace later or directly to tensorforce as it's not really different)
        # get a random batch from the main buffer
        indexes = tf.random_shuffle(tf.range(size_main_experience_buffer))
        sample = tf.gather(main_experience_buffer, indexes[:args.learn_batch_size])
        # separate discounted accum. reward and action (0=left, 1=right)
        actions, returns = tf.split(sample, num_or_size_splits=len_buffer_record, axis=1)
        # probability of the action actually taken
        action_prob = tf.abs(tf.subtract(actions, tf.ones(tf.shape(actions))) + action_right_prob)
        log_action_prob = tf.log(action_prob)

        # define our loss function
        alpha_1 = 0.5  # log-action-prob loss
        alpha_2 = 0.4  # value function loss
        alpha_3 = 0.1  # regularization

        # reduce-mean on batch and timestep axes
        advantage = (returns - values)
        loss = - alpha_1 * tf.reduce_mean(tf.multiply(log_action_prob, advantage)) + \
               alpha_2 * tf.reduce_mean(0.5 * tf.square(advantage)) + \
               alpha_3 * 0  # TODO: add regularization

        train_op = tf.train.AdagradOptimizer(args.learning_rate).minimize(loss, global_step=global_step)

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(args.task_index == 0),
                                               hooks=[]) as mon_sess:
            while not mon_sess.should_stop():
                sample_out, _, loss_out, g_step, avg_right_prob_out = mon_sess.run(
                    [sample, train_op, loss, global_step, avg_action_right_prob]
                )
                print("task {} step {} loss {} avg-right-prob={}".
                      format(args.task_index, g_step, loss_out, avg_right_prob_out))
        print("Learner {} is done!".format(args.task_index))


if __name__ == "__main__":
    main()
