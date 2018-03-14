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
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument("-l", "--learner-hosts", type=str, required=True,
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("-e", "--explorer-hosts", type=str, required=True,
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument("-j", "--job-name", type=str, default="worker", help="One of 'learner', 'explorer'")
    parser.add_argument("-t", "--task-index", type=int, default=0, help="Index of task within the job")
    parser.add_argument("-s", "--steps-per-explorer", type=int, default=10000,
                        help="Max. steps an explorer should make in its env (in total before stopping).")
    parser.add_argument("-m", "--max-steps-per-episode", type=int, default=100,
                        help="Max. steps an explorer should make in one episode.")
    parser.add_argument("-b", "--buffer-size", type=int, default=100,
                        help="Number of time steps to store at any time for each explorer.")
    parser.add_argument("--learn-batch-size", type=int, default=32,
                        help="Size of a batch (number of episodes) to pull randomly from the main buffer "
                             "for each learner iteration.")
    parser.add_argument("-f", "--upload-frequency", type=int, default=4,
                        help="Every how many episodes does an explorer upload its local buffer of episodes "
                             "to the learners?")
    parser.add_argument("--num-hidden", type=int, default=10,
                        help="Number of hidden nodes.")
    parser.add_argument("-g", "--gamma", type=float, default=0.9,
                        help="The discount factor gamma (default 0.9).")
    parser.add_argument("-a", "--learning-rate", type=float, default=0.0001,
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
    len_buffer_record = 3  # 1 disc. return, 1 log(action-right-probability) (from network), 1 value (from network)

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
        # number of actions to sample (usually 1 as we are doing step-by-step exploration)
        num_action_samples = tf.placeholder(dtype=tf.int32, shape=())

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
        values = logits[:, 2]  # last output is the state-value
        log_action_prob = tf.log(action_prob)
        action_distr = tf.distributions.Bernoulli(probs=action_prob[:, 1])
        actions = action_distr.sample(sample_shape=num_action_samples)

        # store incoming step (R, log(a), V) in local experience buffer
        returns_in = tf.placeholder(dtype=tf.float32, shape=(None,), name="returns-in")  # None=timesteps in the episode
        log_ap_in = tf.placeholder(dtype=tf.float32, shape=(None,), name="log-a-prob-in")
        v_in = tf.placeholder(dtype=tf.float32, shape=(None,), name="V-in")
        # concat returns and log_aps within each timestep
        episode = tf.concat([tf.expand_dims(returns_in, 1), tf.expand_dims(log_ap_in, 1), tf.expand_dims(v_in, 1)], 1)
        episode_len = tf.shape(episode)[0] + 1
        # don't have to lock as the only one that's ever touching the local buffer is ourselves
        add_episode = tf.assign(experience_buffer[experience_buffer_idx:experience_buffer_idx+episode_len],
                                episode, use_locking=False)

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
            while not total_steps > args.steps_per_explorer:
                rs = []  # discounted accum. rewards over one episode
                log_aps = []  # log action probabilities over one episode
                vs = []  # the value outputs produced  by the network
                # reset the env
                env.reset()
                episode_steps = 0
                buffer_idx = 0
                # update our mu with pi from learner
                fetches = mon_sess.run(sync_ops)

                while episode_steps < args.max_steps_per_episode and total_steps < args.steps_per_explorer:
                    s = env.state
                    a, log_ap, v = mon_sess.run([actions, log_action_prob, values], feed_dict={num_action_samples: 1})
                    a = a[0][0]  # a=0 for 'left', a=1 for 'right'
                    log_ap = log_ap[0][a]  # probability of picking a
                    v = v[0]
                    log_aps.append(log_ap)
                    vs.append(v)
                    r, is_terminal = env.execute(a)
                    rs.append(r)

                    print("{:03d} explorer {}: s={} action {} (1=right) s'={} is-term={} v-out={}".
                          format(total_steps, args.task_index, s, a, env.state, is_terminal, v))
                    total_steps += 1
                    episode_steps += 1

                    if is_terminal:
                        # calculate discounted accum rewards (returns)
                        returns = discount(rs, args.gamma)

                        # add episode to our buffer
                        mon_sess.run([add_episode], feed_dict={returns_in: returns, log_ap_in: log_aps,
                                                               v_in: vs,
                                                               experience_buffer_idx: buffer_idx})

                        env.reset()
                        rs = []
                        log_aps = []
                        vs = []
                        episode_steps = 0
                        buffer_idx = (buffer_idx + 1) % args.buffer_size
                        num_episodes += 1
                        fetches = sync_ops
                        if num_episodes % args.upload_frequency == 0:
                            fetches.append(experience_upload)
                        fetches = mon_sess.run(fetches)

    # - every learner iteration, it samples randomly from the main buffer and learns
    else:
        # for now: do simple REINFORCE (add v-trace later or directly to tensorforce as it's not really different)

        # get a random batch from the main buffer
        indexes = tf.random_shuffle(tf.range(size_main_experience_buffer))
        # sample is a rank3 tensor with rank0=batch (episode), rank1=timestep,
        # rank2=[accum return, log-action-right-prob]
        sample = tf.gather(main_experience_buffer, indexes[:args.learn_batch_size])
        # separate discounted accum. reward and log-action-right-prob
        [returns, log_aps, values] = tf.split(sample, num_or_size_splits=3, axis=1)

        # define our loss function
        alpha_1 = 0.5  # log-action-prob loss
        alpha_2 = 0.4  # value function loss
        alpha_3 = 0.1  # regularization
        # TODO: add regularization
        # reduce-mean on batch and timestep axes
        avg_log_right_prob = tf.reduce_mean(log_aps, (0, 1))
        loss = - alpha_1 * avg_log_right_prob + \
               alpha_2 * tf.reduce_mean(0.5 * tf.square(returns - values), (0, 1)) + \
               alpha_3 * 0
        avg_right_prob = tf.exp(avg_log_right_prob)

        train_op = tf.train.AdagradOptimizer(0.001).minimize(loss, global_step=global_step)

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(args.task_index == 0),
                                               hooks=[]) as mon_sess:
            while not mon_sess.should_stop():
                _, loss_out, g_step, avg_prob_right_out = mon_sess.run([train_op, loss, global_step, avg_right_prob])
                print("task {} step {} loss {} avg-right-prob={}".
                      format(args.task_index, g_step, loss_out, avg_prob_right_out))


if __name__ == "__main__":
    main()
