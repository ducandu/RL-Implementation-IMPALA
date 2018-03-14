import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=str,
                        help="The hosts to run on (comma separated).")
    parser.add_argument("-t", "--task", type=int, default=0,
                        help="The index for the task running on this client.")
    args = parser.parse_args()

    hosts = args.hosts.split(",")
    cluster = tf.train.ClusterSpec({"local": hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name="local", task_index=args.task)

    # simple test: one task creates variable on one server (shape=A)
    # the other task creates same variable (same name and same server) with another shape (shape=B) -> should fail
    with tf.device("/job:local/task:0"):
        # this will work all fine (same var-name, same shape)
        var_ = tf.Variable([1.0], dtype=tf.float16, name="test-var")

        # this will not work as chief will not initialize test-var-of-task-1 (it doesn't have it!)
        # var_ = tf.Variable([1.0], dtype=tf.float16, name="test-var-of-task-{}".format(args.task))

        # this will not work (chief will have problem assigning value to `task-var` of task 1 as it has different shape)
        # var_ = tf.Variable([1.0] if args.task == 0 else [2.0, 1.0], dtype=tf.float16, name="test-var")

        # Has to sit on one task (same task for all clients; otherwise won't be global)!
        global_step = tf.train.get_or_create_global_step()
        global_step_incr = tf.assign_add(global_step, 1)  # fake global_step incrementor

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    step_counter_hook = tf.train.StepCounterHook(every_n_steps=100)
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(args.task == 0),
                                           #checkpoint_dir="/tmp/train_logs",
                                           hooks=[step_counter_hook]) as mon_sess:
        while not mon_sess.should_stop():
            vo, g_step = mon_sess.run([var_, global_step_incr])
            print("task {} vo={} g_step={}".format(args.task, vo, g_step))

    server.join()


if __name__ == "__main__":
    main()

