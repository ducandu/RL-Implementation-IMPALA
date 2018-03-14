import tensorflow as tf


def main():
    # simple test: create a distribution from a placeholder input
    # then sample from the distribution

    # the input (n probabilities to select: True)
    input_probs = tf.placeholder(dtype=tf.float32, shape=(None,))
    num_samples = tf.placeholder(dtype=tf.int32, shape=())
    distr = tf.distributions.Bernoulli(probs=input_probs)
    out = distr.sample(sample_shape=num_samples)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out, feed_dict={input_probs: [0.5, 0.25, 0.1, 0.9], num_samples: 10})
        print("Result is {}".format(res))
        res = sess.run(out, feed_dict={input_probs: [0.5, 0.25, 0.1, 0.9], num_samples: 5})
        print("Result is {}".format(res))


if __name__ == "__main__":
    main()

