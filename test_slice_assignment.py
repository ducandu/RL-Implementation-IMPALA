import tensorflow as tf


def main():
    # simple test: assign a tensor to a slice of another tensor
    # - then run some op depending on that assignment (control_dependency)

    # the "buffer" to assign a slice to (matrix of 3x3)
    buffer = tf.Variable(tf.zeros(shape=(2, 2)))

    # the slice (one row of a 3x3 matrix)
    slice_ = tf.Variable(tf.ones(shape=(2,)))

    # the slice assignment op (inserts slice_ as the second row into the buffer)
    insert_row = tf.assign(buffer[1,:], slice_)

    # need to add the control dependency here
    # - otherwise, the insert_row op would not be executed when calling sess.run(out)
    #   because out is not directly dependent on the 'calculation' of the assignment
    with tf.control_dependencies([insert_row]):
        out = tf.identity(buffer, name="identity")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out)
    print("Result is {}".format(res))


if __name__ == "__main__":
    main()

