import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# define computational graphs
x = tf.compat.v1.placeholder(tf.float32, name='x')
y = tf.compat.v1.placeholder(tf.float32, name='y')

addition = tf.add(x,y, name='addition')

# create a session
with tf.compat.v1.Session() as session:
    result = session.run(addition, feed_dict={x: [1, 2, 3], y: [2, 3, 4]})
    print(result)
