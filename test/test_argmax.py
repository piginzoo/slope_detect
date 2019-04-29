import tensorflow as tf
# data = tf.constant([[0.2, 0.2, 0.4, 0.2],[0.3, 0.5, 0.1, 0.1]])
data = tf.constant([[0.2, 0.2, 0.4, 0.2]])
print(tf.shape(data))
a = tf.argmax(data,axis=1)
with tf.Session() as s:
    b = s.run(a)
    print(b.shape)
    print(b)