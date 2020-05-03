import tensorflow as tf
"""
This spectral_norm implementation was taken from https://github.com/taki0112/Spectral_Normalization-Tensorflow
"""
def spectral_norm(w, iteration=1):
	w_shape = w.shape.as_list()
	w = tf.reshape(w, [-1, w_shape[-1]])

	u = tf.Variable(tf.random.truncated_normal(shape=[1, w_shape[-1]], \
		stddev=.1, dtype=tf.float32), trainable=False, name="u")

	u_hat = u
	v_hat = None
	for i in range(iteration):
		"""
		power iteration
		Usually iteration = 1 will be enough
		"""
		v_ = tf.matmul(u_hat, tf.transpose(w))
		v_hat = tf.nn.l2_normalize(v_)

		u_ = tf.matmul(v_hat, w)
		u_hat = tf.nn.l2_normalize(u_)

	u_hat = tf.stop_gradient(u_hat)
	v_hat = tf.stop_gradient(v_hat)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

	with tf.control_dependencies([u.assign(u_hat)]):
		w_norm = tf.math.divide(w, sigma)
		w_norm = tf.reshape(w_norm, w_shape)


	return w_norm

def spectral_conv(inputs, weight, stride, bias=None): 
	x = tf.nn.conv2d(input=inputs, filters=spectral_norm(weight), strides=stride, padding="SAME")
	if bias != None: 
		x = tf.nn.bias_add(x, bias)
	return x
	