import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import backend as K
import numpy as np


class VariationalNN:
	def __init__(
		self,
		lr=0.001, 
		dropout=0.5, 
		out_dim=2, 
		dropout_flag=False,
		gru_n_hidden=100,
		cnn_n_filter=64,
		cnn_n_kernel=3):

		self.lr = lr
		self.dropout = dropout
		self.out_dim = out_dim
		self.dropout_flag = dropout_flag
		self.gru_n_hidden = gru_n_hidden
		self.cnn_n_filter = cnn_n_filter
		self.cnn_n_kernel = cnn_n_kernel


	def create_model(
		self,
		nn_type, #['cnn', 'gru', 'gru_pv']
		time_length,
		n_feat):

		input_shape = (time_length,n_feat)
		input_tensor = tf.keras.Input(shape=input_shape,name='input')

		if nn_type == 'cnn':

			logits = tf.keras.layers.Conv1D(
				self.cnn_n_filter, 
				kernel_size=self.cnn_n_kernel, 
				activation=tf.nn.relu)(input_tensor)
			logits = tf.keras.layers.BatchNormalization()(logits)
			logits = tf.keras.layers.Dropout(self.dropout)(logits)

			logits = tfp.layers.Convolution1DReparameterization(
				self.cnn_n_filter, 
				kernel_size=self.cnn_n_kernel, 
				activation=tf.nn.relu,
				name='un_layer')(logits)
			logits = tf.keras.layers.BatchNormalization()(logits)
			if self.dropout_flag:
			    logits = tf.keras.layers.Dropout(self.dropout)(logits,training=True)
			else:
			    logits = tf.keras.layers.Dropout(self.dropout)(logits)
			logits = tf.keras.layers.Flatten()(logits)

		elif nn_type == 'gru':

			gru_mu = tf.keras.layers.GRU(
				self.gru_n_hidden,
				activation='tanh', 
				recurrent_activation='sigmoid', 
				use_bias=True,
				kernel_initializer='glorot_uniform', 
				recurrent_initializer='orthogonal', 
				bias_initializer='zeros',
				return_sequences=False)(input_tensor)
			gru_sigma = tf.keras.layers.GRU(
				self.gru_n_hidden,
				activation='tanh', 
				recurrent_activation='sigmoid', 
				use_bias=True,
				kernel_initializer='glorot_uniform', 
				recurrent_initializer='orthogonal', 
				bias_initializer='zeros',
				return_sequences=False)(input_tensor)
			gru_h = tf.keras.layers.Lambda(lambda x: x[0]+tf.random.normal(mean=0,
				stddev=K.abs(x[1]),shape=[1,]),name='un_layer')([gru_mu,gru_sigma])
			if self.dropout_flag:
				logits = tf.keras.layers.Dropout(self.dropout)(gru_h,training=True)
			else:
				logits = tf.keras.layers.Dropout(self.dropout)(gru_h)

		elif nn_type == 'gru_pv':

			gru_mu = tf.keras.layers.GRU(
				self.gru_n_hidden,
				activation='tanh', 
				recurrent_activation='sigmoid', 
				use_bias=True,
				kernel_initializer='glorot_uniform', 
				recurrent_initializer='orthogonal', 
				bias_initializer='zeros',
				return_sequences=True)(input_tensor)
			gru_sigma = tf.keras.layers.GRU(
				self.gru_n_hidden,
				activation='tanh', 
				recurrent_activation='sigmoid', 
				use_bias=True,
				kernel_initializer='glorot_uniform', 
				recurrent_initializer='orthogonal', 
				bias_initializer='zeros',
				return_sequences=True)(input_tensor)
			gru_h = tf.keras.layers.Lambda(lambda x: x[0]+tf.random.normal(mean=0,
				stddev=K.abs(x[1]),shape=[1,]),name='un_layer')([gru_mu,gru_sigma])
			gru_out = tf.keras.layers.Lambda(lambda x: x[:,-1,:])(gru_h)
			gru_out = tf.keras.layers.Reshape((self.gru_n_hidden,))(gru_out)
			max_gru = tf.keras.layers.MaxPooling1D(time_length)(gru_h)
			max_gru = tf.keras.layers.Reshape((self.gru_n_hidden,))(max_gru)
			avg_gru = tf.keras.layers.Lambda(lambda x: K.mean(x,axis=-2))(gru_h)
			avg_gru = tf.keras.layers.Reshape((self.gru_n_hidden,))(avg_gru)
			min_gru = tf.keras.layers.Lambda(lambda x: -x)(gru_h)
			min_gru = tf.keras.layers.MaxPooling1D(time_length)(min_gru)
			min_gru = tf.keras.layers.Lambda(lambda x: -x)(min_gru)
			min_gru = tf.keras.layers.Reshape((self.gru_n_hidden,))(min_gru)
			gru_h = tf.keras.layers.concatenate([gru_out,max_gru,avg_gru,min_gru])
			if self.dropout_flag:
				logits = tf.keras.layers.Dropout(self.dropout)(gru_h,training=True)
			else:
				logits = tf.keras.layers.Dropout(self.dropout)(gru_h)

		logits = tf.keras.layers.Dense(100, activation=tf.nn.relu)(logits)
		logits = tf.keras.layers.Dense(self.out_dim,activation=tf.nn.softmax)(logits)
		model = tf.keras.Model(inputs=input_tensor, outputs=logits)

		def variational_loss(y_true, y_pred):
		    neg_log_likelihood = tf.keras.losses.categorical_crossentropy(y_true,y_pred)
		    kl = sum(model.losses)
		    loss = neg_log_likelihood + 0.5*kl
		    return loss

		model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
			loss=variational_loss)

		return model

# heteroscedastic NN
class HNN:
	def __init__(
		self,
		lr=0.001, 
		dropout=0.5, 
		out_dim=2, 
		dropout_flag=False,
		gru_n_hidden=100,
		cnn_n_filter=64,
		cnn_n_kernel=3):

		self.lr = lr
		self.dropout = dropout
		self.out_dim = out_dim
		self.dropout_flag = dropout_flag
		self.gru_n_hidden = gru_n_hidden
		self.cnn_n_filter = cnn_n_filter
		self.cnn_n_kernel = cnn_n_kernel




	# Bayesian categorical cross entropy.
	# N data points, C classes, T monte carlo simulations
	# true - true values. Shape: (N, C)
	# pred_var - predicted logit values and variance. Shape: (N, C + 1)
	# returns - loss (N,)
	def bayesian_categorical_crossentropy(self, T, num_classes):
	  def bayesian_categorical_crossentropy_internal(true, pred_var):
	    # shape: (N,)
	    std = K.sqrt(pred_var[:, num_classes:])
	    # shape: (N,)
	    variance = pred_var[:, num_classes]
	    variance_depressor = -K.exp(-variance) #- K.ones_like(variance)
	    # shape: (N, C)
	    pred = pred_var[:, 0:num_classes]
	    # shape: (N,)
	    undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
	    # shape: (T,)
	    iterable = K.variable(np.ones(T))
	    dist = tfd.Normal(loc=K.zeros_like(std), scale=std)
	    monte_carlo_results = K.map_fn(self.gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes), iterable, name='monte_carlo_results')
	    
	    variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss
	    return variance_loss + 0.5*variance_depressor + undistorted_loss
	  
	  return bayesian_categorical_crossentropy_internal

	# for a single monte carlo simulation, 
	#   calculate categorical_crossentropy of 
	#   predicted logit values plus gaussian 
	#   noise vs true values.
	# true - true values. Shape: (N, C)
	# pred - predicted logit values. Shape: (N, C)
	# dist - normal distribution to sample from. Shape: (N, C)
	# undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
	# num_classes - the number of classes. C
	# returns - total differences for all classes (N,)
	def gaussian_categorical_crossentropy(self, true, pred, dist, undistorted_loss, num_classes):
	  def map_fn(i):
	    std_samples = K.transpose(dist.sample(num_classes))
	    distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
	    return distorted_loss
	    diff = undistorted_loss - distorted_loss
	    return -K.elu(diff)
	  return map_fn

	def create_model(
		self,
		nn_type, #['cnn', 'gru', 'gru_pv']
		time_length,
		n_feat):

		input_shape = (time_length,n_feat)
		input_tensor = tf.keras.Input(shape=input_shape,name='input')

		if nn_type == 'cnn':

			logits = tf.keras.layers.Conv1D(
				self.cnn_n_filter, 
				kernel_size=self.cnn_n_kernel, 
				activation=tf.nn.relu)(input_tensor)
			logits = tf.keras.layers.BatchNormalization()(logits)
			logits = tf.keras.layers.Dropout(self.dropout)(logits)

			logits = tf.keras.layers.Conv1D(
				self.cnn_n_filter, 
				kernel_size=self.cnn_n_kernel, 
				activation=tf.nn.relu,
				name='un_layer')(logits)
			logits = tf.keras.layers.BatchNormalization()(logits)
			if self.dropout_flag:
			    logits = tf.keras.layers.Dropout(self.dropout)(logits,training=True)
			else:
			    logits = tf.keras.layers.Dropout(self.dropout)(logits)
			logits = tf.keras.layers.Flatten()(logits)

		elif nn_type == 'gru':

			gru_h = tf.keras.layers.GRU(
				self.gru_n_hidden,
				activation='tanh', 
				recurrent_activation='sigmoid', 
				use_bias=True,
				kernel_initializer='glorot_uniform', 
				recurrent_initializer='orthogonal', 
				bias_initializer='zeros',
				return_sequences=False)(input_tensor)
			if self.dropout_flag:
				logits = tf.keras.layers.Dropout(self.dropout)(gru_h,training=True)
			else:
				logits = tf.keras.layers.Dropout(self.dropout)(gru_h)

		elif nn_type == 'gru_pv':

			gru_h = tf.keras.layers.GRU(
				self.gru_n_hidden,
				activation='tanh', 
				recurrent_activation='sigmoid', 
				use_bias=True,
				kernel_initializer='glorot_uniform', 
				recurrent_initializer='orthogonal', 
				bias_initializer='zeros',
				return_sequences=True)(input_tensor)
			gru_out = tf.keras.layers.Lambda(lambda x: x[:,-1,:])(gru_h)
			gru_out = tf.keras.layers.Reshape((self.gru_n_hidden,))(gru_out)
			max_gru = tf.keras.layers.MaxPooling1D(time_length)(gru_h)
			max_gru = tf.keras.layers.Reshape((self.gru_n_hidden,))(max_gru)
			avg_gru = tf.keras.layers.Lambda(lambda x: K.mean(x,axis=-2))(gru_h)
			avg_gru = tf.keras.layers.Reshape((self.gru_n_hidden,))(avg_gru)
			min_gru = tf.keras.layers.Lambda(lambda x: -x)(gru_h)
			min_gru = tf.keras.layers.MaxPooling1D(time_length)(min_gru)
			min_gru = tf.keras.layers.Lambda(lambda x: -x)(min_gru)
			min_gru = tf.keras.layers.Reshape((self.gru_n_hidden,))(min_gru)
			gru_h = tf.keras.layers.concatenate([gru_out,max_gru,avg_gru,min_gru])
			if self.dropout_flag:
				logits = tf.keras.layers.Dropout(self.dropout)(gru_h,training=True)
			else:
				logits = tf.keras.layers.Dropout(self.dropout)(gru_h)

		logits = tf.keras.layers.Dense(100, activation=tf.nn.relu)(logits)
		variance = tf.keras.layers.Dense(1, activation='softplus',name='variance')(logits)
		logits = tf.keras.layers.Dense(2)(logits)
		softmax_output = tf.keras.layers.Activation(activation=tf.nn.softmax,name='softmax_output')(logits)
		logits_variance = tf.keras.layers.concatenate([logits, variance], name='logits_variance')
		model = tf.keras.Model(inputs=input_tensor, outputs=[logits_variance,softmax_output])


		model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3,decay=0.001),
			loss={
			'logits_variance': self.bayesian_categorical_crossentropy(100, 2),
			'softmax_output': 'categorical_crossentropy'
			},
			metrics={'softmax_output': tf.keras.metrics.categorical_accuracy},
			loss_weights={'logits_variance': .5, 'softmax_output': 1.}
			)

		return model