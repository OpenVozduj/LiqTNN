import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

#%% Variables
SIZE = 256
latent_dim = 10
path_name = 'path_input_10'
name_Z = 'Zinput_10.npy'
name_fig = 'loss_converg_10.svg'
Epochs = 300
Batch_size = 20
act_funtion = 'relu'
# act_funtion = layers.LeakyReLU()
#%% Create a sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
#%% Build the encoder
encoder_inputs = keras.Input(shape=(SIZE, SIZE, 3))
x = layers.Conv2D(8, 3, activation=act_funtion, padding="same")(encoder_inputs)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, 3, activation=act_funtion, padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, 3, activation=act_funtion, padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, 3, activation=act_funtion, padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(128, 3, activation=act_funtion, padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(256, 3, activation=act_funtion, padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, activation='sigmoid', name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, activation='linear', name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

#%% Build the decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(4 * 4 * 256, activation=act_funtion)(latent_inputs)
x = layers.Reshape((4, 4, 256))(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(128, 3, activation=act_funtion, padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(64, 3, activation=act_funtion, padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, 3, activation=act_funtion, padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(16, 3, activation=act_funtion, padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(8, 3, activation=act_funtion, padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

#%% Define the VAE as a Model with a costum train_step
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

#%%
# img_array = np.load('inp_img_train.npy')

# #%%
# initial_lr = 0.001
# decaySteps = 50*(900/20)
# lr_schedule = ExponentialDecay(initial_lr, decaySteps, decay_rate=0.8)

# #%%
# vae = VAE(encoder, decoder)
# vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
# history = vae.fit(img_array, 
#                   epochs=Epochs, 
#                   batch_size=Batch_size)

# #%%
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# font = {'family' : 'Liberation Serif',
#         'weight' : 'normal',
#         'size'   : 10}
# cm=1/2.54
# mpl.rc('font', **font)
# mpl.rc('axes', linewidth=1)
# mpl.rc('lines', lw=1)

# fig = plt.figure(1, figsize=(10*cm, 8*cm))
# ax = plt.subplot(111)
# ax.plot(history.history['loss'], label='Training')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss function')
# ax.set_title('z = '+str(latent_dim))
# ax.legend(loc='best')
# ax.grid()
# fig.tight_layout()
# fig.savefig(name_fig)

# #%%
# vae.save_weights(path_name)
# config = vae.get_config()
# weights = vae.get_weights()

# #%% Extract z vector
# # img_test = np.load('images_set_test.npy')
# Z_main, Z_std, Z_train = vae.encoder.predict(img_array)
# np.save(name_Z, Z_main)
# # img_decoded = vae.decoder.predict(Z_main)
