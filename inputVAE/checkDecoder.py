import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

#%% Variables
SIZE = 256
latent_dim = 9
path_name = 'path_input_9'
name_Z = 'Zinput_9.npy'
name_Z_test = 'Zinput_9_t.npy'
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
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.load_weights(path_name)

#%% Training
# img_train = np.load('inp_img_train.npy')
# Z_train = np.load(name_Z)
# img_pred = vae.decoder.predict(Z_train)

#%% Testing
img_test = np.load('inp_img_test.npy')
Z_enc, _, _, = vae.encoder.predict(img_test)
np.save(name_Z_test, Z_enc)
img_pred = vae.decoder.predict(Z_enc)

#%%
import matplotlib.pyplot as plt
from cv2 import split
import matplotlib as mpl
font = {'family' : 'Liberation Serif',
        'weight' : 'normal',
        'size'   : 10}
cm=1/2.54
mpl.rc('font', **font)
mpl.rc('axes', linewidth=1)
mpl.rc('lines', lw=1)

# fig = plt.figure(1, figsize=(12*cm, 6*cm))
# ax = plt.subplot(121)
# ax.imshow(img_train[15])
# ax = plt.subplot(122)
# ax.imshow(img_pred[15])
# fig.tight_layout()
# plt.show()

MAE_O = np.array([])
MAE_B = np.array([])
MAE_I = np.array([])
npx = SIZE*SIZE
for i in range(150):
    O, B, I = split(img_test[i])
    Op, Bp, Ip = split(img_pred[i])
    MAE_O = np.append(MAE_O, np.sum(abs(Op-O))/npx)
    MAE_B = np.append(MAE_B, np.sum(abs(Bp-B))/npx)
    MAE_I = np.append(MAE_I, np.sum(abs(Ip-I))/npx)

# np.save('MAE_O_'+str(latent_dim)+'train.npy', MAE_O)
# np.save('MAE_B_'+str(latent_dim)+'train.npy', MAE_B)
# np.save('MAE_I_'+str(latent_dim)+'train.npy', MAE_I)

#%%
from mpl_toolkits.axes_grid1 import make_axes_locatable

O, B, I = split(img_test[111])
Op, Bp, Ip = split(img_pred[111])

fig = plt.figure(3, figsize=(18*cm, 18*cm))
ax = plt.subplot(3,3,1)
ax.imshow(O, cmap='bone')
ax.set_ylabel('Original')
ax.set_title('Inlet')
ax = plt.subplot(3,3,2)
ax.imshow(B, cmap='bone')
ax.set_title('Volume control')
ax = plt.subplot(3,3,3)
ax.imshow(I, cmap='bone')
ax.set_title('Outlets')
ax = plt.subplot(3,3,4)
ax.imshow(Op, cmap='bone')
ax.set_ylabel('Reconstruction')
ax = plt.subplot(3,3,5)
ax.imshow(Bp, cmap='bone')
ax = plt.subplot(3,3,6)
ax.imshow(Ip, cmap='bone')
ax = plt.subplot(3,3,7)
im = ax.imshow(abs(Op-O), cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_ylabel('Absolute error')
ax = plt.subplot(3,3,8)
im = ax.imshow(abs(Bp-B), cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax = plt.subplot(3,3,9)
im = ax.imshow(abs(Ip-I), cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
fig.tight_layout()
plt.show()