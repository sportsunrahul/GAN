import keras
from keras import layers
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import time
import os
from PIL import Image


###################################################################
### READING IMAGES
###################################################################
start_time = time.time()
folder = 'C:/Users/sport/Desktop/Practice/Danbooru/tagged-anime-illustrations/moeimouto-faces/'
# folder = 'C:/Users/sport/Desktop/temp'

x1 = []
for file in glob.glob('{}/**/**/*png'.format(folder)):
    # images.append(cv2.imread(file))
    images=np.asarray(Image.open(file).resize((64,64)))
    x1.append(images)

x_train = np.array(x1)
print("Shape of x_train",x_train.shape)
print("Time taken to read the whole dataset",time.time()-start_time, "s")
# x_train = x_train/255

latent_dim = x_train[0].shape[0]
height,width,channels = x_train[0].shape


###################################################################
### GENERATOR
###################################################################
generator_input = keras.Input(shape=(latent_dim,))

# First, transform the input into a 16x16 128-channels feature map
x = layers.Dense(128 * 32 * 32)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((32, 32, 128))(x)

# Then, add a convolution layer
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Upsample to 32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

# Few more conv layers
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Produce a 32x32 1-channel feature map
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

###################################################################
### DISCRIMINATOR
###################################################################
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# One dropout layer - important trick!
x = layers.Dropout(0.4)(x)

# Classification layer
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')


###################################################################
### GAN
###################################################################
# Set discriminator weights to non-trainable
# (will only apply to the `gan` model)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')



###################################################################
### TRAINING
###################################################################
import os
from keras.preprocessing import image

# Normalize data

x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.
print("Shape After Reshaping",x_train.shape)


iterations = 10000
batch_size = 20
save_dir = 'C:/Users/sport/Google Drive/First Semester/Topics in DS/FinalProject/26.11/gan_images_64x64/'
label1 = {"l1":"Discriminator Loss","l2":"Adversarial Loss"}


d_loss = []
a_loss = []
# Start training loop
start = 0
start_time = time.time()
for step in range(iterations):



    # Sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Decode them to fake images
    generated_images = generator.predict(random_latent_vectors)

    # Combine them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    # Assemble labels discriminating real from fake images
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    # Add random noise to the labels - important trick!
    labels += 0.05 * np.random.random(labels.shape)

    # Train the discriminator
    d_loss.append(discriminator.train_on_batch(combined_images, labels))

    # sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Assemble labels that say "all real images"
    misleading_targets = np.zeros((batch_size, 1))

    # Train the generator (via the gan model,
    # where the discriminator weights are frozen)
    a_loss.append(gan.train_on_batch(random_latent_vectors, misleading_targets))
    
    start += batch_size
    if start > len(x_train) - batch_size:
	    start = 0

    # Occasionally save / plot
    if step % 100 == 0:
        # Save model weights
        gan.save_weights('gan.h5')

        # Print metrics
        # print('discriminator loss at step %s: %s' % (step, d_loss))
        # print('adversarial loss at step %s: %s' % (step, a_loss))

        plt.plot(d_loss,'k',label = label1["l1"])
        plt.plot(a_loss,'r',label = label1["l2"])
        plt.xlabel("Iteration")
        plt.ylabel("Loss (%)")
        plt.legend()
        plt.savefig(save_dir +"Loss-Plot{}".format(step))
        label1["l1"] = "__nolegend__"
        label1["l2"] = "__nolegend__"

        # Save one generated image
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_faces' + str(step) + '.png'))

        # Save one real image, for comparison
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_faces' + str(step) + '.png'))
        print(step," step: ", ((time.time()-start_time)),"seconds")
        start_time = time.time()

