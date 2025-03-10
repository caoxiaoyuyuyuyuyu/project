import os
import numpy as np
from datetime import datetime
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, UpSampling2D, Dense, LeakyReLU, Lambda
from keras.optimizers import Adam
from keras.applications import VGG19
import tensorflow as tf
from data_loader import DataLoader
import matplotlib
matplotlib.use('Agg')  # 强制使用 Agg 后端
import matplotlib.pyplot as plt

class SRGAN:
    def __init__(self):
        # Input shape
        self.channels = 3
        self.lr_height = 64 # Low resolution height
        self.lr_width = 64  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * 4  # High resolution height
        self.hr_width = self.lr_width * 4   # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = 'images'
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)
        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=optimizer)

        # 创建保存模型的目录
        self.save_dir = 'saved_model'
        os.makedirs(self.save_dir, exist_ok=True)


    def build_vgg(self):
            """
            Builds a pre-trained VGG19 model that outputs image features extracted at the
            third block of the model
            """
            vgg = VGG19(weights="imagenet")
            vgg.trainable = False
            # Set outputs to outputs of last conv. layer in block 3
            # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
            output_layer = [vgg.layers[9].output]
            vgg_model = Model(inputs=vgg.input, outputs=output_layer)

            img = Input(shape=self.hr_shape)
            # change the shape of image
            resized_input = Lambda(lambda x: tf.image.resize(x, (224, 224)))(img)
            # Extract image features
            img_features = vgg_model(resized_input)

            return Model(img, img_features)

    def build_generator(self):
        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):
        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)

        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.now()
        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(self.generator.predict(imgs_lr), fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, self.vgg.predict(imgs_hr)])

            elapsed_time = datetime.now() - start_time
            print(f"Epoch {epoch}/{epochs} | D loss: {d_loss[0]:.4f}, G loss: {g_loss[0]:.4f} | Time: {elapsed_time}")

            # Save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            # Save the models
            if epoch % 10 == 0:
                self.generator.save(os.path.join(self.save_dir, f'generator_{epoch}.h5'))
                self.discriminator.save(os.path.join(self.save_dir, f'discriminator_{epoch}.h5'))


    def sample_images(self, epoch):
            os.makedirs(f'images/{self.dataset_name}', exist_ok=True)
            r, c = 1, 1

            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
            fake_hr = self.generator.predict(imgs_lr)

            # Rescale images 0 - 1
            imgs_lr = 0.5 * imgs_lr + 0.5
            fake_hr = 0.5 * fake_hr + 0.5
            imgs_hr = 0.5 * imgs_hr + 0.5

            # 单独保存生成的图片 (fake_hr)
            for i, image in enumerate(fake_hr):
                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')  # 关闭坐标轴
                fig.savefig('images/%s/%d_generated%d.png' % (self.dataset_name, epoch, i))
                plt.close()

            # 保存原始高分辨率图片 (imgs_hr)
            for i, image in enumerate(imgs_hr):
                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')  # 关闭坐标轴
                fig.savefig('images/%s/%d_original%d.png' % (self.dataset_name, epoch, i))
                plt.close()

            # 保存低分辨率图片 (imgs_lr)
            for i, image in enumerate(imgs_lr):
                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')  # 关闭坐标轴
                fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
                plt.close()


if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=100, batch_size=1, sample_interval=50)