from keras_resnet.models import ResNet50
from keras import layers, models
import tensorflow as tf
from losses import db_loss


def dbnet(input_size=640, k=50):
    image_input = layers.Input(shape=(None, None, 3))
    gt_input = layers.Input(shape=(input_size, input_size))
    mask_input = layers.Input(shape=(input_size, input_size))
    thresh_input = layers.Input(shape=(input_size, input_size))
    thresh_mask_input = layers.Input(shape=(input_size, input_size))
    backbone = ResNet50(inputs=image_input, include_top=False, freeze_bn=True)
    C2, C3, C4, C5 = backbone.outputs
    in2 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
    in3 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
    in4 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
    in5 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)

    # 1 / 32 * 8 = 1 / 4
    P5 = layers.UpSampling2D(size=(8, 8))(
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5))
    # 1 / 16 * 4 = 1 / 4
    out4 = layers.Add()([in4, layers.UpSampling2D(size=(2, 2))(in5)])
    P4 = layers.UpSampling2D(size=(4, 4))(
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4))
    # 1 / 8 * 2 = 1 / 4
    out3 = layers.Add()([in3, layers.UpSampling2D(size=(2, 2))(out4)])
    P3 = layers.UpSampling2D(size=(2, 2))(
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3))
    # 1 / 4
    P2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(
        layers.Add()([in2, layers.UpSampling2D(size=(2, 2))(out3)]))
    # (b, /4, /4, 256)
    fuse = layers.Concatenate()([P2, P3, P4, P5])

    # probability map
    p = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                               activation='sigmoid')(p)

    # threshold map
    t = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(t)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                               activation='sigmoid')(t)

    # approximate binary map
    b_hat = layers.Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([p, t])

    loss = layers.Lambda(db_loss, name='db_loss')([p, b_hat, gt_input, mask_input, t, thresh_input, thresh_mask_input])
    training_model = models.Model(inputs=[image_input, gt_input, mask_input, thresh_input, thresh_mask_input],
                                  outputs=loss)
    prediction_model = models.Model(inputs=image_input, outputs=p)
    return training_model, prediction_model


if __name__ == '__main__':
    model, _ = dbnet()
    # model.summary()
