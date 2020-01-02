import datetime
import os.path as osp
from keras import callbacks
from keras import optimizers
from keras.utils import get_file
import os

from generator import generate
from model import dbnet
checkpoints_dir = f'checkpoints/{datetime.date.today()}'

batch_size = 16

if not osp.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

train_generator = generate('datasets/total_text', batch_size=batch_size, is_training=True)
val_generator = generate('datasets/total_text', batch_size=batch_size, is_training=False)

model, prediction_model = dbnet()
resnet_filename = 'ResNet-50-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
resnet_filepath = get_file(resnet_filename, resnet_resource, cache_subdir='models',
                           md5_hash='3e9f4e4f77bbe2c9bec13b53ee1c2319')
model.load_weights(resnet_filepath, by_name=True, skip_mismatch=True)
model.compile(optimizer=optimizers.Adam(lr=1e-3), loss={'db_loss': lambda y_true, y_pred: y_pred})
checkpoint = callbacks.ModelCheckpoint(
    osp.join(checkpoints_dir, 'db_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'),
    verbose=1,
)
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=200,
    initial_epoch=0,
    epochs=100,
    verbose=1,
    callbacks=[checkpoint],
    validation_data=val_generator,
    validation_steps=19
)

