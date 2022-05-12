import tensorflow as tf
from tensorflow.python.keras import layers, models, losses, optimizers, regularizers


class ResnetBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, strides, l2=0.01):
        super(ResnetBlock, self).__init__(name='')
        filters1, filters2 = filters
        strides1, strides2 = strides

        self.conv2a = layers.Conv2D(filters1, kernel_size, padding='same', strides=strides1, kernel_regularizer=regularizers.l2(l2=l2))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', strides=strides2, kernel_regularizer=regularizers.l2(l2=l2))
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.shortcut = models.Sequential()
        if strides1 != 1:
            self.shortcut = models.Sequential([
                layers.Conv2D(filters1, kernel_size=1, strides=strides1, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        x += self.shortcut(input_tensor)
        return tf.nn.relu(x)


class SmallResNet(models.Sequential):
    def __init__(self, n=3, l2=0, **kwargs):
        super(SmallResNet, self).__init__(**kwargs)
        assert (n >= 3)
        self._build_model(n, l2)
        self.n_layers = None

    def _build_model(self, n=3, l2=0.01):
        self.add(layers.Conv2D(16, (3, 3), strides=1, padding='same'))

        self.n_layers = 6*n + 2

        # feature map (32x32)
        for i in range(2*n):
            self.add(ResnetBlock(kernel_size=(3, 3), filters=[16, 16], strides=[1, 1], l2=l2))

        # feature map (16x16)
        for i in range(2*n):
            if i == 0:
                self.add(ResnetBlock(kernel_size=(3, 3), filters=[32, 32], strides=[2, 1], l2=l2))
            else:
                self.add(ResnetBlock(kernel_size=(3, 3), filters=[32, 32], strides=[1, 1], l2=l2))

        # feature map (8x8)
        for i in range(2*n):
            if i == 0:
                self.add(ResnetBlock(kernel_size=(3, 3), filters=[64, 64], strides=[2, 1], l2=l2))
            else:
                self.add(ResnetBlock(kernel_size=(3, 3), filters=[64, 64], strides=[1, 1], l2=l2))

        self.add(layers.GlobalAveragePooling2D())
        self.add(layers.Dense(10))
        self.add(layers.Softmax())
        
        
class DenseNet(models.Sequential):
    def __init__(self, dropout=0, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.dropout = dropout
        self._build_model()

    def _build_model(self):
        self.add(layers.Flatten())
        self.add(layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
        self.add(layers.Dropout(self.dropout))
        self.add(layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
        self.add(layers.Dropout(self.dropout))
        self.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.add(layers.Dropout(self.dropout))
        self.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.add(layers.Dropout(self.dropout))
        self.add(layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
        self.add(layers.Dropout(self.dropout))
        self.add(layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
        self.add(layers.Dropout(self.dropout))
        self.add(layers.Dense(10, activation='softmax'))


class SimpleCNN(models.Sequential):
    def __init__(self, dropout=0, **kwargs):
        super(SimpleCNN, self).__init__(**kwargs)
        self.dropout = dropout
        self._build_model()

    def _build_model(self):
        self.add(layers.Conv2D(16, kernel_size=3, kernel_initializer='he_uniform', activation='relu'))
        self.add(layers.Conv2D(32, kernel_size=3, kernel_initializer='he_uniform', activation='relu'))
        self.add(layers.Conv2D(64, kernel_size=3, kernel_initializer='he_uniform', activation='relu'))
        self.add(layers.Flatten())
        self.add(layers.Dropout(self.dropout))
        self.add(layers.Dense(256, activation='relu'))
        self.add(layers.Dropout(self.dropout))
        self.add(layers.Dense(256, activation='relu'))
        self.add(layers.Dropout(self.dropout))
        self.add(layers.Dense(10, activation='sigmoid'))
