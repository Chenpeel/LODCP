from tensorflow.keras import layers, models

def build_tiny_unet(input_shape=(256, 512, 3)):  # 匹配预处理尺寸 (height, width, channels)
    inputs = layers.Input(shape=input_shape)

    # 编码器
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPool2D(2)(x)  # 128x256

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2)(x)  # 64x128

    # 中间层
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # 解码器
    x = layers.UpSampling2D(2)(x)  # 128x256
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(2)(x)  # 256x512
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)

    # 输出层
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)  # 二值分割输出

    return models.Model(inputs, outputs, name='TinyUNet')
