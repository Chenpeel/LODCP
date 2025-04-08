import tensorflow as tf
from .tiny_unet import build_tiny_unet
from .npz_dataset import NpzDataset
from datetime import datetime
import os

# 配置参数
config = {
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 1e-4,
    'log_dir': 'logs',
    'model_save_path': 'models/unet_model.h5'
}

# 初始化数据集
train_dataset = NpzDataset(
    npz_dir=config['train_dir'],
    batch_size=config['batch_size']
)

val_dataset = NpzDataset(
    npz_dir=config['val_dir'],
    batch_size=config['batch_size']
)

# 构建模型
model = build_tiny_unet(input_shape=(256, 512, 3))
model.summary()  # 打印模型结构

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)

# 回调函数
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        config['model_save_path'],
        save_best_only=True,
        monitor='val_loss'
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S")),
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True)
]

# 训练模型
history = model.fit(
    train_dataset,
    epochs=config['epochs'],
    steps_per_epoch=train_dataset.steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_dataset.steps_per_epoch,
    callbacks=callbacks
)

# 保存最终模型
model.save(config['model_save_path'])
print("训练完成，模型已保存至:", config['model_save_path'])import tensorflow as tf
from .tiny_unet import build_tiny_unet
from .npz_dataset import NpzDataset
from datetime import datetime
import os

# 配置参数
config = {
    'train_dir': 'data/processed_data',  # 训练数据路径
    'val_dir': 'data/val_processed_data',  # 验证数据路径
    'batch_size': 8,
    'epochs': 30,
    'learning_rate': 1e-4,
    'log_dir': 'logs',  # TensorBoard日志目录
    'model_save_path': 'saved_models/unet_model.h5'
}

# 初始化数据集
train_dataset = NpzDataset(
    npz_dir=config['train_dir'],
    batch_size=config['batch_size']
)

val_dataset = NpzDataset(
    npz_dir=config['val_dir'],
    batch_size=config['batch_size']
)

# 构建模型
model = build_tiny_unet(input_shape=(256, 512, 3))
model.summary()  # 打印模型结构

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)

# 回调函数
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        config['model_save_path'],
        save_best_only=True,
        monitor='val_loss'
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S")),
    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True)
]

# 训练模型
history = model.fit(
    train_dataset,
    epochs=config['epochs'],
    steps_per_epoch=train_dataset.steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_dataset.steps_per_epoch,
    callbacks=callbacks
)

# 保存最终模型
model.save(config['model_save_path'])
print("训练完成，模型已保存至:", config['model_save_path'])
