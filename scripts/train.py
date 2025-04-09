import tensorflow as tf
from tiny_unet import build_tiny_unet, dice_coeff
from npz_dataset import NpzDataset
from datetime import datetime
import os
import yaml

def load_config(config_path="config.yml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        if isinstance(config.get('learning_rate'), str):
            config['learning_rate'] = float(config['learning_rate'])
        return config

config = load_config()

# 初始化数据集
train_data = NpzDataset(
    npz_dir=config['train_dir'],
    batch_size=config['batch_size'],
    augment=True
)
train_steps = train_data.steps_per_epoch
train_dataset =train_data.to_tf_dataset()

val_data = NpzDataset(
    npz_dir=config['val_dir'],
    batch_size=config['batch_size']
)
val_steps = val_data.steps_per_epoch
val_dataset=val_data.to_tf_dataset()

# 构建模型
model = build_tiny_unet(input_shape=(256, 512, 3))
model.summary()

# 自定义指标
def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1]),
        dice_coeff
    ]
)

# 回调函数
os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
os.makedirs(config['log_dir'], exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        config['model_save_path'],
        save_best_only=True,
        monitor='val_dice_coeff',
        mode='max'
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S")),
        update_freq='batch'
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=10,
        monitor='val_dice_coeff',
        mode='max',
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5
    )
]

# 训练模型
history = model.fit(
    train_dataset,
    epochs=config['epochs'],
    steps_per_epoch=train_steps,
    validation_data=val_dataset,
    validation_steps=val_steps,
    callbacks=callbacks
)

print("训练完成，模型已保存至:", config['model_save_path'])
