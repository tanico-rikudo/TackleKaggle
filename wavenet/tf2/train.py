
import tensorflow as tf

import numpy as np
import os

from model.wavenet import WaveNet
from model.module import CrossEntropyLoss
from dataset import get_train_data
import hparams

@tf.function
def train_step(model, x, mel_sp, y, loss_fn, optimizer):
    # 最適化する範囲
    with tf.GradientTape() as tape:
        y_hat = model(x, mel_sp)
        loss = loss_fn(y, y_hat)

    # 勾配計算→最適化
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss



def train():
    # 保存先
    os.makedirs(hparams.result_dir + "weights/", exist_ok=True)
    # summary
    summary_writer = tf.summary.create_file_writer(hparams.result_dir)
    # wavenet model OBJ
    wavenet = WaveNet(hparams.num_mels, hparams.upsample_scales)
    # loss func
    loss_fn = CrossEntropyLoss(num_classes=256)

    # 学習率を途中で変化させる
    # ここではstep毎に指数的に
    # それをoptimizserに乗せる
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(hparams.learning_rate,
                                                                 decay_steps=hparams.exponential_decay_steps,
                                                                 decay_rate=hparams.exponential_decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                         beta_1=hparams.beta_1)

    if hparams.load_path is not None:
        wavenet.load_weights(hparams.load_path)
        step = np.load(hparams.result_dir + "weights/step.npy")
        step = step
        print(f"weights load: {hparams.load_path}")
    else:
        step = 0

    # epoch
    for epoch in range(hparams.epoch):
        train_data = get_train_data()
        # １データずつ学習する
        for x, mel_sp, y in train_data:
            loss = train_step(wavenet, x, mel_sp, y, loss_fn, optimizer)
            with summary_writer.as_default():
                tf.summary.scalar('train/loss', loss, step=step)

            step += 1

        # 一定周期で保存
        if epoch % hparams.save_interval == 0:
            print(f'Step {step}, Loss: {loss}')
            np.save(hparams.result_dir + f"weights/step.npy", np.array(step))
            wavenet.save_weights(hparams.result_dir + f"weights/wavenet_{epoch:04}")

    np.save(hparams.result_dir + f"weights/step.npy", np.array(step))
    wavenet.save_weights(hparams.result_dir + f"weights/wavenet_{epoch:04}")
