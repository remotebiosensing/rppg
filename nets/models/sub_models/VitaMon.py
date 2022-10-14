import os
import torch
import torch.nn as nn


class Phase2(vitamon):
    def __init__(self):
        super(Phase1, self).__init__()
        self.vitamon = Sequential(
            [
                # input(224x224x25)
                layers.Conv2D(input_shape=(224, 224, 25), filters=32, kernel_size=(3, 3)),  # 확인하기
                layers.BatchNormalization(),
                layers.ReLU(),

                # {3x3 conv(32)} x 2
                layers.Conv2D(32, (3, 3)),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(32, (3, 3)),
                layers.BatchNormalization(),
                layers.ReLU(),

                # 3x3 conv(64)
                layers.Conv2D(64, (3, 3)),
                layers.BatchNormalization(),
                layers.ReLU(),

                # Average Pooling
                layers.averagePooling2D(),
                layers.averagePooling2D(),
                layers.averagePooling2D(),
                layers.averagePooling2D(),

                # Dropout(0.5)
                layers.Dropout(0.5),

                # Flatten
                layers.Flatten(),

                # Fully connected
                layers.Dense(7, activation='relu'),
                lyaers.Dense(7, activation='softmax')
            ]
        )

    def call(self, x):
        x = self.vitamon(x)
        return x

    def summary(self):
        self.vitamon.summary()

        ]
        )
