import os
import torch
import torch.nn as nn

class Phase1(torch.nn.Module):
    def __init__(self):
        super(Phase2, self).__init__()
        self.vitamon = nn.Sequential(

            # input(224x224x25)
            nn.Conv2D(input_shape=(224, 224, 25), filters=32, kernel_size=(3, 3)),  # 확인하기
            nn.BatchNormalization(),
            nn.ReLU(),

            # {3x3 conv(32)} x 2
            nn.Conv2D(32, (3, 3)),
            nn.BatchNormalization(),
            nn.ReLU(),
            nn.Conv2D(32, (3, 3)),
            nn.BatchNormalization(),
            nn.ReLU(),

            # 3x3 conv(64)
            nn.Conv2D(64, (3, 3)),
            nn.BatchNormalization(),
            nn.ReLU(),

            # Max Pooling
            nn.MaxPool2d(),  # 확인하기

            # 3x3 conv(80)
            nn.Conv2D(80, (3, 3)),
            nn.BatchNormalization(),
            nn.ReLU(),

            # 3x3 conv(192)
            nn.Conv2D(192, (3, 3)),
            nn.BatchNormalization(),
            nn.ReLU(),

            # Max Pooling
            nn.MaxPool2d(), #확인하기

            # Inception Module


            # {Average Pooling} x 4
            nn.averagePooling2D(),
            nn.averagePooling2D(),
            nn.averagePooling2D(),
            nn.averagePooling2D(),

            # Dropout(0.5)
            nn.Dropout(0.5),

            # Flatten
            nn.Flatten(),

            # {Fully connected} x 2
            nn.Dense(7, activation='relu'),
            nn.Dense(7, activation='softmax')

        )

    def call(self, x):
        x = self.vitamon(x)
        return x

    def summary(self):
        self.vitamon.summary()

class Phase2(torch.nn.Module):
    def __init__(self):
        super(Phase2, self).__init__()
        self.vitamon = nn.Sequential(

            # input(224x224x7)
            nn.Conv2D(input_shape=(224, 224, 7), filters=32, kernel_size=(3, 3)),  # 확인하기
            nn.BatchNormalization(),
            nn.ReLU(),

            # {3x3 conv(32)} x 2
            nn.Conv2D(32, (3, 3)),
            nn.BatchNormalization(),
            nn.ReLU(),
            nn.Conv2D(32, (3, 3)),
            nn.BatchNormalization(),
            nn.ReLU(),

            # 3x3 conv(64)
            nn.Conv2D(64, (3, 3)),
            nn.BatchNormalization(),
            nn.ReLU(),

            # Average Pooling
            nn.averagePooling2D(),
            nn.averagePooling2D(),
            nn.averagePooling2D(),
            nn.averagePooling2D(),

            # Dropout(0.5)
            nn.Dropout(0.5),

            # Flatten
            nn.Flatten(),

            # Fully connected
            nn.Dense(7, activation='relu'),
            nn.Dense(7, activation='softmax')

        )

    def call(self, x):
        x = self.vitamon(x)
        return x

    def summary(self):
        self.vitamon.summary()
