"""
model.py
---------
Defines the Deep Neural Network (DNN) used by StudySync.

Architecture (simple, 3 linear layers -> "deep" because it has more than
one hidden layer, which is the usual beginner-level definition of DNN):

    Input (6 features)
        -> Linear(6, 16) -> ReLU
        -> Linear(16, 8) -> ReLU
        -> Linear(8, 3)      <- raw scores ("logits") for 3 priority classes

We do NOT put a Softmax layer at the end because PyTorch's
nn.CrossEntropyLoss expects raw logits and applies log-softmax internally
(this is the standard, numerically-stable way to do multi-class
classification in PyTorch, and it is important to say this clearly in
the report/viva).
"""

import torch.nn as nn


class StudySyncDNN(nn.Module):
    def __init__(self, input_size=6, hidden1=16, hidden2=8, num_classes=3):
        super(StudySyncDNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output_layer(x)  # logits, softmax applied by the loss fn
        return x
