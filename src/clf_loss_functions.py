import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math

#Code from https://github.com/UKPLab/emnlp2020-debiasing-unknown

class ClfLossFunction(nn.Module):
    """Torch classification debiasing loss function"""

    def forward(self, hidden, logits, bias, teach_probs, labels):
        """
        :param hidden: [batch, n_features] hidden features from the model
        :param logits: [batch, n_classes] logit score for each class
        :param bias: [batch, n_classes] log-probabilties from the bias for each class
        :param labels: [batch] integer class labels
        :return: scalar loss
        """
        raise NotImplementedError()


class Plain(ClfLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        return F.cross_entropy(logits, labels)


class BiasProductByTeacher(ClfLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        logits = F.log_softmax(logits, 1)
        teacher_logits = torch.log(teacher_probs)
        return F.cross_entropy(logits + teacher_logits, labels)

class BiasProductByTeacherAnnealed(ClfLossFunction):
    def __init__(self, max_theta=1.0, min_theta=0.8,
                 total_steps=12272, num_epochs=3):
        super().__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.num_train_optimization_steps = int(total_steps)
        self.num_epochs = num_epochs
        self.current_step = 0

    def get_current_theta(self):
        linspace_theta = np.linspace(self.max_theta, self.min_theta,
                                     self.num_train_optimization_steps+self.num_epochs)
        current_theta = linspace_theta[self.current_step]
        self.current_step += 1
        return current_theta

    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        logits = F.log_softmax(logits, 1)

        current_theta = self.get_current_theta()
        denom = (teacher_probs ** current_theta).sum(1).unsqueeze(1).expand_as(teacher_probs)
        scaled_probs = (teacher_probs ** current_theta) / denom

        teacher_logits = torch.log(scaled_probs)
        return F.cross_entropy(logits + teacher_logits, labels)

class DistillLoss(ClfLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        example_loss = -(teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss

class DistillLossAnnealed(ClfLossFunction):
    def __init__(self, max_theta=1.0, min_theta=0.8,
                 total_steps=12272, num_epochs=3):
        super().__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.num_train_optimization_steps = int(total_steps)
        self.num_epochs = num_epochs
        self.current_step = 0

    def get_current_theta(self):
        linspace_theta = np.linspace(self.max_theta, self.min_theta,
                                     self.num_train_optimization_steps+self.num_epochs)
        current_theta = linspace_theta[self.current_step]

        if self.current_step < self.num_train_optimization_steps: 
            self.current_step += 1
        return current_theta

    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        current_theta = self.get_current_theta()
        denom = (teacher_probs ** current_theta).sum(1).unsqueeze(1).expand_as(teacher_probs)
        scaled_teacher_probs = (teacher_probs ** current_theta) / denom

        example_loss = -(scaled_teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss


class SmoothedDistillLoss(ClfLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)
        
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = (1 - (one_hot_labels * torch.exp(bias)).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

        example_loss = -(norm_teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss


class SmoothedDistillLossAnnealed(ClfLossFunction):
    def __init__(self, max_theta=1.0, min_theta=0.8,
                 total_steps=12272, num_epochs=3):
        super().__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.num_train_optimization_steps = int(total_steps)
        self.num_epochs = num_epochs
        self.current_step = 0

    def get_current_theta(self):
        linspace_theta = np.linspace(self.max_theta, self.min_theta,
                                     self.num_train_optimization_steps+self.num_epochs)
        current_theta = linspace_theta[self.current_step]
        if self.current_step < self.num_train_optimization_steps: 
            self.current_step += 1
        return current_theta

    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        bias_probs = torch.exp(bias)
        current_theta = self.get_current_theta()
        denom = (bias_probs ** current_theta).sum(1).unsqueeze(1).expand_as(bias_probs)
        scaled_bias_probs = (bias_probs ** current_theta) / denom

        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = (1 - (one_hot_labels * scaled_bias_probs).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

        example_loss = -(norm_teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss


class ReweightBaseline(ClfLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction='none')
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = 1 - (one_hot_labels * torch.exp(bias)).sum(1)

        return (weights * loss).sum() / weights.sum()


class ReweightByTeacher(ClfLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels, theta=1.0):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction='none')
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]

        weights = 1 - (one_hot_labels * teacher_probs).sum(1)
        # weights = weights ** theta
        
        return (weights * loss).sum() / weights.sum()


class ReweightByTeacherAnnealed(ClfLossFunction):
    def __init__(self, max_theta=1.0, min_theta=0.8,
                 total_steps=12272, num_epochs=3):
        super().__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.num_train_optimization_steps = int(total_steps)
        self.num_epochs = num_epochs
        self.current_step = 0

    def get_current_theta(self):
        linspace_theta = np.linspace(self.max_theta, self.min_theta,
                                     self.num_train_optimization_steps+self.num_epochs)
        current_theta = linspace_theta[self.current_step]
        if self.current_step < self.num_train_optimization_steps: 
            self.current_step += 1
        return current_theta


    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction='none')
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]

        weights = 1 - (one_hot_labels * teacher_probs).sum(1)

        current_theta = self.get_current_theta()
        weights = weights ** current_theta

        return (weights * loss).sum() / weights.sum()