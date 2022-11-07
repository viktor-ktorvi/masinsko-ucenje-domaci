import numpy as np


class Logger:
    """
    Logger class. Tracks loss.
    """

    def __init__(self, epochs):
        """
        Initialize loss.
        :param epochs: int; number of samples to track.
        """
        self.running_loss = 0
        self.loss = np.zeros((epochs,))

    def clear(self):
        """
        Clear running loss.
        :return:
        """
        self.running_loss = 0

    def running_update(self, loss_val):
        """
        Update running loss.
        :param loss_val: float; loss value
        :return:
        """
        self.running_loss += loss_val

    def update(self, index, scale_factor=1):
        """
        Write the loss value at index scaled by a factor.
        :param index: int
        :param scale_factor: float
        :return:
        """
        self.loss[index] = self.running_loss / scale_factor


class ClassificationLogger(Logger):
    """
    Classification logger class. Tracks loss and accuracy.
    """

    def __init__(self, epochs):
        """
        Initialize loss.
        :param epochs: int; number of samples to track.
        """
        super(ClassificationLogger, self).__init__(epochs)
        self.running_acc_cnt = 0
        self.accuracy = np.zeros((epochs,))

    def clear(self):
        """
        Clear running loss and accuracy count.
        :return:
        """
        super(ClassificationLogger, self).clear()
        self.running_acc_cnt = 0

    def running_update(self, loss_val, acc_cnt=0):
        """
        Update running loss.
        :param loss_val: float; loss value
        :param acc_cnt: int; number of correct predictions
        :return:
        """
        super(ClassificationLogger, self).running_update(loss_val)
        self.running_acc_cnt += acc_cnt

    def update(self, index, scale_factor=1):
        """
        Write the loss value at index scaled by a factor.
        :param index: int
        :param scale_factor: float
        :return
        """
        super(ClassificationLogger, self).update(index, scale_factor)
        self.accuracy[index] = self.running_acc_cnt / scale_factor
