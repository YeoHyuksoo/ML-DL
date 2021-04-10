import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.

        # Tip : log computation may cause some error, so try to solve it by adding an epsilon(small value) within log term.
        epsilon = 1e-7
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        w = self.W
        #print("x.shape {}, y.shape{}".format(x.shape, y.shape))
        #print(int(x.shape[0] / batch_size))

        avgloss = 0
        for i in range(epochs):
            los = 0
            grad = 0
            for j in range(int(x.shape[0] / batch_size) + 1):
                predict = np.dot(x[j * batch_size:(j + 1) * batch_size], w)
                predict = self._sigmoid(predict)
                error = predict - y[j * batch_size:(j + 1) * batch_size]
                sum_error = 0.0
                for k in range(len(error)):
                    if y[j * batch_size + k][0] == 0:
                        sum_error += -1 * np.log(1 - predict[k][0] + epsilon)
                    elif y[j * batch_size + k][0] == 1:
                        sum_error += -1 * np.log(predict[k][0] + epsilon)

                los += 1 / len(error) * sum_error
                grad += 1 / x.shape[0] * np.dot(x[j * batch_size:(j + 1) * batch_size].T, error)

            avgloss = los / (int(x.shape[0] / batch_size) + 1)
            #print("avg loss: ", avgloss)
            w = optim.update(w, grad, lr)
            self.W = w

        loss = avgloss
        # ============================================================
        return loss

    def forward(self, x):
        threshold = 0.5
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        predict = np.dot(x, self.W)
        for i in range(len(predict)):
            if self._sigmoid(predict[i][0]) >= threshold:
                predict[i][0] = 1
            else:
                predict[i][0] = 0
        y_predicted = predict

        # ============================================================

        return y_predicted

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1 / (1 + np.exp(-x))
        # ============================================================
        return sigmoid
