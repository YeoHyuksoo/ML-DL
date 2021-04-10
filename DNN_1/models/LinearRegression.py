import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        w = self.W
        #print("x.shape {}, y.shape{}".format(x.shape, y.shape))
        #print(int(x.shape[0] / batch_size))
        avgloss = 0
        for i in range(epochs):
            loss = 0
            grad = 0
            for j in range(int(np.ceil(x.shape[0] / batch_size))):
                predict = self.forward(x[j*batch_size:(j+1)*batch_size])
                error = predict - y[j*batch_size:(j+1)*batch_size]
                sum_error = 0
                for e in error:
                    e = e * e
                    sum_error += e
                loss += 1/len(error) * sum_error
                grad += 1/(x.shape[0]) * np.dot(x[j*batch_size:(j+1)*batch_size].T, error)

            avgloss = loss/(int(x.shape[0] / batch_size)+1)
            #print("avg loss: ", avgloss)
            w = optim.update(w, grad, lr)
            self.W = w

        final_loss = avgloss
        # ============================================================

        return final_loss

    def forward(self, x):
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        y_predicted = np.dot(x, self.W)
        # ============================================================
        return y_predicted
