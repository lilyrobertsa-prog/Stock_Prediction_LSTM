from sklearn.linear_model import LinearRegression

class Lin_Regression:

    def __init__(self):

        self.model = LinearRegression()

    def train(self, X_train_raw, y_train_raw):
        self.model.fit(X_train_raw, y_train_raw)

    def predict(self, X):

        return self.model.predict(X)

