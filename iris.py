from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix

iris = datasets.load_iris()
inputs = iris.data
outputs = iris.target
print(inputs.shape)

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2)

network = MLPClassifier(max_iter=2000,
                        verbose=True,
                        tol=0.0000100,
                        activation='logistic',
                        solver='adam',
                        learning_rate='constant',
                        learning_rate_init=0.001,
                        batch_size=32,
                        hidden_layer_sizes=(4, 5)
                        # early_stopping = True,
                        # n_iter_no_change = 50
                        )
network.fit(X_train, y_train)
print(network.coefs_)
print(network.intercepts_)
print(network.n_layers_)
print(network.out_activation_)

predictions = network.predict(X_test)
print(predictions)
accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
print(cm)
confusion_matrix = ConfusionMatrix(network, classes=iris.target_names)
confusion_matrix.fit(X_train, y_train)
confusion_matrix.score(X_test, y_test)
confusion_matrix.show()

print(X_test[0], y_test[0])
print(X_test[0].shape)
new = X_test[0].reshape(1, -1)
print(new.shape)
network.predict(new)
print(iris.target_names[network.predict(new)[0]])
