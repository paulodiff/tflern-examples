import tflearn
import numpy as np
from tflearn.data_utils import load_csv
from tabulate import tabulate
from sklearn.metrics import mean_squared_error


y_true = [3, -0.5, 2, 7]
y_pred = [4, -0.4, 2.1, 7.1]
print(mean_squared_error(y_true, y_pred))


exit(1)

# data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=13)
#data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)

# X = data[0:10,:]
#X_slice = Dataset_minmax[0:1,0:]
#Y = la[:,13]

#X_minmax = min_max_scaler.fit_transform(X)
#Y_minmax = min_max_scaler.fit_transform(Y)

#print(data)
#input("Press Enter to continue evaluation...")
#print(labels)
#input("Press Enter to continue evaluation...")
#print(tabulate(data))
#input("Press Enter to continue evaluation...")
#print(tabulate(labels))
#input("Press Enter to continue evaluation...")





# Regression data
# X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
X = [[3.3,4.4],
     [5.5,6.71],
     [6.93,4.168],
     [9.779,6.182],
     [7.59,2.167],
     [7.042,10.791],
     [5.313,7.997],
     [7.59,2.167],
     [7.042,10.791],
     [5.313,7.997],
     [7.59,2.167],
     [7.042,10.791],
     [5.313,7.997],
     [7.042,10.791],
     [5.313,7.997],
     [5.313,7.997]]
     
Y = [[1.7],
     [2.76],
     [2.09],
     [3.19],
     [1.694],
     [1.573],
     [3.366],
     [2.596],
     [2.53],
     [1.221],
     [2.827],
     [3.465],
     [1.65],
     [2.904],
     [2.42],
     [2.94]]
     
# Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]     

# Linear Regression graph
X = np.reshape(X, (-1, 2))
Y = np.reshape(Y, (-1, 1))

network = tflearn.input_data(shape=[None,2], name='input')
network = tflearn.fully_connected(network, 32)
network = tflearn.fully_connected(network, 1, activation='linear')
# linear = tflearn.single_unit(input_)
regression = tflearn.regression(network, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01, name='targets')
model = tflearn.DNN(regression)
model.fit(  {'input': X}, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

print("\nRegression result:")
print("Y = " + str(model.get_weights(network.W)) + "*X + " + str(model.get_weights(network.b)))

prediction_value = [[3.3,4.4], [3.3,4.4], [9.1,24.4], [5.313,7.997]]
print("\nTest prediction for:", prediction_value)
print(model.predict(prediction_value))
# should output (close, not exact) y = [1.5315033197402954, 1.5585315227508545, 1.5855598449707031]