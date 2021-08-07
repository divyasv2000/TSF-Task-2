import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 6

irisDataset = pd.read_csv('assets\iris.csv', header = 0)
x = irisDataset.iloc[:, :2]
y = irisDataset.iloc[:, -1]

model = KNeighborsClassifier(n_neighbors, weights='distance')
model.fit(x, y)

length = float(input('Enter the Sepal Length (cm) : '))
width = float(input('Enter the Sepal Width (cm) : '))
prediction = model.predict([[length, width]])
print('Prediction : ' + prediction)
