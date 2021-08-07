import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
#from pandas.core.indexes.api import Index

irisDataset = pd.read_csv('assets\iris.csv')
sns.scatterplot(x='sepals-length', y='sepals-width', hue='label', data=irisDataset).set_title('Iris Data Distribution')
plt.figure(1)
plt.show()