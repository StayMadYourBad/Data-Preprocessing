# ---------------------------------------------------------------------------- #
#                Importing the necessary libraries and datasets                #
# ---------------------------------------------------------------------------- #

#Adding a shortcut to save character count
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1 ].values #Taking all the rows, then all the columns excluding the last column
y = dataset.iloc[:, -1].values #Taking all the rows, and only the last column which Purchased
print(X, y)


