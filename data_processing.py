# ---------------------------------------------------------------------------- #
#                Importing the necessary libraries and datasets                #
# ---------------------------------------------------------------------------- #

#Adding a shortcut to save character count
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import  LabelEncoder
# Importing Dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1 ].values #Taking all the rows, all columns excluding the last column
y = dataset.iloc[:, -1].values #Taking the last column

# ---------------------------------------------------------------------------- #
#                          Taking Care of Missing Data                         #
# ---------------------------------------------------------------------------- #

imputer = SimpleImputer(missing_values = np.nan, strategy= 'mean') #Missing all nan/NaN values with the mean
imputer.fit(X[:, 1:3]) #Look at missing values in the columns and calculate the mean. Only takes in numerical data
X[:, 1:3] = imputer.transform(X[:, 1:3]) #Replaces all the NaN values with the strategy, mean

# Since transform returns the adjusted rows, we update the rows with it.


# ---------------------------------------------------------------------------- #
#                           Encoding Categorical Data                          #
# ---------------------------------------------------------------------------- #

#Independant Variables

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X)) # Encode the country data with OneHotEncoder and passthrough the rest.
print(X)

#Dependent Variable

le = LabelEncoder()
y = le.fit_transform(y)
print(y)