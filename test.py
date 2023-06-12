import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Loading the data
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston_df = pd.read_csv('housing.csv', header = None, delimiter=r"\s+", names = columns)
boston_df.head()

# Summarizing the stats of the data
boston_df.describe()

# Check for missing values
boston_df.isnull().sum()

# Exploratory Analysis
boston_df.corr()
sns.pairplot(boston_df)

plt.scatter(boston_df['AGE'],boston_df['MEDV'])
plt.xlabel('AGE')
plt.ylabel('Price')

sns.regplot(x="AGE",y="MEDV",data=boston_df)

# Prep for modeling
x=boston_df.iloc[:,:-1]
y=boston_df.iloc[:,-1]

# Train Test Split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .25, random_state=16)

# Standardize the dataset
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

# Train the model
regression = LinearRegression()
regression.fit(x_train,y_train)

# Let's peak at the coefficients & intercept
print(regression.coef_)
print(regression.intercept_)

# Prediction using test data
pred = regression.predict(x_test)

# Plot the prediction, examine residuals
plt.scatter(y_test, pred)
residuals=y_test-pred
sns.displot(residuals,kind="kde")

# Check mean square error
print(mean_squared_error(y_test,pred))

# R squared
score=r2_score(y_test,pred)
print(score)

# New data prediction
boston_df.values[0,:-1].reshape(1,-1)
scaler.transform(boston_df.values[0,:-1].reshape(1,-1))
regression.predict(scaler.transform(boston_df.values[0,:-1].reshape(1,-1)))

# Pickling the model for deployment
import pickle

pickle.dump(regression,open('regmodel.pkl','wb'))
pickled_model = pickle.load(open('regmodel.pkl','rb'))
pickled_model.predict(scaler.transform(boston_df.values[0,:-1].reshape(1,-1)))



















