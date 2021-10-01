
# Task: 
#   1) predict rated power of ESS based on its functionality defined in a one-hot encoding manner
# Sources:
    # kaggle intro to ML course materials
# Database
    # edited SANDIA Global Energy Storage Database
# Model: DecisionTreeRegressor

# step 1: Select data for modelling
import pandas as pd
import numpy as np 

#  index = None, header=True
ess_db = pd.read_excel(r'/home/wk/OneDrive/ML_AI/ess_set1.xlsx')
# print(ess_db.describe)
# print(ess_db.columns)

ess_db.dropna(inplace=True)
# print(ess_db.describe)
# print(ess_db.head(50))
# print(ess_db.tail(50))

# step 2: Choose prediction target (y) and features (X)
# y - target
# y = ess_db['rated_power_kW']
y = ess_db.rated_power_kW

# print(y.head(10))

# X - features
features = list(ess_db)
# print(features)
features_select = features
del features_select[0:2]

X = ess_db[features_select]
# print(X.head(10))
# print(X.describe)

# step 3: split data
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

# step 3: Define model
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

# step 4: fit model
model.fit(train_X,train_y)

# step 5: make predictions
preds = model.predict(val_X)
# print(preds)

# step 6: model validation
from sklearn.metrics import mean_absolute_error as MAE 
mae_error = MAE(val_y, preds)
 
print(mae_error)
 
  
