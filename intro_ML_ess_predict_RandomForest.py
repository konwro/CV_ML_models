
# Task: predict rated power of ESS based on functionality
# Sources:
    # kaggle intro to ML course materials
# Database
    # edited SANDIA Global Enerrgy Storage Database
# Model: RandomForestRegressor

# step 1: Select data for modelling
import pandas as pd
import numpy as np 

#  index = None, header=True
ess_db = pd.read_excel(r'/path/ess_set1.xlsx')
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

# step 7: find optimal value of max_leaf_nodes
# define get_mae function
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    mae = MAE(val_y, preds)
    return(mae)
    
# define list of max_leaf_nodes values to be tested
max_leaf_nodes_test = [3, 30, 300, 3000, 30000, 30000, 300000, 3000000]

# for loop to compare different values of maax_leaf_nodes
for max_leaf_nodes in max_leaf_nodes_test:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print('max leaf nodes: %d \t MAE: %d'%(max_leaf_nodes, my_mae))

# store optimal max_leaf_nodes value - for loop one liner
scores =  {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in max_leaf_nodes_test}
optimal_size = min(scores, key=scores.get)
print('optimal max_leaf_nodes: \t%d' %(optimal_size))

# step 8: Use RandomForestRegressor
# build RandomForest model
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)

# fit RandomForest model
forest_model.fit(train_X, train_y)

# make predictions
preds_rf = forest_model.predict(val_X)

# model validation
error_mae = MAE(val_y, preds_rf)
print('MAE (RandomForestRegressor):\t%d'%(error_mae))
