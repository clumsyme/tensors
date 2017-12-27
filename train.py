import numpy as np
from sklearn.linear_model import LinearRegression
from prepare import housing_prepared, housing_labels, full_pipeline
from utils import housing

# a linear modal train
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print(f'Predictions:\t{lin_reg.predict(some_data_prepared)}')
print(f'Labels:\t\t{list(some_labels)}')

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# decision tree
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_tree_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_tree_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)