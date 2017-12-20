import matplotlib.pyplot as plt
from utils import strat_train_set
from pandas.plotting import scatter_matrix

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()
# plt.show()

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)

housing["room_per_household"] = housing["total_rooms"] / housing['households']
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing['total_rooms']
housing["population_pre_household"] = housing["population"] / housing["households"]

# plt.show()
