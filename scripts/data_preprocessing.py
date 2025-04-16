import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

poland_data = pd.read_csv("../data/raw/apartments_pl_2023_08.csv")
poland_data_features = ['city', 'type', 'squareMeters', 'rooms', 'buildYear',
                        'buildingMaterial', 'hasBalcony', 'hasParkingSpace']
poland_data.info()
X = poland_data[poland_data_features]
y = poland_data['price']
#print('\nINITIAL X.INFO()\n')
#X.info()

# dividing data into training and comparing values
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

#One-hot encoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
s = (train_X.dtypes == 'object') # finding non-numeric columns
object_cols = list(s[s].index) # list of non-numeric columns

#OH_encoder deletes names and indexes of columns, so we have to add it back
OH_train_columns = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]),
                                columns = OH_encoder.get_feature_names_out(object_cols),
                                index = train_X.index)
OH_val_columns = pd.DataFrame(OH_encoder.transform(val_X[object_cols]),
                              columns = OH_encoder.get_feature_names_out(object_cols),
                              index = val_X.index)
# deleting 'old' non-numeric columns
num_X_train = train_X.drop(object_cols, axis=1)
num_X_valid = val_X.drop(object_cols, axis=1)

# merge of numeric-only and OH-coded columns
OH_train_X = pd.concat([num_X_train, OH_train_columns], axis=1)
OH_valid_X = pd.concat([num_X_valid, OH_val_columns], axis=1)

#OH_train_X.info()
# at the moment we have 29!!! columns

# I've decided to save it to csv
OH_train_X.to_csv("../data/processed/train_X.csv", index = False)
OH_valid_X.to_csv("../data/processed/val_X.csv", index = False)
