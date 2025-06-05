import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

poland_data = pd.read_csv("../data/raw/apartments_pl_2023_08.csv")

fake_boolean_columns = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']

for col in fake_boolean_columns:
    # column exists and type check
    if col in poland_data.columns and poland_data[col].dtype == 'object':
        # unique values check
        unique_values = poland_data[col].dropna().unique()
        if set(unique_values).issubset({'yes', 'no'}):
            # yes = 1, no = 0, NaN = 0
            poland_data[col] = poland_data[col].map({'yes': 1, 'no': 0}).fillna(0)
        else:
            print(f"Column {col} has other values, then yes/no: {unique_values}")
    else:
        print(f"Column {col} doesnt exist or is not object type")

s = (poland_data.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
# 'id', 'city', 'type', 'ownership', 'buildingMaterial', 'condition'
# 'id' we'll not use, 'city' will be converted to avg_price in city with TargetEncoder
# 'ownership' will be converted the way like fake_boolean_columns
# 'type'
# 'buildingMaterial', 'condition' we will NOT use,
# because of lack of information: 39% and 76% of NaN

if 'ownership' in poland_data.columns and poland_data['ownership'].dtype == 'object':
    unique_values = poland_data['ownership'].dropna().unique()
    if set(unique_values).issubset({'condominium', 'cooperative'}):
        #  'condominium' = 1  'cooperative' or NaN = 0
        poland_data['ownership'] = poland_data['ownership'].map({'condominium': 1, 'cooperative': 0}).fillna(0)
    else:
        print(f"Column 'ownership' has other values than 'condominium'/'cooperative': {unique_values}")
else:
    print(f"Column 'ownership' does not exist or is not of object type")

print(f"\nUnique types({len(poland_data['type'].unique())}): {poland_data['type'].unique()}")

print(poland_data.groupby('type')['price'].mean().sort_values())
# blockOfFlats         605831.715260
# tenement             734817.556477
# apartmentBuilding    918659.025330
# NaN ~ 20% so I decided to do label encoding:
# NaN = 0
# blockOfFlats = 1
# tenement = 2
# apartmentBuilding = 3

if 'type' in poland_data.columns and poland_data['type'].dtype == 'object':
    unique_values = poland_data['type'].dropna().unique()
    if set(unique_values).issubset({'blockOfFlats', 'tenement', 'apartmentBuilding'}):
        poland_data['type'] = poland_data['type'].map({
            'blockOfFlats': 1,
            'tenement': 2,
            'apartmentBuilding': 3
        }).fillna(0)
    else:
        print(f"Column 'type' has other values than our map: {unique_values}")
else:
    print(f"Column 'type' does not exist or is not of object type")

corr_matrix = poland_data.corr(numeric_only=True)
print(corr_matrix["price"].sort_values(ascending=False))

selected_features = ['squareMeters', 'rooms', 'type', 'longitude', 'hasElevator',
                    'poiCount', 'hasSecurity', 'hasParkingSpace', 'buildYear',
                    'centreDistance']

X = poland_data.drop(columns = ['price', 'id'])
y = poland_data['price']
print('\nINITIAL X.INFO()\n')
X.info()

# dividing data into training and comparing values
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
print(f"\ntrain_X unique cities ({len(train_X['city'].unique())}): {train_X['city'].unique()}")
print(f"\nval_X unique cities ({len(val_X['city'].unique())}): {val_X['city'].unique()}")
print("\ntrain_X unique cities count:")
print(train_X['city'].value_counts())

# TargetEncoder initialisation
encoder = TargetEncoder(cols = ['city'], smoothing = 1.0)

train_X['city_encoded'] = encoder.fit_transform(train_X['city'], train_y)
val_X['city_encoded'] = encoder.transform(val_X['city'])
selected_features.append('city_encoded')
train_X = train_X[selected_features]
val_X = val_X[selected_features]

#print(train_X.shape, val_X.shape, train_y.shape, val_y.shape)

train_X.to_csv("../data/processed/train_X.csv", index = False)
val_X.to_csv("../data/processed/val_X.csv", index = False)
train_y.to_csv("../data/processed/train_y.csv", index = False)
val_y.to_csv("../data/processed/val_y.csv", index = False)