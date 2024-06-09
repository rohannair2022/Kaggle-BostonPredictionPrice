import pandas as pd
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataFrame = pd.read_csv("Dataset/train.csv")


def data_preprocess(data_frame: pd.DataFrame) -> pd.DataFrame:
    # First set of reduction on the size of int and float variables
    for column in data_frame.columns:
        if data_frame[column].dtype == 'int64':
            if 'year' in column.lower():
                data_frame[column] = data_frame[column].astype('int8')
            else:
                data_frame[column] = data_frame[column].astype('int32')
        elif data_frame[column].dtype == 'float64':
            if 'year' in column.lower():
                data_frame[column] = data_frame[column].astype('int8')
            else:
                data_frame[column] = data_frame[column].astype('float32')

    # Dropping columns with 85 percent or more missing values (arbitrary value chosen):
    missing_values_columns_drop = [[column, data_frame[column].isnull().sum()] for column in data_frame.columns if
                              data_frame[column].isnull().sum() > 0]
    for column in missing_values_columns_drop:
        if column[1] > len(data_frame)*0.15:
            data_frame.drop(column[0], axis=1, inplace=True)

    # Dropping columns with no important value:
    data_frame.drop('Id',axis=1, inplace=True)

    # Filling in the missing values:
    missing_values_columns_fill = [[column, data_frame[column].isnull().sum()] for column in data_frame.columns if
                              data_frame[column].isnull().sum() > 0]

    for column in missing_values_columns_fill:
        if data_frame[column[0]].dtype == 'float32':
            mean_value = data_frame[column[0]].mean().astype('float16')
            data_frame[column[0]] = data_frame[column[0]].fillna(mean_value)
        elif data_frame[column[0]].dtype == 'object':
            data_frame[column[0]] = data_frame[column[0]].fillna(mode(data_frame[column[0]]))

    # Encoding Categorical Value and Scaling numerical values"
    for column in data_frame.columns:
        if data_frame[column].dtype == 'object':
            if column in "BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, GarageFinish, GarageQual, GarageCond, ExterQual, ExterCond, HeatingQC, KitchenQual, PavedDrive, Street, CentralAir":
                label_encoder = LabelEncoder()
                scaler = MinMaxScaler()
                data_frame[column] = label_encoder.fit_transform(data_frame[column])
                data_frame[column] = data_frame[column].astype('int8')
                data_frame[column] = scaler.fit_transform(data_frame[[column]])
                data_frame[column] = data_frame[column].astype('float16')
            else:
                one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
                one_hot_encoded = one_hot_encoder.fit_transform(data_frame[[column]])
                data_frame = pd.concat([data_frame, one_hot_encoded], axis=1)
                data_frame.drop(column, axis=1, inplace=True)

        elif column != 'SalePrice':
            scaler = MinMaxScaler()
            data_frame[column] = scaler.fit_transform(data_frame[[column]])
            data_frame[column] = data_frame[column].astype('float16')

    return data_frame


# Apply the preprocessing function
dataFrame = data_preprocess(dataFrame)
dataFrame.to_csv('output.csv', index=False)
print(dataFrame.info())