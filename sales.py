import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# target encoding
def target_encoding(train, test, target, cat_features):
    train['target'] = target.copy()
    kf = KFold(n_splits = 5, shuffle = True, random_state = 21)
    # loop
    for train_index, val_index in kf.split(train):
        train_fold, val_fold = train.iloc[train_index], train.iloc[val_index]

        for col in cat_features:
            train.loc[val_index, col+'_min_sales'] = val_fold[col].map(train_fold.groupby(col).target.min())
            train.loc[val_index, col+'_max_sales'] = val_fold[col].map(train_fold.groupby(col).target.max())
            train.loc[val_index, col+'_mean_sales'] = val_fold[col].map(train_fold.groupby(col).target.mean())
            train.loc[val_index, col+'_25perc_sales'] = val_fold[col].map(train_fold.groupby(col).target.quantile(.25).to_dict())
            train.loc[val_index, col+'_75perc_sales'] = val_fold[col].map(train_fold.groupby(col).target.quantile(.75).to_dict())
    # for test set
    for col in cat_features:
        test[col +'_min_sales'] = test[col].map(train.groupby(col).target.min())
        test[col +'_max_sales'] = test[col].map(train.groupby(col).target.max())
        test[col +'_mean_sales'] = test[col].map(train.groupby(col).target.mean())
        test[col +'_25perc_sales'] = test[col].map(train.groupby(col).target.quantile(.25))
        test[col +'_75perc_sales'] = test[col].map(train.groupby(col).target.quantile(.75))
    # return
    train.drop('target', axis = 1, inplace=True)
    return train, test

if __name__ == "__main__":
    # import data
    print('Import data\n')
    train = pd.read_csv('train.csv')
    xtest = pd.read_csv('test.csv')
    sub = pd.read_csv('sample_submission.csv')

    xtrain = train.drop('Item_Outlet_Sales', axis = 1)
    ytrain = np.array((train['Item_Outlet_Sales'] / train['Item_MRP']))

    print('Train shape', train.shape)
    print('Test shape', xtest.shape)
    print('..........................')

    # data cleaning
    print('data cleaning')
    mapping = {'Low Fat' : 'Low Fat',
               'LF' : 'Low Fat',
               'low fat' : 'Low Fat',
               'Regular' : 'Regular',
               'reg' : 'Regular'}

    xtrain['Item_Fat_Content'] = xtrain['Item_Fat_Content'].map(mapping)
    xtest['Item_Fat_Content'] = xtest['Item_Fat_Content'].map(mapping)
    
    # Missing value impute
    print('Impute missing value')
    xtrain['Item_Weight'] = xtrain['Item_Weight'].fillna(xtrain.Item_Weight.mean())
    xtrain['Outlet_Size'] = xtrain['Outlet_Size'].fillna('unknown')

    xtest['Item_Weight'] = xtest['Item_Weight'].fillna(xtrain.Item_Weight.mean())
    xtest['Outlet_Size'] = xtest['Outlet_Size'].fillna('unknown')

    # new item identi
    xtrain['item_identifier_1'] = xtrain['Item_Identifier'].apply(lambda x: x[:2])
    xtrain['item_identifier_2'] = xtrain['Item_Identifier'].apply(lambda x: x[2:3])
    xtrain['item_identifier_3'] = xtrain['Item_Identifier'].apply(lambda x: x[3:])

    xtest['item_identifier_1'] = xtest['Item_Identifier'].apply(lambda x: x[:2])
    xtest['item_identifier_2'] = xtest['Item_Identifier'].apply(lambda x: x[2:3])
    xtest['item_identifier_3'] = xtest['Item_Identifier'].apply(lambda x: x[3:])

    # target encoding
    cat_features = ['Outlet_Identifier']
    xtrain, xtest = target_encoding(xtrain, xtest, ytrain, cat_features)

    # label encoding
    lbl_cols = ['Item_Fat_Content','Item_Type','Outlet_Size', 'Outlet_Location_Type','Outlet_Type',
                'Outlet_Establishment_Year','item_identifier_1','item_identifier_2','item_identifier_3']
    
    for col in lbl_cols:
        le = LabelEncoder()
        le.fit(xtrain[col].values.tolist() + xtest[col].values.tolist())
        xtrain.loc[:,col] = le.transform(xtrain[col].values.tolist()) 
        xtest.loc[:,col] = le.transform(xtest[col].values.tolist())

    print('Drop unused columns')
    drop_cols = ['Item_Identifier','Outlet_Identifier','Item_Weight','Item_Visibility']
    xtrain.drop(drop_cols, axis = 1, inplace = True)
    xtest.drop(drop_cols, axis = 1, inplace = True)

    # build model
    # bagged model
    bagged_train_pred = 0
    bagged_test_pred = 0
    no_of_bags = 3

    for i, (trees, depth, seed) in enumerate([[500, 6, 21], [500, 7, 42], [1000, 8, 84]], start = 1):
        print('bag {} of bags {}'.format(i, no_of_bags))
        print('Model training for estimators {}, depth {},random state {}'.format(trees, depth, seed))
        model = RandomForestRegressor(
                                      n_estimators = trees,
                                      max_depth = depth,
                                      n_jobs = -1,
                                      max_features = 0.8,
                                      random_state = seed
                                     )
        model.fit(xtrain, ytrain)
        test_pred = model.predict(xtest)
        bagged_test_pred += test_pred

    bagged_test_pred = bagged_test_pred/no_of_bags
    print('Model training complete....................')

    # Submission file
    sub['Item_Outlet_Sales'] = (bagged_test_pred * xtest['Item_MRP'])
    sub.to_csv('bagged_rf_final.csv', index = False)