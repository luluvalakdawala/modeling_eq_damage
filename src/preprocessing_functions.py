import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def one_hot_encode_feature(df, name):
    """This funciton takes in a dataframe and a feature name and 
    One hot encodes the feature and adds it to the dataframe
    
    Returns transformed dataframe and the ohe object 
    used to transform the frame
    """
    
    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
    single_feature_df = df[[name]]
    ohe.fit(single_feature_df)
    feature_array = ohe.transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0], index=df.index)
    df = df.drop(name, axis=1)
    df = pd.concat([df, ohe_df], axis=1)
    
    #returning ohe here so that it can be used to transform X_test later
    return df, ohe

def ohe_all_categorical_features(df):
    """This function takes in a dataframe, identifies the
    dtypes in the dataframe and uses the object dtypes to
    list out categorical columns
    
    Next it use OneHotEncoder to convert those Categorical 
    features
    
    Returns: the transformed dataframe and a dictionary 
    containing the ohe object that can be used later to 
    transform the testing dataset
    """
    categorical_feats = df.dtypes[df.dtypes == 'object'].index
    #use helper function in loop to transform dataframe
    encoders = {}
    
    for name in categorical_feats:
        df, ohe = one_hot_encode_feature(df, name)
        encoders[name] = ohe
    
    return df, encoders

def one_hot_encode_test_features(df, name, ohe):
    """This funciton takes in the test dataframe, a feature name and 
    an ohe object and then One hot encodes the feature and adds
    it to the dataframe
    
    Returns the transformed test dataframe
    """
    
    single_feature_df = df[[name]]
    feature_array = ohe.transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0], index=df.index)
    df = df.drop(name, axis=1)
    df = pd.concat([df, ohe_df], axis=1)
    
    return df

def ohe_train_and_test_features(X_train, X_val, X_test):
    """This function takes in the train, validation and test sets.
    OneHotEncodes the features on train and transforms the 
    validation and test set with extracted encoders"""
    
    X_train, encoders = ohe_all_categorical_features(X_train)
    
    for key in encoders:
        X_test = one_hot_encode_test_features(X_test, key, encoders[key])
        X_val = one_hot_encode_test_features(X_val, key, encoders[key])
    
    return X_train, X_val, X_test

def one_hot_encode_geos(df, name):
    """This funciton takes in a dataframe and a feature name and 
    One hot encodes the feature and adds it to the dataframe
    
    Returns transformed dataframe and the ohe object 
    used to transform the frame
    """
    
    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
    single_feature_df = df[[name]]
    ohe.fit(single_feature_df)
    feature_array = ohe.transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0], index=df.index)
    df = df.drop(name, axis=1)
    df = pd.concat([df, ohe_df], axis=1)
    
    #returning ohe here so that it can be used to transform X_test later
    return df, ohe

def ohe_all_categorical_geos(df):
    """This function takes in a dataframe, identifies the
    dtypes in the dataframe and uses the object dtypes to
    list out categorical columns
    
    Next it use OneHotEncoder to convert those Categorical 
    features
    
    Returns: the transformed dataframe and a dictionary 
    containing the ohe object that can be used later to 
    transform the testing dataset
    """
    categorical_feats = ['district_id']
    #use helper function in loop to transform dataframe
    encoders = {}
    
    for name in categorical_feats:
        df, ohe = one_hot_encode_geos(df, name)
        encoders[name] = ohe
    
    return df, encoders

def one_hot_encode_test_geos(df, name, ohe):
    """This funciton takes in the test dataframe, a feature name and 
    an ohe object and then One hot encodes the feature and adds
    it to the dataframe
    
    Returns the transformed test dataframe
    """
    
    single_feature_df = df[[name]]
    feature_array = ohe.transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0], index=df.index)
    df = df.drop(name, axis=1)
    df = pd.concat([df, ohe_df], axis=1)
    
    return df

def ohe_train_val_test_geos(X_train, X_val, X_test):
    """This function takes in the train, validation and test sets.
    OneHotEncodes the features on train and transforms the 
    validation and test set with extracted encoders"""
    
    X_train, encoders = ohe_all_categorical_geos(X_train)
    
    for key in encoders:
        X_test = one_hot_encode_test_geos(X_test, key, encoders[key])
        X_val = one_hot_encode_test_geos(X_val, key, encoders[key])
    
    return X_train, X_val, X_test

def scaling_numeric_features(X_train, X_val, X_test):
    """
    This function uses StandardScaler object from sklearn.preprocessing
    and standardizes the numeric features in the train, validation and test sets
    fitting on train and transforming train, validation and test features.
    """
    
    #isolate numeric feature column names
    col_names = ['count_floors_pre_eq', 'count_families', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq']
    
    #fit and transform train numeric features
    X_train_scaled = X_train.copy()
    features = X_train_scaled[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)

    X_train_scaled[col_names] = features
    

    #transform validation and test numeric features using the scaler object
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    features = X_val_scaled[col_names]
    features = scaler.transform(features.values)
    X_val_scaled[col_names] = features
    features = X_test_scaled[col_names]
    features = scaler.transform(features.values)
    X_test_scaled[col_names] = features
    
    return X_train_scaled, X_val_scaled, X_test_scaled

