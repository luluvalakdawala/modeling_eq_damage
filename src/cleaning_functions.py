import numpy as np
import pandas as pd


def load_raw_files():
    """This function loads the raw csvs as dataframes
    """
    
    structure_df = pd.read_csv('../../data/csv_building_structure.csv')
    ownership_df = pd.read_csv('../../data/csv_building_ownership_and_use.csv')
    ward_df = pd.read_csv('../../data/ward_vdcmun_district_name_mapping.csv')
    
    return structure_df, ownership_df, ward_df


def clean_text(row):
    """This function replaces characters in the text data 
    with underscores to make the data more pythonic
    """
    
    return row.lower().replace('-', '_').replace('/', '_or_').replace(' ', '_')

def cleaned_dataframe():
    """This function loads the raw data relating to structure, ownership and ward information as dataframes. 
    - It then merges the structure and ownership dataframes.
    
    - Transforms the damage_grade column as target with ordinal values 1,2,3
    
    - Drops a few columns not required for modeling purposes
    
    - Removes Nans from the dataset
    
    - Cleans the text data in categorical features
    
    >>>Returns a single dataframe fit for preprocessing/modeling
    """

    structure_df, ownership_df, ward_df = load_raw_files()

    data = pd.merge(structure_df, ownership_df, on=['building_id', 'district_id', 'vdcmun_id', 'ward_id'])

    data['target'] = data.damage_grade.map({'Grade 1': 1, 'Grade 2': 1, 'Grade 3': 2, 'Grade 4': 3, 'Grade 5': 3})

    data.drop(['count_floors_post_eq', 'height_ft_post_eq', 'damage_grade', 'condition_post_eq',
               'technical_solution_proposed'], axis=1, inplace=True)

    data = data.set_index('building_id')
    
    data.dropna(inplace=True)

    categorical_feats = data.dtypes[data.dtypes == 'object'].index

    
    for feat in categorical_feats:
        data[feat] = f'{feat}_' + data[feat].map(clean_text)
        
    return data


def structure_df_unique_values(feature):
    structure_df, ow_df, wd_df = load_raw_files()
    print(f'Unique values for {feature} are : ', structure_df[feature].unique())
    print(f'Number of nulls for {feature} are : ', structure_df[feature].isna().sum())
    print('===========================================================================')

def ownership_unique_values(feature):
    st_df, ownership_df, wd_df = load_raw_files()
    print(f'The unique values for {feature} : ', ownership_df[feature].unique())
    print(f'Number of nulls for {feature} are : ', ownership_df[feature].isna().sum())
    print('==============================================================')
    
def cleaned_dataframe_for_mvp():
    """This function loads the raw data relating to structure, ownership and ward information as dataframes. 
    - It then merges the structure and ownership dataframes.
    
    - Transforms the damage_grade column as target with ordinal values 1,2,3
    
    - Drops a few columns not required for modeling purposes
    
    - Removes Nans from the dataset
    
    - Cleans the text data in categorical features
    
    >>>Returns a single dataframe fit for preprocessing/modeling
    """

    structure_df, ownership_df, ward_df = load_raw_files()

    data = pd.merge(structure_df, ownership_df, on=['building_id', 'district_id', 'vdcmun_id', 'ward_id'])

    data['target'] = data.damage_grade.map({'Grade 1': 1, 'Grade 2': 1, 'Grade 3': 2, 'Grade 4': 3, 'Grade 5': 3})

    data.drop(['count_floors_post_eq', 'height_ft_post_eq', 'damage_grade', 'condition_post_eq',
               'technical_solution_proposed', 'vdcmun_id', 'ward_id'], axis=1, inplace=True)

    data = data.set_index('building_id')
    
    data.dropna(inplace=True)

    categorical_feats = data.dtypes[data.dtypes == 'object'].index

    
    for feat in categorical_feats:
        data[feat] = f'{feat}_' + data[feat].map(clean_text)
        
    return data   
