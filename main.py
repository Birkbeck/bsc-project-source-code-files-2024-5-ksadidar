#importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split


# scikitlearn packages for further statistical learning from the dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, mean_squared_error

#pandera library for schema validation, data checking and integration with ML 
import pandera
from pandera import Column, DataFrameSchema, Check

#creating all the dataframes
df_year = pd.read_csv('data_by_year.csv')
print(df_year.head())

df_genres = pd.read_csv('data_by_genres.csv')
print(df_year.head())

df_artist = pd.read_csv('data_by_artist.csv')
print(df_artist.head())

df_data = pd.read_csv('data.csv')
print(df_data.head())

#creating a function to analyse all the dataframes
def analyse_dataframes(df, df_name):
    print(f"analysingDataframe: {df_name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"First 5 rows:\n{df.head()}")
    print("\n")
    print(df.isnull().sum())
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))

#DATA CLEANING

#checking and filling in for missing values
for col in df_arti
st.columns:
    if df_artist[col].isnull().any():
        print(f"Missing values are found in the df_artist column '{col}'.")
        if pd.api.types.is_numeric_dtype(df_artist[col]):
            df_artist[col].fillna(df_artist[col].mean(), inplace = True)
        else:
            df_artist[col].fillna(df_artist[col].mode()[0], inplace = True)

for col in df_year.columns:
    if df_year[col].isnull().any():
        print(f"Missing values are found in the df_year column '{col}'.")
        if pd.api.types.is_numeric_dtype(df_year[col]):
            df_artist[col].fillna(df_year[col].mean(), inplace = True)
        else:
            df_artist[col].fillna(df_year[col].mode()[0], inplace = True)

for col in df_genres.columns:
    if df_genres[col].isnull().any():
        print(f"Missing values are found in the df_genres column '{col}'.")
        if pd.api.types.is_numeric_dtype(df_genres[col]):
            df_artist[col].fillna(df_genres[col].mean(), inplace = True)
        else:
            df_artist[col].fillna(df_genres[col].mode()[0], inplace = True)

for col in df_data.columns:
    if df_data[col].isnull().any():
        print(f"Missing values are found in the df_data column '{col}'.")
        if pd.api.types.is_numeric_dtype(df_data[col]):
            df_artist[col].fillna(df_data[col].mean(), inplace = True)
        else:
            df_artist[col].fillna(df_data[col].mode()[0], inplace = True)

#removing the duplicates
df_artist.drop_duplicates(inplace = True)
df_year.drop_duplicates(inplace = True)
df_genres.drop_duplicates(inplace = True)
df_data.drop_duplicates(inplace = True)

#stripping off the whitespace and lowercase strings in the object columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip().str.lower()
