#importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, mean_squared_error

#preparing all the dataframes
df_year = pd.read_csv('data_by_year.csv')
print(df_year.head())

df_genres = pd.read_csv('data_by_genres.csv')
print(df_year.head())

df_artist = pd.read_csv('data_by_artist.csv')
print(df_artist.head())

df_data = pd.read_csv('data.csv')
print(df_data.head())

#create a function to analyse the dataframes

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