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
#import pandera
#from pandera import Column, DataFrameSchema, Check

#ast for safe evaluation of string representation of dicts and lists
import ast

#creating all the dataframes
df_year = pd.read_csv('data_by_year.csv')
print(df_year.head())

df_genres = pd.read_csv('data_by_genres.csv')
print(df_genres.head())

df_artist = pd.read_csv('data_by_artist.csv')
print(df_artist.head())

df_data = pd.read_csv('data.csv')
print(df_data.head())


#DATA CLEANING

#checking and filling in for missing values
for col in df_artist.columns:
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

#removing redundant records
df_artist.drop_duplicates(inplace = True)
df_year.drop_duplicates(inplace = True)
df_genres.drop_duplicates(inplace = True)
df_data.drop_duplicates(inplace = True)


#cleaning all the dataframes
dataframes = {
    'main': df_data,
    'year': df_year,
    'artist': df_artist,
    'genre':df_genres
}

#stripping off the whitespace and lowercase strings in the object columns
for name, df in dataframes.items():
    for col in df.select_dtypes(include='object').columns:
        df[col]=df[col].astype(str).str.strip().str.lower()
    print(f"Cleaned {name} DataFrame")



#EDA ~ explore through each dataset to see contents & get insights 


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

#create a list of audio features from observaytions of data.csv for easier feature selection
audioFeatures = ['acoustisness', 'danceability', 'duration_ms', 'energy', 
                'instrumentalness', 'loudness', 'liveness', 'speechiness', 'tempo', 'valence', 
                'popularity', 'duration_ms', 'key', 'mode', 'time_period']


#Distributing the numerical values of audioFeatures

#checking if the features are available in the dataset
availableAudioFeatures = [col for col in audioFeatures if col in df_data.columns]

#plotting the features
for i, col in enumerate(availableAudioFeatures):
    if df_data[col].dtype in ['int64', 'float64']:
        plt.subplot(4,4, i+1)
        sns.histplot(df_data[col], kde=True,bins=50)
        plt.title(col)
    plt.tight_layout
    plt.suptitle("Distribution of the numerical values of audioFeatures in data.csv")
    plt.show()

#plotting the Popularity Distributions
plt.figure(figsize=(11,7))
sns.histplot(df_data['popularity'], kde=True, bins=50)
plt.title('Distribution of song popularity in the dataset')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()
print(f"Popularity Stats:\n{df_data['popularity'].describe()} ")

#CORRELATION of audioFeatures & Popularity
correlationFeature = audioFeatures+['popularity']
correlationFeature = [f for f in correlationFeature if f in df_data.columns and df_data[f].dtype in ['int64', 'float64']]
corrMatrrix = df_data[correlationFeature].corr()
plt.figure(figsize=(14,10))
sns.heatmap(corrMatrrix, annot=True, cmap='coolwarm')
plt.title("CorrelationMatrix of AudioFeatures against Popularity")
plt.show()

#PARSING 'artist' column to create a list of Populart artists
df_data['parsedArtists']=df_data['artists'].apply(lambda x: ast.literal_eval(x))
df_data['popularArtists']=df_data['parsedArtists'].apply(lambda x: x[0] if isinstance(x, list) and len(x)> 0 else None)
print("successfully parsed artists column and made a list of the most popular artist")
print("\nTop10 most played popularArtists:")
print(df_data['popularArtists'].value_counts().nlargest(10))


#plotting it in a graph
Popular_artists = df_data['popularArtists'].value_counts().nlargest(10)
plt.figure(figsize=(12,8))
sns.barplot(x=Popular_artists.values, y= Popular_artists.index, palette='viridis')
plt.title('Top10 most played popularArtists', fontsize=12)
plt.xlabel('number of tracks')
plt.ylabel('Artist')
for i, value in enumerate(Popular_artists.values):
    plt.text(value +2, i, str(value), va='center')
plt.tight_layout()
plt.show()

#EDA of data_by_year, to find trend over time
print("\n-- Data by YEAR from the dataset data_by_year.csv --")
print(f"Shape: {df_year.shape}")
print(df_year.head)

#plotting key audioFeatures to see trends over the years
keyAudioFeatures = ['popularity', 'danceability', 'loudness', 'energy', 'tempo', 'valence', 'acousticness', 'speechiness', 'liveness']
plt.figure(figsize=(17,14))
for i, feature in enumerate(keyAudioFeatures):
    if feature in df_year.columns:
        plt.subplot(5,4,i+1)
        sns.lineplot(x='year', y=feature, data=df_year)
        plt.title('Audio Features over the years')
    plt.tight_layout()
    plt.suptitle("Trends of KeyAudioFeatures & Popularity Over the Years", y=1.02, fontsize=14)
    plt.show()

#EDA on the Genre characteristics, data_by_genre.csv
print("\n-- Data by GENRE from the dataset data_by_genres.csv --")
print(f"Shape: {df_genres.shape}")
print(df_genres.head)
print(f"\nNumber of Unique Genres: {df_genres['genres'].nunique()}")

#creating a small batch of genres for simplicity
sampleGenres = df_genres.sample(n=min(10, len(df_genres)), random_state=42)

#plotting the sample genres
plt.figure(figsize=(17,14))
plotFeatures=[f for f in audioFeatures 
if f in sampleGenres.columns and 
df_genres[f].dtype in ['float64', 'int64'] 
#filter the audio features with high variances to make the data plot ready
and f not in ['key', 'mode', 'time_period', 'duration_ms', 'tempo', 'loudness']]

#melt genres dataframe for easier plotting 
if plotFeatures:
    meltedGenres = sampleGenres.melt(id_vars=['genres'], value_vars=plotFeatures, var_name='features', value_name='value')
    sns.barplot(x='features', y='value', hue='genres', data=meltedGenres)
    plt.title('Comparison of audioFeatures from a sample of GENRES')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.15, 1), loc=2)
    plt.tight_layout()
    plt.show()


#PIPELINE for further Genre Analysis using TSNE
#def_genresClean=df_genres.dropna(subset=['genres', 'danceability', 'energy', 'loudness'])

#def_genresSummary=def_genresClean.groupby('genres').agg(
 #   mean_danceability=('danceability', 'mean'), 
  #  max_energy=('energy', 'max'),
   # stddev_loudness=('loudness', 'std')).reset_index()

#dropping rows with NaNs
#def_genresSummary.dropna(inplace=True)

#separating the genre names
#genre_names=def_genresSummary['genres']
#X = def_genresSummary.drop(columns=['genres'])

#normalising the features
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

#dimensionality reduction for visualisation using  t-SNE
#tsne = TSNE(n_components=2,perplexity=30,n_iter=100,random_state=42)
#X_tsne = tsne.fit_transform(X_scaled)

#combining genre names for visualisation
#df_tsne=pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
#df_tsne['genres']=genre_names

#plotting
#plt.figure(figsize=(14,10))
#sns.scatterplot(data=df_tsne, x='Dim1',y='Dim2',hue='genres',palette='tab10',s=100, legend=False)
#plt.title("t-sne projection of Genres based on audioFeatures")
#plt.show()



#EDA on the ARTIST characteristics, data_by_artist.csv
print("\n-- Data by GENRE from the dataset data_by_artist.csv --")
print(f"Shape: {df_artist.shape}")
print(df_artist.head)
print(f"\nNumber of Unique Artists: {df_artist['artists'].nunique()}")

#distribution of artist popularity
if 'popularity' in df_artist.columns:
    plt.figure(figsize=(10, 7))
    sns.histplot(df_artist['popularity'], kde=True, bins=50)
    plt.title('Distribution of Artist popularity from the artist dataset')
    plt.xlabel('Mean Popularity')
    plt.show()

#top 10 artists by Popularity
if 'popularity' in df_artist.columns:
    print("\nTop 10 artists by Mean Popularity from the Artist dataset")
    print(df_artist.nlargest(10, 'popularity')[['artists', 'popularity', 'count']])

