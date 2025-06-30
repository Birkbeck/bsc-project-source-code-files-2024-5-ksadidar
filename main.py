#importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# scikitlearn packages for further statistical learning from the dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
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


## DATA CLEANING

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
                'instrumentalness', 'loudness', 'liveness', 'speechiness', 
                'tempo', 'valence', 'popularity', 'key', 'mode', 'time_period']


#Distributing the numerical values of audioFeatures
availableAudioFeatures = [col for col in audioFeatures if col in df_data.columns if df_data[col].dtype in ['int64', 'float64']]

#setting up one big figure to plot all the graphs in
n=len(availableAudioFeatures)
ncols=4
nrows=(n+ncols-1)//ncols
fig, axes=plt.subplots(nrows,ncols,figsize=(4*ncols,3*nrows))

#drawing eaxh subplot by looping over the features
for ax, col in zip(axes.flatten(), availableAudioFeatures):
        sns.histplot(df_data[col], kde=True,bins=50,ax=ax)
        ax.set_title(col, fontsize=10)
for ax in axes.flatten()[n:]:
    ax.axis('off')
    #layout ajustments and by keeping it outside the loop
fig.suptitle("Distribution of the numerical values of audioFeatures in data.csv", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95])#leaving enuf space for title
plt.show()

#plotting the Popularity Distributions
plt.figure(figsize=(12,8))
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
plt.figure(figsize=(12,8))
sns.heatmap(corrMatrrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of AudioFeatures against Popularity")
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

## EDA of data_by_year, to find trend over time ##

print("\n-- Data by YEAR from the dataset data_by_year.csv --")
print(f"Shape: {df_year.shape}")
print(df_year.head)

#plotting key audioFeatures to see trends over the years
keyAudioFeatures = ['popularity', 'danceability', 'loudness', 'energy', 'tempo', 'valence', 'acousticness', 'speechiness', 'liveness']

#prepare a large canvas for all the subplots
n=len(keyAudioFeatures)
ncols=3
nrows=(n+ncols-1)//ncols
fig,axes=plt.subplots(nrows,ncols,figsize=(12,8), sharex=True)
axes=axes.flatten()

#loop over the keyAudioFeatures to populate each subplot
for ax, feature in zip(axes, keyAudioFeatures):
    if feature in df_year.columns:
        sns.lineplot(x='year', y=feature, data=df_year,ax=ax)
        ax.set_title(f"{feature.capitalize()} Over Years")
        ax.grid(alpha=0.3)
for ax in axes[n:]:
    ax.axis('off')#turning off unused subplots
fig.tight_layout(rect=[0,0,1,0.92])#adjusting the layout
fig.suptitle("Trends of KeyAudioFeatures & Popularity Over the Years", y=0.98, fontsize=16)
plt.show()#showing all the subplots in one large canvas

#E# DA on the Genre characteristics, data_by_genre.csv ##

print("\n-- Data by GENRE from the dataset data_by_genres.csv --")
print(f"Shape: {df_genres.shape}")
print(df_genres.head)
print(f"\nNumber of Unique Genres: {df_genres['genres'].nunique()}")

#creating a small batch of genres for simplicity
sampleGenres = df_genres.sample(n=min(10, len(df_genres)), random_state=42)

#picking audio features for genre plotting
plotFeatures=[f for f in audioFeatures 
if (f in sampleGenres.columns 
and df_genres[f].dtype in ['float64', 'int64'] 
#filter the audio features with high variances to make the data plot ready
and f not in ['key', 'mode', 'time_signature', 'duration_ms', 'tempo', 'loudness'])]

#scaling eaxh column to [0,1]
scaled=sampleGenres.copy()
for feat in plotFeatures:
    col=scaled[feat]
    scaled[feat]=(col-col.min())/(col.max()-col.min())

#melting genres dataframe before plotting
meltedGenres=scaled.melt(id_vars=['genres'], value_vars=plotFeatures, var_name='features', value_name='value')
plt.figure(figsize=(12,8))
sns.barplot(x='features', y='value', hue='genres', data=meltedGenres)
plt.title('Comparison of *Normalised* audioFeatures from a sample of GENRES')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.15, 1), loc=2)
plt.tight_layout()
plt.show()




#EDA on the ARTIST characteristics, data_by_artist.csv
print("\n-- Data by GENRE from the dataset data_by_artist.csv --")
print(f"Shape: {df_artist.shape}")
print(df_artist.head)
print(f"\nNumber of Unique Artists: {df_artist['artists'].nunique()}")

#distribution of artist popularity
if 'popularity' in df_artist.columns:
    plt.figure(figsize=(10, 8))
    sns.histplot(df_artist['popularity'], kde=True, bins=50)
    plt.title('Distribution of Artist popularity from the artist dataset')
    plt.xlabel('Mean Popularity')
    plt.show()

#top 10 artists by Popularity
if 'popularity' in df_artist.columns:
    print("\nTop 10 artists by Mean Popularity from the Artist dataset")
    print(df_artist.nlargest(10, 'popularity')[['artists', 'popularity', 'count']])

#scatterPlot showing energy against danceability 

if 'energy' in df_artist.columns and 'danceability' in df_artist.columns:

    #we create a sample of artists to avoid overplotting
    sampleOfArtist = (df_artist.sample(n=min(5000, len(df_artist)), random_state=1) if len(df_artist) > 0 else pd.DataFrame())
    if sampleOfArtist.empty:
        print("there is not enough artist data available")
    else:
        #setting up the big hexbin fig
        plt.figure(figsize=(10,8))
        hexBin=plt.hexbin(x=sampleOfArtist['danceability'], y=sampleOfArtist['energy'], gridsize=40,cmap='Blues',mincnt=1)
        countBin=plt.colorbar(hexBin)
        countBin.set_label('count in bin')
        plt.title("Hexbin: Artist Energy versus Danceability")
        plt.xlabel("Mean Danceability")
        plt.ylabel("Mean Energy")
        plt.tight_layout()#for tidying up
        plt.show()
else:
    print("data_by_artist.csv is empty or the columns we are looking for do not exist")

## RECOMMENDER SETUP ##

#data preparation for the recommender
if not df_data.empty and 'popularArtists' in df_data.columns:
    df_recommender=df_data.sample(frac=0.2, random_state=42)#use only 20% of the main dataset
    print(f"recommender dataset shape: {df_recommender.shape}")

    #selecting features for content-based filtering
    contentFeatures=['acousticness', 'danceability', 'energy', 'instrumentalness', 
    'liveness', 'loudness', 'tempo', 'speechiness', 'valence', 'duration_ms']

    #checking presence of these features in dataset
    contentFeatures=[f for f in contentFeatures if f in df_recommender.columns]
    if not contentFeatures:
        print("no content features available in the recommender dataframe")
        #dummy df recommender features
        dfRecommenderFeatures=pd.DataFrame()
    else:
        recommenderCols=['id', 'name', 'popularArtists', 'popularity'] + contentFeatures
        dfRecommenderFeatures=df_recommender[recommenderCols].copy()

        dfRecommenderFeatures.dropna(subset=contentFeatures, inplace=True)
        print(f"shape after dropping NA from contentFeatures: {dfRecommenderFeatures}")

        scaler=MinMaxScaler()
        dfRecommenderFeatures[contentFeatures]=scaler.fit_transform(dfRecommenderFeatures[contentFeatures])
        print("\nprocessed features for recommender {first 5 rows}:")
        print(dfRecommenderFeatures.head())
else:
    print("main datset is empty or popular artists column was not created")
    print(dfRecommenderFeatures)==pd.DataFrame()


##BUILDING the RECOMMENDER function

if not dfRecommenderFeatures.empty and 'id' in dfRecommenderFeatures.columns:
    itemsFeatureMatrix=dfRecommenderFeatures.set_index('id')[contentFeatures] #item feature matrix for cosine similarity
    cosineSimilarityMatrix = cosine_similarity(itemsFeatureMatrix) #calculating the cosine similarity
    #mapping the song index
    songId_to_index=pd.Series(itemsFeatureMatrix.index)
    index_to_songId=pd.Series(itemsFeatureMatrix.index.values, index=range(len(itemsFeatureMatrix.index)))
    print(f"cosine similarity matrix shape: {cosineSimilarityMatrix.shape}")

#generating hybrid recommendations for a given song id 
    def hybreedRecommender(song_id, N_content=50, K_final=10):
        if song_id not in songId_to_index.values:
            return f"Song ID {song_id} not found in the recommender dataset"
        songIndex=songId_to_index[songId_to_index == song_id].index[0] #getting index of song in the sim Matrix 
        similarityScores=list(enumerate(cosineSimilarityMatrix[songIndex])) #getting sim score for songs
        similarityScores=sorted(similarityScores,key=lambda x: x[1], reverse=True) #sorting songs based on sim score
        
        #getting the top N_content, most similar songs 
        topContentSongIndices=[i[0]for i in similarityScores[1:N_content+1]]
        topContentSongIds=index_to_songId[topContentSongIndices].tolist()

        #retrieving details of the recomnd songs + sim score
        recommendations= []
        for i, score_tuple in enumerate(similarityScores[1:N_content+1]):
            idx, score = score_tuple
            songDetails=dfRecommenderFeatures[dfRecommenderFeatures['id']==index_to_songId[idx]]
            if not songDetails.empty:
                recommendations.append({
                    'id': songDetails['id'].iloc[0],
                    'name': songDetails['name'].iloc[0],
                    'popularArtists': songDetails['popularArtists'].iloc[0],
                    'similarityScores': score,
                    'popularity': songDetails['popularity'].iloc[0]
                })

        recFunction=pd.DataFrame(recommendations)
        # re ranking songs by popularity
        recFunctionReranked=recFunction.sort_values(by='popularity', ascending=False)
        return recFunctionReranked.head(K_final)
    print("recommender function defined")
else:
    print("recommender features are empty,skipping recommendation!")

##SAMPLE RECOMMENDATION##

if not dfRecommenderFeatures.empty and 'hybreedRecommender' in locals() and len (dfRecommenderFeatures) >10:
    sampleSongs=dfRecommenderFeatures.sample(3, random_state=123)

    for idx, songRows in sampleSongs.iterrows():
        songIdtoRecommend=songRows['id']
        songNames=songRows['name']
        songArtist=songRows['popularArtists']
        print(f"\n~~~ Recommendations for: '{songNames}' by {songArtist} (ID: {songIdtoRecommend}) ~~~")

        recomendedSongs=hybreedRecommender(songIdtoRecommend, N_content=50, K_final=5)
        if isinstance(recomendedSongs, str):
            print(recomendedSongs)
        elif not recomendedSongs.empty:
            print(recomendedSongs[['name', 'popularArtists', 'similarityScores', 'popularity']])
        else:
            print("recommendation cannot be found")



##EVALS OF THE RECOMMENDER##

#creating a small test set og songs
if not dfRecommenderFeatures.empty and 'hybreedRecommender' in locals() and 'itemsFeatureMatrix' in locals() and len(dfRecommenderFeatures)>50: #ensuring enuf data for testing
    testSongIDs=dfRecommenderFeatures['id'].sample(n=min(50, len(dfRecommenderFeatures)-1), random_state=77).tolist()#ensuring songs are in itemFeatMatrix

    allMeanSimilarities=[]
    allMeanPopularities=[]
    print(f"evaluating the score on {len(testSongIDs)} test songs")

    for songIDtest in testSongIDs:
        recommendationsDF=hybreedRecommender(songIDtest, N_content=50, K_final=10)

        #Mean cosine similarity calc
        if isinstance(recommendationsDF, pd.DataFrame) and not recommendationsDF.empty:
            inputSongVector=itemsFeatureMatrix.loc[songIDtest].values.reshape(1, -1)#grabbing feature vector of input song
            recommendedIDs=recommendationsDF['id'].tolist()#grabbing feature vectors of recommended songs
            validRecommendedIDs=[rid for rid in recommendedIDs if rid in itemsFeatureMatrix.index]#filtering item featue matrix
            if validRecommendedIDs:
                recommendedVectors=itemsFeatureMatrix.loc[validRecommendedIDs].values
                #calculating cosine similarities between input and recommended songs
                similaritiesToInput=cosine_similarity(inputSongVector, recommendedVectors)[0]
                allMeanSimilarities.append(np.mean(similaritiesToInput)) 

        #Mean Popularity of Recommendations
        allMeanPopularities.append(recommendationsDF['popularity'].mean())
    
    #calculating the Total average Metrics
    totalMeanCosineSimilarity=np.mean(allMeanSimilarities) if allMeanSimilarities else 0
    totalMeanPopularityofRecs=np.mean(allMeanPopularities) if allMeanPopularities else 0

    print(f"\n~~~~  EVALS OUTPUT ~~~~")
    print(f"Number of Test Songs: {len(testSongIDs)}")
    print(f"Number of recommendations generated per song for eval:10")
    print(f"Mean Cosine Similarity of recommendations to input song:{totalMeanCosineSimilarity:.4f}")
    print(f"Mean Popularity of recommended songs: {totalMeanPopularityofRecs:.2f}")

    #calc average popularity in the recs datset
    avgPopsInRecDataset=dfRecommenderFeatures['popularity'].mean()
    print(f"average popoularity in recommender datset: {avgPopsInRecDataset:.2f}")
    print("TASK COMPLETED, END OF RS-OUTPUT!")
else:
    print("evals cant be performd")