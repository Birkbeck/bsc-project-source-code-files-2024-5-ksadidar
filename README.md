finalYearProject
BSc Data Science Final Year Project: Music Recommendation System

This gitRepo provides tools for exploratory data analysis and a music recommendation system based on song audio features. 
The project includes a Python script(main.py) for end-to-end pipeline execution and a Jupyter notebook(recommender.ipynb) for interactive exploration.
There is a requirements.txt that enlists the version of packages that were used for this project.


 PROJECT STRUCTURE
.
├── main.py                 # main sourceCode to perform EDA and run recommendation pipeline
├── recommender.ipynb       # Interactive Jupyter notebook for experimentation
├── data.csv                # Main dataset of songs with audio features
├── data_by_year.csv        # Aggregated feature trends by year
├── data_by_genres.csv      # Aggregated feature trends by genre
├── data_by_artist.csv      # Aggregated feature trends by artist
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation


DEPENDENCIES
~
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn

the above dependencies can be installed using pip: pip instll -r requirements.txt


PROJECT SOURCECODE USAGE GUIDELINES
~
> place the csv dataset files within the project root directory
> run the sourcecode main.py to perform detailed eda on the datasets, generate visualisations, build the recommender function to generate recommendations, perform evals and print them 
> use the Notebook recommender.ipynb to view intermediate results and visuals with ease


SOURCECODE FILE DESCRIPTIONS

> main.py: this is the primary source code for the recommender that contains necessary codebase to perform EDA, popularity Re-ranking based recommendations and evaluations of the recommendations
> recommender.ipynb: the python notebook for the same source code, to be used for going through EDA output, visualisations and recommender pipeline output, the notebook allows to perform further ex[eriments, if required 
