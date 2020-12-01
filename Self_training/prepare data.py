#0) prepare data
import pandas as pd
unlabeled_movies = pd.read_csv('/content/gdrive/My Drive/unlabeled_movie_dataset.csv')
unlabeled_movies = unlabeled_movies[['title', 'genres', 'keywords', 'overview']]
