#0) read_data from google drive

import pandas as pd

movies = pd.read_csv('/content/gdrive/My Drive/tmdb_5000_movies.csv')
#movies = movies.drop(2656).reset_index(drop = True)
#movies = movies.drop(4139).reset_index(drop = True)
#movies = movies.drop(4339).reset_index(drop = True)
#movies = movies.drop(4398).reset_index(drop = True)
#movies = movies.drop(4427).reset_index(drop = True)
title = []
for i in range(len(movies)):
  title.append(movies['title'][i])

action_movies = []
for j in range(len(movies)):
  if 'Action' in movies['genres'][j]:
    action_movies.append(j)

fantasy_movies = []
for j in range(len(movies)):
  if 'Fantasy' in movies['genres'][j]:
    fantasy_movies.append(j)

comedy_movies = []
for j in range(len(movies)):
  if 'Comedy' in movies['genres'][j]:
    comedy_movies.append(j)

horror_movies = []
for j in range(len(movies)):
  if 'Horror' in movies['genres'][j]:
    horror_movies.append(j)

romance_movies = []
for j in range(len(movies)):
  if 'Romance' in movies['genres'][j]:
    romance_movies.append(j)
