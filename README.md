# recommendation-system-ubcf-
recommendation 

import numpy as np
import pandas as pd

movies=pd.read_csv("E:/sem2/ML/movies.csv")
ratings=pd.read_csv("E:/sem2/ML/ratings.csv")

## finding Unique Ids
ratings['userId'].nunique(),ratings['movieId'].nunique()

## Creating pivot table
ui_matrix=ratings.pivot_table(index='userId',columns='movieId',values='rating')
ui_matrix.head(10)


## Average ratings given by an user
ratings['userId'].value_counts().mean()

## Standardisation of the data¶

## in default when we find mean,it is for the column,so for row

#user_avg=ui_matrix.mean(axis=1)
#user_std=ui_matrix.std(axis=1)

#user_avg,user_std


## applying formula of standardising on the matrix across row and then filling the NA values with zero.

ui_matrix_norm=ui_matrix.apply(lambda v :(v-v.mean())/v.std(),axis=1).fillna(0)

ui_matrix_norm.head(10)

## Cosine Similarity

from sklearn.metrics.pairwise import cosine_similarity

user_sim=pd.DataFrame(cosine_similarity(ui_matrix_norm),index=ui_matrix_norm.index,columns=ui_matrix_norm.index)
user_sim.head()



## Finding the neihbours of any particular user id=10 (for example)

user_Id=10
neighbour=user_sim[user_Id].drop(user_Id).sort_values(ascending=False).head(3)
neighbour

## Just considering the index of the above matrix ,because we dont need the cosine similarity values
neighbour=neighbour.index
neighbour

user_movies=ui_matrix.loc[user_Id]

movies_not_watched=user_movies[pd.isnull(user_movies)]
movies_not_watched.index

neighbour_matrix=ui_matrix_norm.loc[neighbour]
neighbour_matrix

movies_predictions=neighbour_matrix[movies_not_watched.index].mean()
movies_recom=movies_predictions.sort_values(ascending=False).head()
movies_recom=movies_recom.index
movies_recom

## Recomended movies

movies[movies['movieId'].isin(movies_recom)]['title']
