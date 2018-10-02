
import pandas as pd 
import numpy as np 
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer

#read file u.user
u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./u.user', sep='|', names=u_cols, encoding='latin-1')
#print(users[:3])

# read 2 file ua.base and ua.test  
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('./ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('./ua.test', sep='\t', names=r_cols, encoding='latin-1')

#print(ratings_base[:300])
rate_train = ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()
print(rate_train[:100])
print(rate_test[:100])
# read file u.item 
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('./u.item', sep='|', names=i_cols,
 encoding='latin-1')
X0 = items.as_matrix()
X_train_counts = X0[:, -19:]
#print(X_train_counts[:2])

# create feature vector by tfidf
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()
#print(tfidf.shape)

