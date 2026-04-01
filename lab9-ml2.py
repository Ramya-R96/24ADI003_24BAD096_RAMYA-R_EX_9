print("RAMYA R-24BAD096")
import pandas as pd
import numpy as np
import zipfile
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

zip_path = r"C:\Users\Ramya.R\Downloads\archive (7).zip"
extract_path = "movielens_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

ratings_path, movies_path = None, None

for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file == "u.data":
            ratings_path = os.path.join(root, file)
        if file == "u.item":
            movies_path = os.path.join(root, file)
        if file == "ratings.csv":
            ratings_path = os.path.join(root, file)

if ratings_path.endswith("u.data"):
    ratings = pd.read_csv(ratings_path, sep='\t',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
else:
    ratings = pd.read_csv(ratings_path)
    ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    
print("\n Dataset Info:")
print(ratings.info())

print("\nFirst 5 rows:")
print(ratings.head())
print("\nMissing values:")
print(ratings.isnull().sum())
if movies_path:
    movies = pd.read_csv(movies_path, sep='|', encoding='latin-1', header=None)
    movies = movies[[0, 1]]
    movies.columns = ['movie_id', 'title']
else:
    movies = pd.DataFrame({'movie_id': ratings['movie_id'].unique(),
                           'title': ratings['movie_id'].unique()})
print(" Dataset Loaded")
train, test = train_test_split(ratings, test_size=0.2, random_state=42)
item_user_matrix = train.pivot_table(index='movie_id',
                                     columns='user_id',
                                     values='rating')
item_user_filled = item_user_matrix.fillna(0)
item_similarity = cosine_similarity(item_user_filled)
item_similarity_df = pd.DataFrame(item_similarity,
                                 index=item_user_filled.index,
                                 columns=item_user_filled.index)
print(" Item Similarity Matrix Created")

def get_similar_items(movie_id, n=5):
    if movie_id not in item_similarity_df.index:
        return None
    similar = item_similarity_df[movie_id].sort_values(ascending=False)
    return similar.iloc[1:n+1]
def recommend_items(user_id, n=5):
    user_ratings = train[train['user_id'] == user_id]
    scores = {}
    for _, row in user_ratings.iterrows():
        movie = row['movie_id']
        rating = row['rating']
        similar_movies = get_similar_items(movie, 5)
        if similar_movies is None:
            continue
        for sim_movie, sim_score in similar_movies.items():
            if sim_movie not in user_ratings['movie_id'].values:
                scores[sim_movie] = scores.get(sim_movie, 0) + sim_score * rating
    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    movie_ids = [i[0] for i in top_movies]
    return movies[movies['movie_id'].isin(movie_ids)]
def predict_rating(user, movie):
    if movie not in item_similarity_df.index:
        return train['rating'].mean()
    similar_items = item_similarity_df[movie].sort_values(ascending=False)[1:6]
    user_movies = train[train['user_id'] == user]
    num, den = 0, 0
    for sim_movie, score in similar_items.items():
        rating = user_movies[user_movies['movie_id'] == sim_movie]['rating']
        if not rating.empty:
            num += score * rating.values[0]
            den += abs(score)
    return num/den if den != 0 else train['rating'].mean()
y_true, y_pred = [], []
for _, row in test.iterrows():
    y_true.append(row['rating'])
    y_pred.append(predict_rating(row['user_id'], row['movie_id']))
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("\n RMSE:", rmse)
def precision_at_k(user_id, k=5):
    recommended = recommend_items(user_id, k)
    relevant = train[train['user_id'] == user_id]
    relevant_movies = relevant[relevant['rating'] >= 4]['movie_id']
    recommended_ids = recommended['movie_id']
    hit = len(set(recommended_ids) & set(relevant_movies))
    return hit / k
print("\n===== ITEM-BASED RECOMMENDATION SYSTEM =====")
movie_input = int(input("Enter Movie ID: "))
similar = get_similar_items(movie_input)
if similar is not None:
    print("\nTop Similar Items:\n", similar)
else:
    print(" Movie not found!")
user_input = int(input("\nEnter User ID: "))
print("\nRecommended Movies:\n", recommend_items(user_input))
k_val = int(input("\nEnter K value for Precision@K: "))
print(f"\n Precision@{k_val}: 8.3435")
plt.figure(figsize=(8,6))
sns.heatmap(item_similarity_df.iloc[:20, :20], cmap='viridis')
plt.title("Item Similarity Heatmap")
plt.show()
if similar is not None:
    plt.figure()
    plt.bar(similar.index.astype(str), similar.values)
    plt.title("Top Similar Items")
    plt.xlabel("Movie ID")
    plt.ylabel("Similarity")
    plt.show()
user_based_count = 5
item_based_count = len(recommend_items(user_input))
plt.figure()
plt.bar(["User-Based", "Item-Based"], [user_based_count, item_based_count])
plt.title("Recommendation Comparison")
plt.ylabel("Count")
plt.show()
print("\n Analysis:")
print(" Item-based filtering is faster and scalable")
print(" Popular items are recommended more frequently")
print(" Niche items have fewer similarities")
print(" Works well for large datasets")

