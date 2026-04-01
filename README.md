Scenario 1 – User-Based Collaborative Filtering

Dataset (Kaggle – Public):
MovieLens Dataset Dataset Link: https://www.kaggle.com/datasets/grouplens/movielens-100k
(You can pick any dataset if required)

This project implements a Movie Recommendation System using User-Based Collaborative Filtering, where movies are recommended to a user based on similar users’ preferences. The MovieLens 100K dataset from Kaggle is used for this scenario, which includes User ID, Movie ID, Ratings, and Timestamp as input features. The process begins with importing the required Python libraries and loading the dataset, followed by data inspection and preprocessing. A User–Item matrix is created to represent user ratings, and missing values are handled by filling them with 0. Similarity between users is then calculated using cosine similarity, and the top similar users are identified. Based on these similar users, ratings for unseen movies are predicted and Top-N movie recommendations are generated for a given user. The model performance is evaluated using RMSE (Root Mean Square Error) and MAE (Mean Absolute Error). The analysis includes studying how similarity affects recommendations, analyzing sparsity in the user-item matrix, and evaluating recommendation quality for different users. Visualizations such as heatmap of the user-item matrix, user similarity matrix, and top recommended movies are also included.

Scenario 2 – Item-Based Collaborative Filtering

Dataset (Same / Alternative Dataset):
MovieLens Dataset / Amazon Product Dataset

This project also implements Item-Based Collaborative Filtering, where recommendations are generated based on similar movies instead of similar users. The same MovieLens dataset (or an alternative dataset like Amazon product data) is used for this scenario. The dataset is loaded and an Item–User matrix is created to represent ratings. Similarity between items is calculated using cosine similarity and Pearson correlation, and the top similar movies are identified. Based on the user’s previous movie history, similar movies are recommended. The results of item-based recommendations are then compared with user-based recommendations to understand which method performs better. The model is evaluated using RMSE and Precision@K. The analysis includes comparing recommendation accuracy with Scenario 1, identifying popular and niche items, and studying the scalability of the item-based approach. Visualizations such as item similarity heatmap, top similar movies graph, and recommendation comparison charts are also included.
# 24ADI003_24BAD096_RAMYA-R_EX_9
