import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

yarn = pd.read_csv("~/Desktop/16/CS200B/final_yarn.csv")

yarn_names = yarn['name']


numeric_features = yarn.select_dtypes(include=[np.number]).columns.tolist()

X = yarn[numeric_features]

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

X_processed = preprocessor.fit_transform(X)

nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')

nn_model.fit(X_processed)

def recommend_yarns(query_index, n_recommendations=5):
    distances, indices = nn_model.kneighbors(
        X_processed[query_index].reshape(1, -1),
        n_neighbors=n_recommendations + 1)
    recommended_indices = indices.flatten()[1:]

    return yarn.loc[recommended_indices, ['yarn_company_name', 'name', 'fiber_type_name', 'yarn_weight_name']]

recommend_yarns(query_index=42)

feature_columns = X.columns.tolist()

def recommend_from_preference(user_input, n_recommendations=5):

    user_df = pd.DataFrame(columns=feature_columns)
    user_df.loc[0] = np.nan

    for key, value in user_input.items():
        if key in user_df.columns:
            user_df.at[0, key] = value

    user_processed = preprocessor.transform(user_df)

    distances, indices = nn_model.kneighbors(
        user_processed,
        n_neighbors=n_recommendations)

    return yarn.loc[indices.flatten(), ['yarn_company_name', 'name', 'fiber_type_name', 'yarn_weight_name']]

user_input = {
    'grams': 100,
    'yardage': 200,
    'yarn_weight_wpi': 9,
    'texture': "plied"
    }

recommend_from_preference(user_input)

