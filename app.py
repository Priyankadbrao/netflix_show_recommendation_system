import streamlit as st
import ast
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

df = pd.read_csv("cleaned_data.csv") #cleaned data Csv path

#Multi-encode genre 
df['genres'] = df['genres'].apply(ast.literal_eval)
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)
all_genres = mlb.classes_.tolist()

# Normalize rating
scaler = MinMaxScaler()
df['normalized_rating'] = scaler.fit_transform(df[['imdb_score']])

#One-hot encode type
type_dummies = pd.get_dummies(df['type'], prefix='type')

# One-hot encode age_group 
age_group_dummies = pd.get_dummies(df['age_group'], prefix='age')

X = pd.concat([genre_encoded, df[['normalized_rating']], type_dummies, age_group_dummies], axis=1)
feature_columns = X.columns  

#Streamlit code
st.title("🎥 Smart Recommender System")

st.sidebar.header("Choose Input Type")
input_method = st.sidebar.radio("Recommend by:", ["Title", "User Filters"])


if input_method == "Title":
    selected_title = st.sidebar.selectbox("Select a Title", df["title"].unique())

    if st.sidebar.button("Recommend"):
        idx = df[df["title"] == selected_title].index[0]
        sim_scores = cosine_similarity([X.iloc[idx]], X)[0]
        df["similarity"] = sim_scores
        top_matches = df[df["title"] != selected_title].sort_values(by="similarity", ascending=False).head(5)

        st.subheader("📌 Top Recommendations Based on Title")
        st.table(top_matches[["title", "similarity", "imdb_score", "type", "age_group"]])


else:
    genres_input = st.sidebar.multiselect("Select Genres", options=all_genres)
    user_rating = st.sidebar.slider("Rating (1 to 10)", 1.0, 10.0, 7.0)
    normalized_rating = user_rating / 10.0 
    content_type = st.sidebar.selectbox("Type", df["type"].unique())
    age_group = st.sidebar.selectbox("Age Group", df["age_group"].unique())

    if st.sidebar.button("Get Recommendations"):        
        user_vector = pd.DataFrame([0]*len(feature_columns), index=feature_columns).T
        user_vector["normalized_rating"] = normalized_rating
        
        type_col = f"type_{content_type}"
        if type_col in user_vector.columns:
            user_vector[type_col] = 1
        
        age_col = f"age_{age_group}"
        if age_col in user_vector.columns:
            user_vector[age_col] = 1
       
        for genre in genres_input:
            if genre in user_vector.columns:
                user_vector[genre] = 1

        sim_scores = cosine_similarity(user_vector, X)[0]
        df["similarity"] = sim_scores
        top_matches = df.sort_values(by="similarity", ascending=False).head(5)

        st.subheader("🎯 Top Recommendations Based on Your Preferences")
        st.table(top_matches[["title", "similarity", "imdb_score", "type", "age_group"]])
