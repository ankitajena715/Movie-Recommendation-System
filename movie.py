What it does: Recommends movies based on a movie selected by the user.

Tech: Python, Pandas, NLP, cosine similarity

code:
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movies.csv")

df["combined"] = (
    df["genre"].fillna("") + " " +
    df["description"].fillna("")
)

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(df["combined"])

similarity = cosine_similarity(matrix)

def recommend(movie):
    index = df[df["title"].str.lower() == movie.lower()].index[0]

    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    for i, score in scores[1:6]:
        print(df.iloc[i]["title"])

recommend("Inception")
