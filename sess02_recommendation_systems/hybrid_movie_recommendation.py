# Python script to use a Hybrid Recommendation system to recommend movies to
# users on streaming services like Netflix or Amazon Prime
# It combines content-based & collaborative filtering to address / improve on
# the failings of the other

# Import the required modules
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

data = {
    "title": [
        "The Matrix",
        "John Wick",
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Avengers: Endgame",
        "Titanic",
        "The Notebook",

        "Gladiator",
        "The Shawshank Redemption",
        "Forrest Gump",
        "The Godfather",
        "Pulp Fiction",
        "The Lion King",
        "Jurassic Park",
        "The Silence of the Lambs",
        "Frozen",
        "The Social Network",
        "Mad Max: Fury Road",
        "La La Land",
        "Get Out",
        "The Wolf of Wall Street",
        "Harry Potter and the Sorcerer's Stone"
    ],
    "genre": [
        "Action Sci-Fi",
        "Action Thriller",
        "Sci-Fi Thriller",
        "Sci-Fi Drama",
        "Action Crime",
        "Action Superhero",
        "Romance Drama",
        "Romance Drama",

        "Action Drama",
        "Drama",
        "Drama Romance",
        "Crime Drama",
        "Crime Drama",
        "Animation Family",
        "Adventure Sci-Fi",
        "Thriller Crime",
        "Animation Musical",
        "Drama Biography",
        "Action Adventure",
        "Romance Musical",
        "Horror Thriller",
        "Biography Comedy Drama",
        "Fantasy Adventure"
    ],
    "keywords": [
        "virtual reality hacker AI",
        "assassin revenge crime",
        "dream subconscious mind",
        "space time blackhole",
        "joker vigilante batman",
        "superheroes time travel",
        "ship iceberg love",
        "love relationship drama",

        "roman empire revenge gladiator",
        "prison escape hope friendship",
        "life journey love destiny",
        "mafia family power crime",
        "nonlinear crime violence",
        "lion king africa destiny",
        "dinosaurs park science chaos",
        "serial killer investigation FBI",
        "ice powers sister love",
        "facebook startup ambition betrayal",
        "postapocalyptic desert survival",
        "music love dreams hollywood",
        "racism hypnosis horror",
        "stock market greed corruption",
        "wizard magic school friendship"
    ],
    "overview": [
        "A hacker discovers reality is a simulation.",
        "An ex-hitman seeks revenge.",
        "A thief enters dreams to steal secrets.",
        "Explorers travel through a wormhole in space.",
        "Batman faces the Joker in Gotham.",
        "Heroes unite to reverse a catastrophe.",
        "A love story aboard a doomed ship.",
        "A romantic story about enduring love.",

        "A betrayed general seeks revenge in ancient Rome.",
        "Two imprisoned men bond over years and find redemption.",
        "A man's extraordinary life journey unfolds through decades.",
        "The aging patriarch transfers control of his empire to his son.",
        "Interwoven stories of crime in Los Angeles.",
        "A lion prince flees and returns to reclaim his kingdom.",
        "Scientists recreate dinosaurs in a theme park gone wrong.",
        "An FBI trainee hunts a brilliant but twisted serial killer.",
        "A princess with ice powers struggles to control her abilities.",
        "The rise of Facebook and the conflicts behind it.",
        "Survivors race across a desert in a post-apocalyptic world.",
        "Two artists pursue love and dreams in Los Angeles.",
        "A young man uncovers a disturbing secret about his girlfriend's family.",
        "A stockbroker rises and falls amid greed and excess.",
        "A young wizard begins his journey at a magical school."
    ]
}

# -------------------------------------------------------------------------------------------
# Step 1. Create a Pandas Dataframe
# -------------------------------------------------------------------------------------------
df = pd.DataFrame(data)

df['content'] = (
    df['genre'] + " " +
    df['keywords'] + " " +
    df['overview']
)

# -------------------------------------------------------------------------------------------
# Step 2. Content based filtering
# -------------------------------------------------------------------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute similarity between movies
content_similarity = cosine_similarity(tfidf_matrix)

# -------------------------------------------------------------------------------------------
# Step 3. Collaborative filtering
# -------------------------------------------------------------------------------------------
# Generate dummy users and movie ratings (rows -> users, columns -> movies)
ratings_data = {
    "User1": [5, 4, 5, 5, 4, 5, 2, 2, 4, 5, 5, 5, 4, 3, 4, 4, 3, 4, 5, 3, 3, 4, 4],
    "User2": [4, 5, 4, 4, 5, 4, 1, 1, 5, 4, 4, 5, 5, 2, 5, 5, 2, 3, 5, 2, 2, 5, 3],
    "User3": [2, 2, 3, 3, 2, 3, 5, 5, 3, 4, 4, 3, 3, 4, 3, 3, 5, 4, 3, 5, 4, 3, 5],
    "User4": [5, 5, 4, 5, 5, 5, 1, 1, 4, 5, 4, 5, 5, 2, 4, 5, 2, 3, 5, 2, 3, 5, 4]
}

# Create the ratings dataframe
ratings_df = pd.DataFrame(data=ratings_data, index=df['title']).T

# Compute similarity between users
user_similarity = cosine_similarity(ratings_df)


# -------------------------------------------------------------------------------------------
# Step 5. Recommendation Functions
# -------------------------------------------------------------------------------------------
def get_content_recommendations(title, num_of_recommendations=3):
    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(user_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_of_recommendations + 1]
    return [df.iloc[i[0]]["title"] for i in scores]


def get_collaborative_recommendations(user, top_n=5):
    user_idx = ratings_df.index.get_loc(user)

    # Get similar users
    sim_score = list(enumerate(user_similarity[user_idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)[1:]

    # Weighted movie scores
    movie_scores = pd.Series(0, index=ratings_df.columns)

    for sim_user_idx, score in sim_score:
        sim_user = ratings_df.index[sim_user_idx]
        movie_scores += ratings_df.loc[sim_user]

    movie_scores = movie_scores.sort_values(ascending=False)

    return list(movie_scores.head(top_n).index)


def hybrid_recommendation(user, title, alpha=0.5, top_n=5):
    content_recommendations = get_content_recommendations(title, num_of_recommendations=top_n)
    collaborative_recommendations = get_collaborative_recommendations(user, top_n=10)

    scores = {}

    # Assign scores
    for i, movie in enumerate(content_recommendations):
        scores[movie] = scores.get(movie, 0) + alpha * (10 - i)

    for i, movie in enumerate(collaborative_recommendations):
        scores[movie] = scores.get(movie, 0) + (1 - alpha) * (10 - i)

    # Sort the final recommendation
    final_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie in final_recs[:top_n]]

# -------------------------------------------------------------------------------------------
# Step 5. Test the system
# -------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print("Content-Based Recommendations for the movie 'Inception':")
    print(get_content_recommendations("Inception"))
    print()
    print("Collaborative Recommendations for 'User1':")
    print(get_collaborative_recommendations("User1"))
    print()
    print("Hybrid Recommendations for 'User1' based on the movie 'Inception':")
    print(hybrid_recommendation("User1", "Inception"))