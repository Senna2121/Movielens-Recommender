# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:22:24 2026

@author: janma
"""


# ============================================================
# MOVIELENS - SNACK GEBASEERD FILMAANBEVELINGSSYSTEEM
# ============================================================

# 1. IMPORTS
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ============================================================
# 2. DATASET INLEZEN
# ============================================================

bestand = "movielens_snacks.csv"

df = pd.read_csv(bestand, low_memory=False)

# Kolommen omzetten naar getallen
numerieke_kolommen = [
    "userId", "movieId", "rating", "timestamp",
    "imdbId", "tmdbId", "popcorn", "chips",
    "chocolade", "pizza", "gezonde_snacks", "geen_snacks"
]

for kolom in numerieke_kolommen:
    if kolom in df.columns:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

# Dubbele beoordelingen verwijderen
df = df.drop_duplicates(subset=["userId", "movieId"])

# Onvolledige belangrijke gegevens verwijderen
df = df.dropna(subset=["userId", "movieId", "rating", "genres"])

# Snackkolommen
snack_kolommen = [
    "popcorn",
    "chips",
    "chocolade",
    "pizza",
    "gezonde_snacks",
    "geen_snacks"
]

# Ontbrekende snackwaarden opvullen
df[snack_kolommen] = df[snack_kolommen].fillna(0)

# Usernames bewaren
gebruikersnamen = (
    df.dropna(subset=["userName"])
    .groupby("userId")["userName"]
    .first()
)

print("Dataset succesvol ingelezen!")


# ============================================================
# 3. TRAIN- EN TESTDATA MAKEN
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 4. FILMBEOORDELINGENMATRIX
# ============================================================

ratings_train = train_df.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Gemiddelde rating per gebruiker
user_means = ratings_train.mean(axis=1)

# Ratings centreren
ratings_centered = ratings_train.sub(
    user_means,
    axis=0
).fillna(0)


# ============================================================
# 5. COSINE SIMILARITY TUSSEN GEBRUIKERS
# ============================================================

similarity = cosine_similarity(ratings_centered)

user_similarity = pd.DataFrame(
    similarity.copy(),
    index=ratings_train.index,
    columns=ratings_train.index
)

# Gebruikers niet met zichzelf vergelijken
for user in user_similarity.index:
    user_similarity.loc[user, user] = 0


# ============================================================
# 6. RATINGS OP BASIS VAN VERGELIJKBARE GEBRUIKERS
# ============================================================

positive_similarity = user_similarity.clip(lower=0)

numerator = positive_similarity.dot(
    ratings_train.fillna(0)
)

denominator = positive_similarity.dot(
    ratings_train.notna().astype(float)
)

global_mean = train_df["rating"].mean()

similarity_rating_matrix = numerator.div(
    denominator.replace(0, np.nan)
).fillna(global_mean)


# ============================================================
# 7. SIMILARITY RATING TOEVOEGEN
# ============================================================

def voeg_similarity_toe(data):

    lookup = similarity_rating_matrix.stack()

    sleutels = pd.MultiIndex.from_arrays([
        data["userId"],
        data["movieId"]
    ])

    data = data.copy()

    data["similarity_rating"] = (
        lookup.reindex(sleutels)
        .fillna(global_mean)
        .to_numpy()
    )

    return data


train_df = voeg_similarity_toe(train_df)
test_df = voeg_similarity_toe(test_df)


# ============================================================
# 8. RANDOM FOREST MODEL TRAINEN
# ============================================================

features = [
    "userId",
    "movieId",
    "genres",
    "similarity_rating"
]

target = "rating"

preprocessor = ColumnTransformer(
    transformers=[
        (
            "genres",
            OneHotEncoder(handle_unknown="ignore"),
            ["genres"]
        )
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

X_train = train_df[features]
y_train = train_df[target]

model.fit(X_train, y_train)


# ============================================================
# 9. MODEL EVALUEREN
# ============================================================

X_test = test_df[features]
y_test = test_df[target]

voorspellingen = model.predict(X_test)

mae = mean_absolute_error(y_test, voorspellingen)
rmse = np.sqrt(mean_squared_error(y_test, voorspellingen))

print("\nRandom Forest model is getraind!")
print("\nResultaten machine learning:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")


# ============================================================
# 10. SNACK-GENRE PROFIEL VAN EEN GEBRUIKER MAKEN
# ============================================================

def maak_snack_genre_profiel(user_id, snack):

    # Films die de gebruiker aan deze snack heeft gekoppeld
    snack_films = df[
        (df["userId"] == user_id) &
        (df[snack] == 1)
    ].copy()

    # Als gebruiker geen films aan snack heeft gekoppeld
    if snack_films.empty:
        return None, None

    # Genres uit de snack-films ophalen
    genre_lijst = []

    for genres in snack_films["genres"]:
        genre_lijst.extend(genres.split("|"))

    # Genrefrequentie berekenen
    genre_frequentie = pd.Series(genre_lijst).value_counts()

    return snack_films, genre_frequentie


# ============================================================
# 11. SNACKMATCH BEREKENEN
# ============================================================

def bereken_snackmatch(genres, genre_frequentie):

    film_genres = genres.split("|")

    # Alleen genres gebruiken die in snackprofiel voorkomen
    overeenkomende_genres = [
        genre for genre in film_genres
        if genre in genre_frequentie.index
    ]

    if len(overeenkomende_genres) == 0:
        return 0

    # Genrescore gebaseerd op hoe vaak genre voorkomt in snack-films
    score = sum(
        genre_frequentie[genre]
        for genre in overeenkomende_genres
    )

    max_score = genre_frequentie.sum()

    if max_score == 0:
        return 0

    return score / max_score


# ============================================================
# 12. AANBEVELINGEN PER SNACK MAKEN
# ============================================================

def recommend_movies_per_snack(user_id, aantal=10):

    # Gebruikersnaam ophalen
    if user_id in gebruikersnamen.index:
        naam = gebruikersnamen.loc[user_id]
    else:
        naam = f"Gebruiker {user_id}"

    print("\n" + "=" * 60)
    print(f"FILMAANBEVELINGEN VOOR {naam}")
    print("=" * 60)

    # Films die gebruiker al heeft beoordeeld
    bekeken_films = df.loc[
        df["userId"] == user_id,
        "movieId"
    ].unique()

    # Snacks die deze gebruiker daadwerkelijk heeft gebruikt
    gebruikte_snacks = [
        snack for snack in snack_kolommen
        if (df.loc[df["userId"] == user_id, snack] == 1).any()
    ]

    if len(gebruikte_snacks) == 0:
        print("Deze gebruiker heeft geen snack-film koppelingen.")
        return

    # Per snack aanbevelingen maken
    for snack in gebruikte_snacks:

        # Snackfilms en bijbehorende genres ophalen
        snack_films, genre_frequentie = maak_snack_genre_profiel(
            user_id, snack
        )

        if snack_films is None:
            continue

        # Meest voorkomende genres in snackprofiel
        top_genres = genre_frequentie.head(3).index.tolist()

        genre_uitleg = ", ".join(top_genres)

        # Alle films verzamelen
        kandidaten = df[
            ["movieId", "title", "genres"]
        ].drop_duplicates().copy()

        # Films verwijderen die gebruiker al heeft beoordeeld
        kandidaten = kandidaten[
            ~kandidaten["movieId"].isin(bekeken_films)
        ].copy()

        # UserId toevoegen
        kandidaten["userId"] = user_id

        # Similarity rating toevoegen
        kandidaten["similarity_rating"] = (
            similarity_rating_matrix.loc[user_id]
            .reindex(kandidaten["movieId"])
            .fillna(global_mean)
            .values
        )

        # Voorspelde rating met Random Forest
        kandidaten["voorspelde_rating"] = model.predict(
            kandidaten[features]
        )

        # Genre-match met deze snack berekenen
        kandidaten["snackmatch"] = kandidaten["genres"].apply(
            lambda genres: bereken_snackmatch(
                genres, genre_frequentie
            )
        )

        # Films zonder genre-overeenkomst uitsluiten
        kandidaten = kandidaten[
            kandidaten["snackmatch"] > 0
        ].copy()

        if kandidaten.empty:
            print(f"\nGeen aanbevelingen gevonden voor {snack}.")
            continue

        # Eindscore combineren
        kandidaten["eindscore"] = (
            (kandidaten["voorspelde_rating"] / 5) * 0.80
            + kandidaten["snackmatch"] * 0.20
        )

        # Top films selecteren
        aanbevelingen = kandidaten.sort_values(
            "eindscore",
            ascending=False
        ).head(aantal)

        # ====================================================
        # OUTPUT PER SNACK
        # ====================================================

        print("\n" + "-" * 60)
        print(f"SNACK: {snack.upper()}")
        print("-" * 60)

        print(
            f"Gebaseerd op {len(snack_films)} films die "
            f"{naam} aan {snack} heeft gekoppeld."
        )

        print(f"Genres die naar voren komen: {genre_uitleg}")

        print("\nAanbevolen films:")

        for i, (_, film) in enumerate(aanbevelingen.iterrows(), start=1):

            print(
                f"{i}. {film['title']} "
                f"({film['genres']}) | "
                f"Voorspelde rating: {film['voorspelde_rating']:.2f}"
            )

        # Korte uitleg per snack
        print("\nWaarom deze films?")

        print(
            f"Je krijgt deze aanbevelingen omdat je bij {snack} "
            f"films hebt gekozen die vaak de genres {genre_uitleg} bevatten. "
            f"Daarom zoekt het systeem naar andere films met overeenkomende "
            f"genres die je nog niet hebt beoordeeld, en rangschikt deze "
            f"op basis van je voorspelde rating."
        )


# ============================================================
# 13. GEBRUIKER TESTEN
# ============================================================

# Alleen de eerste drie gebruikers hebben een username
# Kies 1 = Harry, 2 = Roos, 3 = Sandra

recommend_movies_per_snack(user_id=1, aantal=5)

