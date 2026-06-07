import sys
print(sys.executable)
print(sys.version)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split

# =====================================
# LOAD DATA
# =====================================

books = pd.read_csv("books.csv")
users = pd.read_csv("users.csv")
ratings = pd.read_csv("ratings.csv")

print("Datasets Loaded Successfully!")

# =====================================
# POPULAR BOOKS
# =====================================

def show_popular_books():

    popular = ratings.groupby("book_id")["rating"].mean().reset_index()

    popular = popular.merge(
        books,
        on="book_id"
    )

    popular = popular.sort_values(
        by="rating",
        ascending=False
    )

    print("\nTOP 10 POPULAR BOOKS\n")

    for i in range(min(10, len(popular))):

        print(f"{i+1}. {popular.iloc[i]['title']}")
        print("Author :", popular.iloc[i]['author'])
        print("Genre  :", popular.iloc[i]['genre'])
        print("Rating :", round(popular.iloc[i]['rating'],2))
        print("-"*40)

# =====================================
# CONTENT BASED RECOMMENDATION
# =====================================

books["features"] = (
    books["author"].astype(str)
    + " "
    + books["genre"].astype(str)
)

cv = CountVectorizer()

matrix = cv.fit_transform(
    books["features"]
)

similarity = cosine_similarity(matrix)

def content_recommend(book_name):

    if book_name not in books["title"].values:
        print("Book Not Found!")
        return

    idx = books[
        books["title"] == book_name
    ].index[0]

    distances = list(
        enumerate(similarity[idx])
    )

    recommendations = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    print("\nCONTENT BASED RECOMMENDATIONS\n")

    for item in recommendations:

        row = books.iloc[item[0]]

        print("Title :", row["title"])
        print("Author:", row["author"])
        print("Genre :", row["genre"])
        print("-"*40)

# =====================================
# SVD COLLABORATIVE FILTERING
# =====================================

print("\nTraining SVD Model...")

reader = Reader(rating_scale=(1,5))

data = Dataset.load_from_df(
    ratings[["user_id","book_id","rating"]],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

svd_model = SVD()

svd_model.fit(trainset)

print("SVD Model Trained Successfully!")

def svd_recommend(user_id, n=5):

    recommendations = []

    for book_id in books["book_id"]:

        predicted_rating = svd_model.predict(
            user_id,
            book_id
        ).est

        recommendations.append(
            (book_id, predicted_rating)
        )

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nTOP {n} RECOMMENDATIONS FOR USER {user_id}\n")

    for book_id, rating in recommendations[:n]:

        title = books[
            books["book_id"] == book_id
        ]["title"].values[0]

        print(
            f"{title} --> Predicted Rating: {rating:.2f}"
        )

# =====================================
# MENU
# =====================================

while True:

    print("\n")
    print("="*50)
    print("SMART BOOK RECOMMENDATION SYSTEM")
    print("="*50)

    print("1. Show Popular Books")
    print("2. Content Based Recommendation")
    print("3. SVD Recommendation")
    print("4. Exit")

    choice = input("\nEnter Choice: ")

    if choice == "1":

        show_popular_books()

    elif choice == "2":

        book_name = input(
            "\nEnter Book Name: "
        )

        content_recommend(book_name)

    elif choice == "3":

        user_id = int(
            input(
                "\nEnter User ID (1-50): "
            )
        )

        svd_recommend(user_id)

    elif choice == "4":

        print("\nThank You!")
        break

    else:

        print("Invalid Choice!")