# Smart-Book-Recommendation-System
Machine Learning project that recommends books using Popularity-Based Recommendation, Content-Based Filtering, and Collaborative Filtering with SVD.


## Overview

The Smart Book Recommendation System is a Python-based application that recommends books to users using multiple recommendation techniques. The system combines Popularity-Based Recommendation, Content-Based Filtering, and Collaborative Filtering to provide personalized suggestions.

---

## Dataset Description

The project uses three datasets:

### 1. books.csv

Contains information about books such as:

- Book ID
- Title
- Author
- Genre

**Purpose:**
This dataset provides the content information required for content-based recommendation.

---

### 2. users.csv

Contains information about users.

- User ID

**Purpose:**
Used to identify individual users for personalized recommendations.

---

### 3. ratings.csv

Contains user ratings for books.

- User ID
- Book ID
- Rating (1–5)

**Purpose:**
This dataset captures user preferences and is used for collaborative filtering.

---

## Recommendation Techniques Used

### 1. Popularity-Based Recommendation

Books with the highest average ratings are displayed.

#### Why?

- Simple and effective.
- Useful for new users who have no rating history.
- Provides generally well-liked books.

---

### 2. Content-Based Filtering

Books are recommended based on similarities in:

- Author
- Genre

The project uses:

- CountVectorizer
- Cosine Similarity

#### Why?

Books having similar characteristics are likely to be preferred by users who enjoyed a particular book.

For example:

```
Harry Potter → Fantasy → Adventure
```

Users who like fantasy books may receive similar recommendations.

---

### 3. Collaborative Filtering using Singular Value Decomposition (SVD)

The project uses the SVD algorithm from the Surprise library.

#### Why SVD?

SVD is one of the most widely used recommendation algorithms because:

- It learns hidden relationships between users and books.
- It predicts ratings for unseen books.
- It handles sparse datasets effectively.
- It provides personalized recommendations.
- It generally produces better accuracy than simple neighborhood methods.

---

## Why SVD Was Chosen

Among various collaborative filtering algorithms, SVD was selected because:

- It works efficiently on rating data.
- It reduces the dimensionality of the user-item matrix.
- It captures latent factors that influence user preferences.
- It offers high prediction accuracy.
- It is extensively used in real-world recommendation systems such as Netflix and Amazon.

---

## Libraries Used

- Pandas
- NumPy
- Scikit-learn
- Surprise Library

---

## Technologies Used

- Python
- Machine Learning
- Recommendation Systems
- Collaborative Filtering
- Content-Based Filtering

---

## Workflow

1. Load datasets.
2. Display top-rated books.
3. Generate content-based recommendations using author and genre similarity.
4. Train the SVD model using user ratings.
5. Predict ratings for unseen books.
6. Recommend books to users based on predicted ratings.

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python recc.py
```

---

## Future Improvements

- Add a graphical user interface.
- Deploy as a web application using Streamlit or Flask.
- Incorporate deep learning-based recommendation models.
- Use larger datasets for improved accuracy.
- Implement hybrid recommendation techniques.

---

## Author

**Devisree**
