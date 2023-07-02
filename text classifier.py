import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.decomposition import TruncatedSVD



# Read the dataset using pandas (assuming the dataset is in CSV format)
dataset_path = "path of the dataset"
newsgroups_data = np.load(dataset_path, allow_pickle=True)

# Extract the 'data' and 'target' columns from the dataset
X = newsgroups_data['data']
y = newsgroups_data['target']

# Step 2: Preprocess and Vectorize the Dataset

# Preprocess and tokenize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Step 3: Build a Text Classification System

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply dimensionality reduction
svd = TruncatedSVD(n_components=100)
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

# Build a Naive Bayes classifier
classifier = MultinomialNB()

# Step 4: Train the classifier using mini-batch updates

# Set the batch size
batch_size = 1000
n_batches = int(np.ceil(X_train.shape[0] / batch_size))

# Iterate over mini-batches and update the model
for batch in range(n_batches):
    start_idx = batch * batch_size
    end_idx = min((batch + 1) * batch_size, X_train.shape[0])
    classifier.partial_fit(X_train[start_idx:end_idx], y_train[start_idx:end_idx], classes=np.unique(y))

# Step 5: Evaluate System Performance

# Predict the labels for the test set
y_pred = classifier.predict(X_test)

# Calculate and print performance metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))
print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))
print("F1-Score:", metrics.f1_score(y_test, y_pred, average='weighted'))
