import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load the movie reviews dataset
dataset = pd.read_csv('movie_review.csv')

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    dataset['text'], dataset['tag'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the text data into numerical features
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# Train a linear support vector machine (SVM) classifier
classifier = LinearSVC()
classifier.fit(train_features, train_labels)

# Predict the sentiment labels for the test set
predictions = classifier.predict(test_features)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

# Visualize the distribution of sentiment labels
sns.countplot(x='tag', data=dataset)
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.show()