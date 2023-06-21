import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# File path of the dataset CSV file
dataset_file = "C:/Users/LEGION 5 PRO/OneDrive/Documents/Semester 4/Temu Kembali Informasi/dataset 14/sentiment labelled sentences/datasets.csv"

# Define the regex pattern to remove punctuation and replace with whitespace
regex = re.compile('[%s]' % re.escape(string.punctuation.replace('.,', '')))

# Define stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Replace "1", "0", ",0,,,", and ",1,,,," with whitespace
    text = re.sub(r'\b[01]\b', ' ', text)
    text = re.sub(r',0,,,,,', ' ', text)
    text = re.sub(r',1,,,,,', ' ', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Read the dataset from the CSV file
dataset_files = pd.read_csv(dataset_file).fillna('')

# Apply preprocessing to each column in the dataframe
for column in dataset_files.columns:
    dataset_files[column] = dataset_files[column].apply(preprocess_text)

# Save the preprocessed dataset as a new CSV file
output_file = "C:/Users/LEGION 5 PRO/OneDrive/Documents/Semester 4/Temu Kembali Informasi/dataset 14/sentiment labelled sentences/preprocessed_dataset.csv"
dataset_files.to_csv(output_file, index=False)

print("Datasets preprocessed and saved to:", output_file)