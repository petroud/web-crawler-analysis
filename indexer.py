import os
import string
import nltk
import pickle
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def save_reverse_index(reverse_index, filename):
    with open(filename, 'wb') as file:
        pickle.dump(reverse_index, file)

def create_reverse_index(directory):
    reverse_index = defaultdict(set)
    stop_words = set(stopwords.words('english'))
    file_word_counts = []
    indexing_times_ms = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            start_time = time.time()
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().lower()
                text = text.translate(str.maketrans('', '', string.punctuation))
                tokens = [word for word in word_tokenize(text) if word not in stop_words]
                file_word_count = len(tokens)
                for token in tokens:
                    reverse_index[token].add(filename)
            end_time = time.time()
            indexing_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            file_word_counts.append(file_word_count)
            indexing_times_ms.append(indexing_time_ms)
            print(f'Indexed {filename} in {indexing_time_ms:.2f} ms')

    return reverse_index, file_word_counts, indexing_times_ms

def plot_indexing_times(file_word_counts, indexing_times_ms):
    plt.scatter(file_word_counts, indexing_times_ms)
    plt.title('Indexing Time vs File Word Count')
    plt.xlabel('File Size (number of words)')
    plt.ylabel('Time to Index (milliseconds)')
    plt.savefig('indexing_times_plot.png')
    plt.show()


def check_and_create_index(directory, index_file):
    if os.path.exists(index_file):
        response = input(f"The file '{index_file}' already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
        if response != 'yes':
            return None, None, None

    return create_reverse_index(directory)

# Download necessary NLTK data
print('Setting up environment...')
nltk.download('punkt')
nltk.download('stopwords')
print('Setting up completed.')

directory_path = 'C:/Users/petro/tuc_db'
index_file = 'r_idx.pkl'

reverse_index, file_word_counts, indexing_times_ms = check_and_create_index(directory_path, index_file)

if reverse_index is not None:
    save_reverse_index(reverse_index, index_file)
    plot_indexing_times(file_word_counts, indexing_times_ms)
    # Example of how to use the reverse index to find files containing a specific word
    word = 'petrakis'
    print(f"Files containing the word '{word}':", reverse_index.get(word, set()))
else:
    print("Indexing was cancelled.")