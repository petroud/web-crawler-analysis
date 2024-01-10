import os
import math
import time
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Preprocess text (tokenization, lowercasing, stemming)
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]

# Load and preprocess documents
def load_and_preprocess_documents(directory):
    raw_documents = {}
    preprocessed_documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                raw_documents[filename] = content
                preprocessed_documents[filename] = preprocess(content)
    return raw_documents, preprocessed_documents

# Calculate IDF for each term in the corpus (for BM25)
def calculate_idf(documents):
    idf = {}
    total_docs = len(documents)
    for doc in documents.values():
        for term in set(doc):
            idf[term] = idf.get(term, 0) + 1
    for term, freq in idf.items():
        idf[term] = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
    return idf

# BM25 score for a document given a query
def bm25(doc, query, idf, k1=1.5, b=0.75, avg_doc_length=0):
    score = 0.0
    doc_length = len(doc)
    for term in query:
        if term in idf:
            df = Counter(doc)
            term_freq = df[term]
            score += idf[term] * term_freq * (k1 + 1) / (term_freq + k1 * (1 - b + b * doc_length / avg_doc_length))
    return score


scaler = MinMaxScaler(feature_range=(0.001, 0.999))

# Store timings
timings_vsm = []
timings_bm25 = []

# Main execution
directory = 'C:/users/petro/tuc_db'
raw_documents, preprocessed_documents = load_and_preprocess_documents(directory)
avg_doc_length = sum(len(d) for d in preprocessed_documents.values()) / len(preprocessed_documents)
idf = calculate_idf(preprocessed_documents)


doc_lengths = [len(doc.split()) for doc in raw_documents.values()]

# Plotting doc len distribution
plt.figure()
plt.hist(doc_lengths, bins=30, log=True)
plt.title('Document Length Distribution Logarithmic')
plt.xlabel('Document Length')
plt.ylabel('Frequency')
plt.savefig('doc_len_dist.png')

# Create TF-IDF model
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(raw_documents.values())

# Process queries
queries = ["Applied Research and Innovation","Cloud Computing"," Laboratory Staff","Air Polution","Data Analysis & Forecasting"]
top_n = 3

for query in queries:
    print(f'\nQuery: "{query}"')

    # VSM timing
    start_time = time.time()
    cosine_similarities = cosine_similarity(tfidf_vectorizer.transform([query]), tfidf_matrix).flatten()
    ranked_docs_vsm = sorted(((score, doc) for doc, score in zip(raw_documents, cosine_similarities)), reverse=True)
    end_time = time.time()
    timings_vsm.append(end_time - start_time)

    # BM25 timing
    start_time = time.time()
    preprocessed_query = preprocess(query)
    scores_bm25 = {doc: bm25(content, preprocessed_query, idf, avg_doc_length=avg_doc_length) for doc, content in preprocessed_documents.items()}
    # Normalize BM25 scores
    scores_list = np.array(list(scores_bm25.values())).reshape(-1, 1)
    normalized_scores = scaler.fit_transform(scores_list).flatten()
    scores_bm25 = dict(zip(scores_bm25.keys(), normalized_scores))
    sorted_scores_bm25 = sorted(scores_bm25.items(), key=lambda x: x[1], reverse=True)
    end_time = time.time()
    timings_bm25.append(end_time - start_time)


    # Display top N results for VSM
    print(f'\nTop-{top_n} Results | VSM')
    for score, doc in ranked_docs_vsm[:top_n]:
        print(f"Document: http://{doc.replace('_','/').replace('.txt','')}    |   Score: {score:.4f}")

    # Display top N results for BM25
    print(f'\nTop-{top_n} Results | Okapi BM25')
    for doc, score in sorted_scores_bm25[:top_n]:
        print(f"Document: http://{doc.replace('_','/').replace('.txt','')}    |   Score: {score:.4f}")


# Plotting the results
plt.figure()
plt.plot(queries, timings_vsm, label='VSM', marker='o')
plt.plot(queries, timings_bm25, label='Okapi-BM25', marker='x')
plt.xlabel('Queries')
plt.ylabel('Time (seconds)')
plt.title('Query Processing Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_time.png')