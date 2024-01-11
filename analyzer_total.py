import os
import math
import time
import pandas as pd
import json
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

# Function to calculate precision and recall
def calculate_precision_recall_f1(retrieved_docs, relevant_docs):
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    true_positives = len(retrieved_set.intersection(relevant_set))
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def load_relevant_documents(filename):
    with open(filename, 'r') as file:
        return json.load(file)

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


relevant_docs = load_relevant_documents('query_relevance.json')

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
queries = ["Applied Research and Innovation","Cloud Computing","Laboratory Staff","Air Polution","Data Analysis & Forecasting"]
topn_Arr = [5, 7, 10]
topn_Rel_Arr = [5, 7, 10, 12]

for query in queries:
    for top_n in topn_Arr:
        results = {query: {'VSM': {}, 'BM25': {}} for query in queries}
        for top_n_rel in topn_Rel_Arr:
            print(f'-----------------------------------------------------------------')
            print(f'TOP-{top_n} Answers for TOP-{top_n_rel} Relevant Documents Labels')

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
            sorted_scores_bm25 = sorted(scores_bm25.items(), key=lambda x: x[1], reverse=True)
            end_time = time.time()
            timings_bm25.append(end_time - start_time)

            # Lower scores might indicate higher difficulty
            avg_score_vsm = sum(score for score, _ in ranked_docs_vsm) / len(ranked_docs_vsm)
            avg_score_bm25 = sum(score for _, score in sorted_scores_bm25) / len(sorted_scores_bm25)

            retrieved_docs_vsm = [doc for _, doc in ranked_docs_vsm[:top_n]]
            retrieved_docs_bm25 = [doc for doc, _ in sorted_scores_bm25[:top_n]]
            
            precision_vsm, recall_vsm, f1_vsm = calculate_precision_recall_f1(retrieved_docs_vsm,  relevant_docs.get(query, [])[:top_n_rel])
            precision_bm25, recall_bm25, f1_bm25 = calculate_precision_recall_f1(retrieved_docs_bm25,  relevant_docs.get(query, [])[:top_n_rel])

            # Store results in dictionary
            results[query]['VSM'][top_n_rel] = precision_vsm
            results[query]['BM25'][top_n_rel] = precision_bm25

            print(f"\n--> Query: {query}")

            # Display top N results for VSM
            print(f'\nTop-{top_n} Results | VSM | Difficulty: {avg_score_vsm}')
            print(f"VSM - Precision: {precision_vsm:.4f}, Recall: {recall_vsm:.4f}  |  F1 Score: {f1_vsm:.4f}")
            for score, doc in ranked_docs_vsm[:top_n]:
                print(f"Document: http://{doc.replace('_','/').replace('.txt','')}    |   Score: {score:.4f}")

            # Display top N results for BM25
            print(f'\nTop-{top_n} Results | Okapi BM25 | Difficulty: {avg_score_bm25}')
            print(f"BM25 - Precision: {precision_bm25:.4f}, Recall: {recall_bm25:.4f}  |  F1 Score: {f1_bm25:.4f}\n")
            for doc, score in sorted_scores_bm25[:top_n]:
                print(f"Document: http://{doc.replace('_','/').replace('.txt','')}    |   Score: {score:.4f}")

        plt.figure()
        plt.title(f'Precision for Top-{top_n} retrieved docs over Top-N relevant docs\n Query: "{query}"')

        # VSM and BM25 precision values for different top_n_rel values
        vsm_precisions = [results[query]['VSM'][top_n_rel] for top_n_rel in topn_Rel_Arr]
        bm25_precisions = [results[query]['BM25'][top_n_rel] for top_n_rel in topn_Rel_Arr]

        plt.plot(topn_Rel_Arr, vsm_precisions, label='VSM')
        plt.plot(topn_Rel_Arr, bm25_precisions, label='BM25')
        plt.xlabel('Top-N Labeled Relevant Documents')
        plt.ylabel('Precision')
        plt.legend()
        
        plt.savefig(f'{query.lower().replace(" ","_")}_topn_{top_n}.png')        

