from __future__ import annotations
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

nltk.download('stopwords')
nltk.download('punkt')

stopwords = set(nltk.corpus.stopwords.words('english'))

# Load pre-trained word2vec embeddings
word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# Function to preprocess text
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stopwords]
    return tokens

# Function to calculate TF-IDF scores
def calculate_tfidf(texts: list[str], k=5):
    vectorizer = TfidfVectorizer(tokenizer=preprocess, stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = {text: [(feature_names[i], tfidf_matrix[idx, i]) for i in tfidf_matrix[idx].nonzero()[1]] for idx, text in enumerate(texts)}
    
    top_k_tfidf = {text: sorted(scores, key=lambda x: x[1], reverse=True)[:k] for text, scores in tfidf_scores.items()}
    return top_k_tfidf

# Function to calculate cosine similarity between word vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to calculate similarity score
def similarity_score(summary: list[str], content: list[str], k=5):
    tfidf_summary = calculate_tfidf([summary], k)[summary]
    tfidf_content = calculate_tfidf([content], k)[content]
    
    tfidf_summary_words = [word for word, _ in tfidf_summary]
    tfidf_content_words = [word for word, _ in tfidf_content]
    
    summary_vectors = [word2vec_model[word] for word in tfidf_summary_words if word in word2vec_model]
    content_vectors = [word2vec_model[word] for word in tfidf_content_words if word in word2vec_model]
    
    similarities = []
    for vec1 in summary_vectors:
        max_similarity = max(cosine_similarity(vec1, vec2) for vec2 in content_vectors)
        similarities.append(max_similarity)
    
    return 1 / np.mean(similarities) if similarities else float('inf')

# Function to extract surprising content
def surprise(summary, contents):
    scores = {}
    for content in contents:
        if isinstance(content, dict):
            # Recursively call surprise for nested dictionaries
            subsec = content.get('subsec', [])
            if subsec:
                surprising_content = surprise(summary, subsec)
                scores[str(content)] = similarity_score(summary, surprising_content)
        else:
            scores[content] = similarity_score(summary, content)
    
    surprising_content = max(scores, key=scores.get)
    return surprising_content

# Main HTM function
def hierarchical_trivia_miner(article, entity_name):
    summary = article['summary']
    contents = article['contents']
    tri_sec = surprise(summary, contents)
    
    tri_subsec = surprise(summary, tri_sec.get('subsec', [])) if isinstance(tri_sec, dict) else None
    tri_subsubsec = surprise(summary, tri_subsec.get('subsec', [])) if isinstance(tri_subsec, dict) else None
    tri_subsubsubsec = surprise(summary, tri_subsubsec.get('subsec', [])) if isinstance(tri_subsubsec, dict) else None

    tri_para = None
    if tri_subsubsubsec:
        tri_para = surprise(summary, tri_subsubsubsec.get('contents', []))
    elif tri_subsubsec:
        tri_para = surprise(summary, tri_subsubsec.get('contents', []))
    elif tri_subsec:
        tri_para = surprise(summary, tri_subsec.get('contents', []))
    elif isinstance(tri_sec, dict):
        tri_para = surprise(summary, tri_sec.get('contents', []))
    
    tri_sen = surprise(summary, tri_para) if isinstance(tri_para, list) else None
    
    if tri_sen and entity_name.lower() in tri_sen.lower():
        tri_sen = filtering(tri_sen, entity_name)
    
    return tri_sen

def filtering(sentence, entity_name):
    return sentence if entity_name.lower() in sentence.lower() else None

# Example usage
if __name__ == "__main__":
    article = {
    'summary': 'Stephen William Hawking was an English theoretical physicist known for his work on black holes and relativity...',
    'contents': [
        {
            'subsec': [
                {
                    'subsec': [
                        {
                            'subsec': [
                                {
                                    'contents': [
                                        'In August 2014, Hawking accepted the Ice Bucket Challenge to promote ALS/MND awareness...',
                                        'Hawking was a supporter of the Labour Party and backed the remain campaign in the 2016 EU referendum...'
                                    ]
                                },
                                {
                                    'contents': [
                                        'Hawking published a popular science book called "A Brief History of Time" in 1988...',
                                        'He made significant contributions to the fields of cosmology and quantum gravity...'
                                    ]
                                }
                            ]
                        },
                        {
                            'contents': [
                                'In 1974, Hawking was elected a Fellow of the Royal Society...',
                                'He became the Lucasian Professor of Mathematics at the University of Cambridge...'
                            ]
                        }
                    ]
                },
                {
                    'contents': [
                        'Hawking received numerous honors and awards, including the Presidential Medal of Freedom in 2009...',
                        'He appeared in several popular TV shows such as "The Simpsons" and "Star Trek: The Next Generation"...'
                    ]
                }
            ]
        },
        {
            'contents': [
                'Hawking was diagnosed with ALS, a form of motor neurone disease, in 1963...',
                'Despite his illness, he continued to work and make groundbreaking discoveries in theoretical physics...'
            ]
        }
    ],
    'entity_name': 'Stephen Hawking'
    }
    
    trivia_fact = hierarchical_trivia_miner(article, article['entity_name'])
    print(trivia_fact)
