from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
from gensim.models import FastText
from transformers import BertTokenizer, BertModel
import torch

nltk.download('punkt')
nltk.download('stopwords')

# Stop words ve stemmer ayarlarının yapıldığı yer.
stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)
stemmer = SnowballStemmer('english')

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/makale"
app.secret_key = 'your_secret_key'

mongo = PyMongo(app)

# Metin ön işleme fonksiyonunu sağladığı yer.
def preprocess_text(text):
    text = text.lower()
    text = text.translate(punctuation_table)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# JSON dosyamızın satır satır okunduğu yer.
def process_json(filepath, start_row, end_row):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            if idx >= start_row and idx < end_row:
                item = json.loads(line)
                item['processed_abstract'] = preprocess_text(item['abstract'])
                data.append(item)
    df = pd.DataFrame(data)
    return df

# Veriyi vektörleştiren fonksiyon
def vectorize_data(data):
    vectorizer = TfidfVectorizer()
    corpus = data['processed_abstract'].tolist()
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer

# Kullanıcı profili için vektör temsilini oluşturur.
def create_user_profile_fasttext(user_interests, model):
    user_profile = np.zeros(model.wv.vector_size)
    for interest in user_interests:
        if interest in model.wv:
            user_profile += model.wv[interest]
    user_profile /= len(user_interests)
    return user_profile

# Kullanıcı profili için vektör temsilini oluşturur.
def create_user_profile_scibert(user_interests, model, tokenizer):
    vectors = []
    for interest in user_interests:
        inputs = tokenizer(interest, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        vectors.append(vector)
    user_profile = np.mean(vectors, axis=0)
    return user_profile

# Öneri yap ve top_indices değerlerini döndürür.
def recommend_articles(data, tfidf_matrix, vectorizer, fasttext_model, scibert_model, scibert_tokenizer, user_input):
    processed_input = preprocess_text(user_input)
    
    # TF-IDF vektör temsili
    user_vector_tfidf = vectorizer.transform([processed_input])
    
    # FastText ve SciBERT vektör temsillerini yapar.
    fasttext_vector = create_user_profile_fasttext(processed_input.split(), fasttext_model)
    scibert_vector = create_user_profile_scibert(processed_input.split(), scibert_model, scibert_tokenizer)
    
    # Cosine Similarity metriği ile benzerlik hesaplar
    cosine_similarities_tfidf = cosine_similarity(user_vector_tfidf, tfidf_matrix).flatten()
    
    fasttext_article_vectors = np.array([create_user_profile_fasttext(article.split(), fasttext_model) for article in data['processed_abstract']])
    cosine_similarities_fasttext = cosine_similarity([fasttext_vector], fasttext_article_vectors).flatten()
    
    scibert_article_vectors = np.array([create_user_profile_scibert(article.split(), scibert_model, scibert_tokenizer).flatten() for article in data['processed_abstract']])
    cosine_similarities_scibert = cosine_similarity(scibert_vector, scibert_article_vectors).flatten()
    
    # En yakın 5 makaleyi bulmamıza olanak sağlar.
    tfidf_top_indices = cosine_similarities_tfidf.argsort()[-5:][::-1]
    fasttext_top_indices = cosine_similarities_fasttext.argsort()[-5:][::-1]
    scibert_top_indices = cosine_similarities_scibert.argsort()[-5:][::-1]

    return tfidf_top_indices, fasttext_top_indices, scibert_top_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    user = mongo.db.users.find_one({'email': email})
    if user and check_password_hash(user['password'], password):
        session['user'] = email
        session['interests'] = user['interests']
        return redirect(url_for('home'))
    flash('Invalid credentials')
    return redirect(url_for('index'))

@app.route('/register', methods=['POST'])
def register():
    email = request.form['email']
    password = generate_password_hash(request.form['password'])
    interests = request.form['interests'].split(',')
    mongo.db.users.insert_one({'email': email, 'password': password, 'interests': interests})
    flash('Registration successful!')
    return redirect(url_for('index'))

@app.route('/home')
def home():
    if 'user' in session:
        return render_template('anasayfa.html', interests=session['interests'])
    return redirect(url_for('index'))

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user' not in session:
        return redirect(url_for('index'))
    
    start_row = int(request.form['start_row'])
    end_row = int(request.form['end_row'])
    user_input = request.form['user_input']
    
    file_path = 'arxiv_metadata_oai_snapshot.json'
    articles_data = process_json(file_path, start_row, end_row)
    tfidf_matrix, tfidf_vectorizer = vectorize_data(articles_data)
    
    fasttext_model = FastText(vector_size=100, window=3, min_count=1, epochs=10)
    sentences = articles_data['processed_abstract'].apply(str.split).tolist()
    fasttext_model.build_vocab(sentences)
    fasttext_model.train(sentences, total_examples=len(sentences), epochs=10)
    
    scibert_tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    scibert_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    tfidf_top_indices, fasttext_top_indices, scibert_top_indices = recommend_articles(articles_data, tfidf_matrix, tfidf_vectorizer, fasttext_model, scibert_model, scibert_tokenizer, user_input)
    
    recommendations = {
        'tfidf': articles_data.iloc[tfidf_top_indices].to_dict(orient='records'),
        'fasttext': articles_data.iloc[fasttext_top_indices].to_dict(orient='records'),
        'scibert': articles_data.iloc[scibert_top_indices].to_dict(orient='records')
    }
    ground_truth = [1, 0, 1, 0, 1]  
    predicted_tfidf = [1 if idx in tfidf_top_indices else 0 for idx in range(len(articles_data))]
    precision_tfidf = precision_score(ground_truth, predicted_tfidf[:len(ground_truth)])
    recall_tfidf = recall_score(ground_truth, predicted_tfidf[:len(ground_truth)])
    
    predicted_fasttext = [1 if idx in fasttext_top_indices else 0 for idx in range(len(articles_data))]
    precision_fasttext = precision_score(ground_truth, predicted_fasttext[:len(ground_truth)])
    recall_fasttext = recall_score(ground_truth, predicted_fasttext[:len(ground_truth)])
    
    predicted_scibert = [1 if idx in scibert_top_indices else 0 for idx in range(len(articles_data))]
    precision_scibert = precision_score(ground_truth, predicted_scibert[:len(ground_truth)])
    recall_scibert = recall_score(ground_truth, predicted_scibert[:len(ground_truth)])
    
    metrics = {
        'tfidf': {'precision': precision_tfidf, 'recall': recall_tfidf},
        'fasttext': {'precision': precision_fasttext, 'recall': recall_fasttext},
        'scibert': {'precision': precision_scibert, 'recall': recall_scibert}
    }
    
    return render_template('recommendations.html', recommendations=recommendations, metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
