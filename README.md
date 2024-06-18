# 📚 Article Recommendation System

This project is a Flask web application that provides personalized article recommendations based on user interests. The recommendations are generated using three different models: TF-IDF, FastText, and SciBERT. The application also includes user registration and login functionalities with MongoDB as the backend database.

## 🚀 Features

- **User Registration and Login:** Secure registration and login system using Flask-PyMongo and Werkzeug for password hashing.
- **Text Preprocessing:** Utilizes NLTK for text tokenization, stopword removal, and stemming.
- **Article Recommendations:** Provides recommendations based on user input using TF-IDF, FastText, and SciBERT models.
- **Performance Metrics:** Displays precision and recall metrics for each recommendation model.

## 🛠️ Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/article-recommendation-system.git
    cd article-recommendation-system
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK data:**

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

5. **Install Dataset**
   link: https://www.kaggle.com/datasets/Cornell-University/arxiv

## ⚙️ Configuration

1. **MongoDB Configuration:**
    - Ensure MongoDB is installed and running on your machine.
    - Update the `MONGO_URI` in `app.config` to point to your MongoDB instance if it's different from the default `mongodb://localhost:27017/makale`.

2. **SciBERT Model:**
    - The application uses the `allenai/scibert_scivocab_uncased` model from the Hugging Face library. Make sure you have internet access to download the model.

## 🔧 Usage

1. **Run the Flask application:**

    ```bash
    flask run
    ```

2. **Access the application:**
    - Open your web browser and go to `http://127.0.0.1:5000/`.

3. **Register a new user:**
    - Enter your email, password, and interests on the registration page.
    - Interests should be comma-separated.

4. **Login:**
    - Use your registered email and password to log in.

5. **Get Recommendations:**
    - Once logged in, you can input your search query and get article recommendations based on the specified range of rows in the JSON file.

## 📊 Recommendation Models

- **TF-IDF:** Term Frequency-Inverse Document Frequency, a statistical measure used to evaluate how important a word is to a document in a collection or corpus.
- **FastText:** A library for efficient learning of word representations and sentence classification.
- **SciBERT:** A pre-trained language model based on BERT, designed specifically for scientific literature.

## 📈 Performance Metrics

- The application calculates and displays precision and recall metrics for each recommendation model based on the ground truth provided.

## 📝 Example JSON Data Processing

- The application processes a JSON file containing article metadata. It extracts and preprocesses abstracts to create vectors for similarity comparison.

## 📂 Project Structure

```bash
├── static
│   ├── css
│   │   └── style.css
├── templates
│   ├── index.html
│   ├── anasayfa.html
│   └── recommendations.html
├── app.py
├── requirements.txt
├── README.md
└── arxiv_metadata_oai_snapshot.json

