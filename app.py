from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Load data
df = pd.read_csv('datajurusan.csv')
df['minat_bakat'] = df['minat'] + ' ' + df['bakat']
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['minat_bakat'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/kampus')
def kampus():
    return render_template('kampus.html')

@app.route('/jurusan')
def jurusan():
    return render_template('jurusan.html')

@app.route('/rekomendasi')
def rekomendasi():
    return render_template('rekomendasi.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    text = request.form['input_text']
    text = preprocess_text(text)
    text_vector = vectorizer.transform([text])

    cosine_similarities = cosine_similarity(text_vector, tfidf_matrix)
    similar_indices = cosine_similarities[0].argsort()[:-6:-1]

    recommendations = df.loc[similar_indices, 'jurusan'].tolist()

    return render_template('rekomendasi.html', recommendations=recommendations)

def preprocess_text(data):
    data = str(data).lower()
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.replace(',', '')
    data = data.replace("'", '')
    return data

if __name__ == '__main__':
    app.run(debug=True)
