from flask import Flask, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))


    words = word_tokenize(text)


    filtered_words = [stemmer.stem(w) for w in words if w not in stop_words]

    return ' '.join(filtered_words)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    message = ""
    if request.method == 'POST':
        message = request.form['message']
        processed = preprocess_text(message)
        vector = vectorizer.transform([processed])
        pred = model.predict(vector)[0]
        prediction = "Spam" if pred == 1 else "Not Spam"
    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
