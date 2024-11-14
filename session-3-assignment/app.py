import os
import random
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the uploaded file is a .txt file."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_text(text):
    """Tokenization and stopword removal."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens), filtered_tokens  # Return both processed text and tokens

def augment_text(text):
    """Sentence shuffling for augmentation."""
    sentences = text.split('. ')
    random.shuffle(sentences)
    return '. '.join(sentences)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the uploaded file
        with open(filepath, 'r') as f:
            file_content = f.read()
        
        return jsonify({'success': True, 'content': file_content}), 200
    else:
        return jsonify({'error': 'File not allowed'}), 400

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Apply text preprocessing to the uploaded file."""
    text = request.json['text']
    processed_text, tokens = preprocess_text(text)
    return jsonify({'processed_text': processed_text, 'tokens': tokens, 'num_tokens': len(tokens)})

@app.route('/augment', methods=['POST'])
def augment():
    """Apply text augmentation to the uploaded file."""
    text = request.json['text']
    augmented_text = augment_text(text)
    return jsonify({'augmented_text': augmented_text})

@app.route('/tokenize', methods=['POST'])
def tokenize():
    """Tokenize the text and return the tokens with their count."""
    text = request.json['text']
    tokens = word_tokenize(text)
    return jsonify({'tokens': tokens, 'num_tokens': len(tokens)})

if __name__ == '__main__':
    app.run(debug=True)
