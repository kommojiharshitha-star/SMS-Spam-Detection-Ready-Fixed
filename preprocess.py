import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text, preserve_line=True)
    tokens = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]
    tokens = [ps.stem(w) for w in tokens]
    return ' '.join(tokens)