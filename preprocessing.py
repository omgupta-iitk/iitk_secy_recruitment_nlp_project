from nltk.corpus import stopwords
import re, string
import nltk
#nltk.download('stopwords')

def preprocess(text):
    text = text.lower()
    text= text.strip()
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub('', text)
    text = re.sub('\s+', ' ', text) 
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text = re.sub(r'\s+',' ',text)
    return text

def stopword(string):
    words = string.split()
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(filtered_words)

def finalpreprocess(string):
    return (stopword(preprocess(string)))
