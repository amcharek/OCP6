from flask import Flask
from flask import request
from flask import jsonify
from flask import json
import scipy.sparse
from nltk.tokenize.treebank import TreebankWordDetokenizer
import boto3
import pickle

BUCKET_NAME = 'projetopenclassrooms/OCP6/models/'
m_bigram_mod = 'bigram_mod.pk'
m_trigram_mod = 'trigram_mod.pk'
m_lemmatizer = 'lemmatizer.pk'
m_p_stemmer = 'bigram_mod.pk'
m_mlb_obj = 'mlb_obj.pk'
m_tokenizer = 'tokenizer.pk'
m_svdt = 'svdt.sav'
m_sgd_clf = 'sgd_clf.sav'
m_engstpw = 'engstpw.txt'

app = Flask(__name__)

S3 = boto3.client('s3', region_name='eu-west-1')
rs3 = boto3.resource('s3')

# fonction pour le TF_IDF_Vectorize
def text_splitter(text):
    return text.split(' ')
	

def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


@app.route('/predict_tags/', methods=['GET'])
def home():
    body = request.get_json(silent=True)
    data = body['data']

    prediction = predict(data)

    result = {'prediction': prediction}
    return json.dumps(result)


@memoize
def load_model(key):
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    model_str = response['Body'].read()

    model = pickle.loads(model_str)

    return model


def predict(data):

	bigram_mod = load_model(m_bigram_mod)
	trigram_mod = load_model(m_trigram_mod)
	lemmatizer = load_model(m_lemmatizer)
	p_stemmer = load_model(m_p_stemmer)
	mlb_obj = load_model(m_mlb_obj)
	tokenizer = load_model(m_tokenizer)
	svdt = load_model(m_svdt)
	sgd_clf = load_model(m_sgd_clf)
	engstpw = S3.get_object(Bucket=BUCKET_NAME, Key=m_engstpw)
	engstpwrd = [line.rstrip('\n') for line in engstpw]

	
	q = tokenizer.tokenize(data.lower())
    q = [token for token in q if token not in engstpwrd]
    q = [lemmatizer.lemmatize(token) for token in q]
    q = [p_stemmer.stem(token) for token in q]    
    q = trigram_mod[bigram_mod[q]]
    q = Tvectorizer.transform([TreebankWordDetokenizer().detokenize(q)])
    q = svdt.transform(q)
    q = scipy.sparse.csr_matrix(q)
    q = sgd_clf.predict(q)
    q = mlb_object.inverse_transform(q)

    return q.tolist()


if __name__ == '__main__':
    # listen on all IPs
    app.run(host='0.0.0.0')
