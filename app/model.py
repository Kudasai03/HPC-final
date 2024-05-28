from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize VADER for sentiment analysis
sa = SentimentIntensityAnalyzer()

# Load RoBERTa tokenizer, configuration and model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyze_sentiment(text):
    # Remove digits
    text_final = re.sub(r'\d+', '', text)
    text_final = re.sub(r'[^\w\s]', '', text_final)
    # Tokenize the text
    words = word_tokenize(text_final)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    processed_doc = ' '.join([word for word in words if word.lower() not in stop_words])

    # Sentiment analysis using VADER
    dd_vader = sa.polarity_scores(processed_doc)

    # Sentiment analysis using RoBERTa
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_roberta = output[0][0].detach().numpy()
    scores_roberta = softmax(scores_roberta)

    # Determine the sentiment label and score from RoBERTa
    label_roberta = config.id2label[np.argmax(scores_roberta)]
    score_roberta = {
        'positive': scores_roberta[2],
        'neutral': scores_roberta[1],   
        'negative': scores_roberta[0]  
    }

    # Determine the maximum VADER score
    max_vader_score = max(dd_vader['pos'], dd_vader['neu'], dd_vader['neg'])

    return {
        'pos_vader': dd_vader['pos'],
        'neu_vader': dd_vader['neu'],
        'neg_vader': dd_vader['neg'],
        'pos_roberta': score_roberta['positive'],
        'neu_roberta': score_roberta['neutral'],
        'neg_roberta': score_roberta['negative'],
        'maxvadersentiment': 'pos' if dd_vader['pos'] == max_vader_score else 'neu' if dd_vader['neu'] == max_vader_score else 'neg',
        'maxroberta': label_roberta
    }