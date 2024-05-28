from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Tải dữ liệu NLTK nếu chưa được tải xuống
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Khởi tạo VADER cho phân tích cảm xúc
sa = SentimentIntensityAnalyzer()

# Tải tokenizer, cấu hình và mô hình RoBERTa
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyze_sentiment(text):
    # Xóa các chữ số và ký tự không phải là từ hoặc khoảng trắng
    text_final = re.sub(r'\d+', '', text)
    text_final = re.sub(r'[^\w\s]', '', text_final)
    # Tokenize văn bản
    words = word_tokenize(text_final)

    # Xóa các stopwords
    stop_words = set(stopwords.words('english'))
    processed_doc = ' '.join([word for word in words if word.lower() not in stop_words])

    # Phân tích cảm xúc sử dụng VADER
    dd_vader = sa.polarity_scores(processed_doc)

    # Phân tích cảm xúc sử dụng RoBERTa
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_roberta = output[0][0].detach().numpy()
    scores_roberta = softmax(scores_roberta)

    # Xác định nhãn cảm xúc và điểm số từ RoBERTa
    label_roberta = config.id2label[np.argmax(scores_roberta)]
    score_roberta = {
        'positive': scores_roberta[2],
        'neutral': scores_roberta[1],   
        'negative': scores_roberta[0]  
    }

    # Xác định điểm số VADER cao nhất
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
