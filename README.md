# Sentiment Analyzer Web App

A flask (Python) Web Interface for sentiment analysis using NLP techniques.

#### Basic Features
* Remove stop words 
* Pre-process text (remove punctuation, lower case)
* Stemming of words

#### Sentiment Analysis
* Shows how much text content is positive

### The Architecture
The machine learning model (built with  `scikit-learn`, and `NLTK Analyzer`) is deployable as a Python package and is placed behind an API written in `Flask`. I've built a simple front end using `HTML, CSS` to make the UI of Web App. I use `Docker` containers to isolate the API and Front End applications and use `Docker Compose` to deploy the whole system.

![image](https://github.com/Kudasai03/HPC-final/assets/114086290/be67096f-4427-418f-a74e-091a03e7313d)

# usage

1. First, clone this repository and open a terminal.
```bash
git clone https://github.com/Kudasai03/HPC-final.git
```

2. Navigate to the project directory
```bash
cd HPC-final/
```

3. Build and start docker containers (open Docker desktop first)
```bash
docker-compose up --build
```

4. Run the app: open your web browser (e.g. Chrome, Edge...) with command
```bash
http://localhost:5000
```
and to stop
```bash
ctrl + C
```

5. To run the application again, open terminal and run following command:
```bash
docker-compose up
```
