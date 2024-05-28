# Sentiment Analyzer Web App

A flask (Python) Web Interface for sentiment analysis.

#### Sentiment Analysis
* Displays the sentiment of the text.

### The Architecture
![image](https://github.com/Kudasai03/HPC-final/assets/114086290/15a7c4f2-00b7-494b-b0e6-cd99de7b8dd1)

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
use (if you want to run container in background)
```bash
docker-compose up -d --build
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
