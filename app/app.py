from flask import Flask, request, render_template
from model import analyze_sentiment

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def my_form():
    return render_template('landing.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text1'].lower()
    results = analyze_sentiment(text)
    return render_template('landing.html', results=results, text1=text)

if __name__ == "__main__":
    print(f"**Running Flask App on http://localhost:5000 (**Debug Mode**)**")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
