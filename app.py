from flask import Flask, render_template, request
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the 5-star sentiment model (ML)
star_clf = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def get_star_rating(text):
    result = star_clf(text)[0]
    label = result['label']        # e.g., "4 stars"
    score = result['score']
    stars = int(label[0])          # get first char
    return stars, score

@app.route('/', methods=['GET', 'POST'])
def index():
    rating = None
    confidence = None
    warning = None

    if request.method == 'POST':
        user_text = request.form['user_input']
        if user_text.strip():
            stars, conf = get_star_rating(user_text)
            rating = stars
            confidence = round(conf, 2)
            if confidence < 0.5:
                warning = "⚠️ Low confidence — the input may be neutral or unclear."

    return render_template('index.html', rating=rating, confidence=confidence, warning=warning)

if __name__ == '__main__':
    app.run(debug=True)

