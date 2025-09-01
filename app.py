from flask import Flask, render_template, request, jsonify
import joblib
from tweets_analysis import predict_sentiment  # Import prediction function

app = Flask(__name__)

# Load the trained model when the app starts
model = joblib.load('sentiment_model.pkl')

# Home route: Renders the index.html page
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/team')
def team():
    return render_template('team.html')

# Sentiment analysis route: Accepts tweet input and returns sentiment
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    tweet = data.get("tweet")
    
    if not tweet:
        return jsonify({"error": "No tweet provided"}), 400

    # Use the model to predict sentiment
    sentiment = predict_sentiment(tweet)
    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
