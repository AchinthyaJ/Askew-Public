from flask import Flask, request, jsonify, render_template
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
   
@app.route("/")
def home():
    return render_template("index.html")



with open('intents.json', 'r') as f:
    intents = json.load(f)

patterns = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, tags)

def get_response(user_input):
    user_input_vec = vectorizer.transform([user_input])
    tag = clf.predict(user_input_vec)[0]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I am not sure how to respond to that."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    response = get_response(user_input)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
