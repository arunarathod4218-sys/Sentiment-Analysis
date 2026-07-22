from flask import Flask, render_template, request
from model import train_model

app = Flask(__name__)

model, vectorizer = train_model()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)

    if prediction[0] == 1:
        result = "😊 Positive Review"
    else:
        result = "😞 Negative Review"

    return render_template(
        "index.html",
        review=review,
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)