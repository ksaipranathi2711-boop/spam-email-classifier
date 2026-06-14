from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        message = request.form["message"]

        data = vectorizer.transform([message])

        try:
            result = model.predict(data)[0]

            if result == 1 or result == "spam":
                prediction = "Spam"
            else:
                prediction = "Ham"

        except:
            prediction = str(model.predict(data))

    return render_template("index.html", prediction=prediction)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)