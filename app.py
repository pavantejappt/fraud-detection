from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pre-trained LabelEncoder from a file
with open('/config/workspace/.vscode/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
# Load the pre-trained machine learning model from a file
with open('/config/workspace/.vscode/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    step = request.form['step']
    customer = request.form['customer']
    age = request.form['age']
    gender = request.form['gender']
    merchant = request.form['merchant']
    category = request.form['category']
    amount = request.form['amount']

    # Encode the categorical columns using the pre-trained LabelEncoder
    category = encoder.transform([category])[0]

    # Use the encoded values as needed
    features = [step, customer, age, gender, merchant, category, amount]

    # Make a prediction using the pre-trained machine learning model
    prediction = model.predict([features])[0]

    # Pass the prediction as data to the result.html template
    return render_template('result.html', prediction=prediction)



if __name__=="__main__":
    app.run(host="0.0.0.0")
