from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)


models = {}
categories = ['hate_speech', 'offensive_language', 'neither']
for category in categories:
    with open(f'{category}_mark6.pkl', 'rb') as f:
        models[category] = pickle.load(f)


with open('cMark6.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():

    text = request.form.get('text')


    text_vectorized = vectorizer.transform([text])


    predictions = {}
    for category, model in models.items():
        predictions[category] = model.predict_proba(text_vectorized)[0][1]

    result= " "


    if (predictions['offensive_language']>predictions['hate_speech'] and predictions['offensive_language']>predictions['neither']):
        #result['expression']='offensive'
        result ="offensive"+str(predictions)
    elif(predictions['hate_speech']>predictions['neither']):
        #result['expression'] = 'hate'
        result ="hate"+str(predictions)
    else:
        #result['expression'] = 'positive'
        result = "positive"+str(predictions)


    return result

if __name__ == '__main__':
    app.run(debug=True)
