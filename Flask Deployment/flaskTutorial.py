from flask import Flask, request, jsonify, render_template
from ml.ml import LanguageModel

app = Flask(__name__)

@app.route('/') #http://google.com/
def home():
    return render_template('index.html')

@app.route('/shakespeare', methods=['POST'])
def shakespeare():
    data = request.json
    gen_count = data['gen_count']
    lang = LanguageModel(3)
    answer = lang.get_generated_text(gen_count)
    return jsonify({'answer':answer})

app.run(port=5000)