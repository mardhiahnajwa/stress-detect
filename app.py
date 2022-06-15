import os
import random
from flask import Flask, request, jsonify
from stress_api import Stress_API
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    audio_file = request.files['file']
    print(audio_file)
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    #invoke keyword spotting service
    stress = Stress_API()
    predicted_level = stress.predict(file_name)

    os.remove(file_name)

    #send predicted in json
    data = {"keyword": predicted_level}

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
