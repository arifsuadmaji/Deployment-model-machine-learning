from flask import Flask, render_template, request
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('randomforest_klasifikasi.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Mapping dictionary for QUALCODE
qualcode_mapping = {
    'CQ2': 0,
    'CQ2MP': 1,
    'CQ2MP1': 2,
    'CQ2PT': 3,
    'CQ2PTU': 4,
    'CQ2UN': 5,
    'CQ2UN1': 6,
    'CQ39PT': 7,
    'CQ3': 8,
    'CQ3EN': 9,
    'CQ4': 10,
    'CQ5': 11,
    'CQ8D': 12,
    'CQ39': 13,
    'CQ45': 14,
    'CQUN': 15,
    'CQUN1': 16,
    'CQUN3': 17,
    'CQUN4': 18,
    'CQUN8C': 19,
    'CQUN8H': 20,
    'CQUN91': 21,
    'CQUNX': 22,
    'CQX': 23,
    'CQ-Z': 24,
    'DDQ1': 25,
    'DQ': 26,
    'DQUN91': 27,
    'HR1': 28
}

# Mapping of SPEC values to numeric order
spec_mapping = {
    'COMMERCIAL QUALITY': 0,
    'JIS G 3101 SS400': 1,
    'JIS G 3113 SAPH370': 2,
    'JIS G 3113 SAPH440': 3,
    'JIS G 3131 SPHC': 4,
    'JIS G 3131 SPHD': 5,
    'JIS G 3132 SPHT1': 6,
    'JIS G 3132 SPHT2': 7,
    'JIS G3141': 8,
    'KSAPH270C': 9,
    'KSA 29H': 10,
    'KSA 37H': 11,
    'KSA 39H': 12,
    'KSA29': 13,
    'KSA37': 14,
    'KSAPH270C': 15,
    'KNSS-1D': 16,
    'MP 38': 17,
    'MP 390': 18,
    'MP 440': 19,
    'MP1A': 20,
    'MP38': 21,
    'MPIC': 22,
    'MPW 2': 23,
    'SECONDARY': 24,
    'SNI 07 3567': 25,
    'SP121BQ': 26,
    'SPC': 27,
    'SPCG': 28,
    'SPCK-6': 29,
    'SPHG 450': 30,
    'TS G3100G': 31,
    'TS G3101G SPH270COD': 32,
    'TS G3101G SPH270DOD': 33,
    'TS G3101G SPH440OD': 34,
    'YSH270C-OP': 35,
    'JSH270C': 36,
    'MS 1705:2003 SPHC': 37
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    QUALCODE = request.form['QUALCODE']
    SPEC = request.form['SPEC']
    THICK = float(request.form['THICK'])
    WIDTH = float(request.form['WIDTH'])
    WEIGHT = float(request.form['WEIGHT'])

    # Convert QUALCODE and SPEC to numeric values
    qualcode_numeric = qualcode_mapping.get(QUALCODE, -1)
    spec_numeric = spec_mapping.get(SPEC, -1)

    if qualcode_numeric == -1 or spec_numeric == -1:
        result = "Hasil Prediksi Tidak Valid"
    else:
        # Scale the input features
        scaled_features = scaler.transform([[THICK, WIDTH, WEIGHT]])

        # Perform prediction using the loaded model
        prediction = model.predict([[qualcode_numeric, spec_numeric, scaled_features[0][0], scaled_features[0][1], scaled_features[0][2]]])

        if prediction[0] == 0:
            result = "HEAVY"
        elif prediction[0] == 1:
            result = "HRPO"
        elif prediction[0] == 2:
            result = "LITE"
        elif prediction[0] == 3:
            result = "MEDIUM"
        else:
            result = "Hasil Prediksi Tidak Valid"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
