import pickle
import pandas as pd
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
        columns = data['columns']
    return model, columns

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form_data = {
            'Rating': float(request.form['rating']),
            'Size': request.form['size'],
            'Type of ownership': request.form['type_of_ownership'],
            'Industry': request.form['industry'],
            'Sector': request.form['sector'],
            'Revenue': request.form['revenue'],
            'num_comp': int(request.form['num_comp']),
            'hourly': int(request.form['hourly']),
            'employer_provided': int(request.form['employer_provided']),
            'job_state': request.form['job_state'],
            'same_state': int(request.form['same_state']),
            'age': int(request.form['age']),
            'python_yn': int(request.form['python_yn']),
            'spark': int(request.form['spark']),
            'aws': int(request.form['aws']),
            'excel': int(request.form['excel']),
            'job_simp': request.form['job_simp'],
            'seniority': request.form['seniority'],
            'desc_len': int(request.form['desc_len'])
        }

        df = pd.DataFrame([form_data])
        df_dum = pd.get_dummies(df, dtype=int)

        model, columns = load_models()
        df_dum = df_dum.reindex(columns=columns, fill_value=0)

        x_in = df_dum.values
        prediction = model.predict(x_in)[0]

        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
