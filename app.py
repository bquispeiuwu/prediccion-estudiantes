from flask import Flask, render_template, request
import pickle
import json
import os

app = Flask(__name__)

MODEL_PATH = 'model.pkl'
METRICS_PATH = 'metrics.json'

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

model = load_model()
metrics = load_metrics()
MODEL_NAME = metrics.get('model') if isinstance(metrics, dict) else None
FEATURE_IMPORTANCES = metrics.get('feature_importances', {}) if isinstance(metrics, dict) else {}

@app.route('/')
def home():
    return render_template('index.html', resultado=None, metrics=metrics, model_name=MODEL_NAME, feature_importances=FEATURE_IMPORTANCES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        horas = float(request.form.get('horas', 0))
        asistencia = float(request.form.get('asistencia', 0))
        tareas = float(request.form.get('tareas', 0))
        promedio = float(request.form.get('promedio', 0))
    except ValueError:
        return render_template('index.html', resultado='Entrada inválida', metrics=metrics)

    # If sklearn model is available use it, otherwise use a simple heuristic
    if model is not None:
        try:
            prediction = model.predict([[horas, asistencia, tareas, promedio]])[0]
            resultado = "Aprobará" if int(prediction) == 1 else "Desaprobará"
            note = f'Model pipeline: {MODEL_NAME}' if MODEL_NAME else 'Model pipeline loaded'
        except Exception:
            resultado = 'Error al predecir con el modelo guardado'
            note = None
    else:
        # Heuristic fallback: weighted score (normalized attendance and promedio)
        score = 0.4 * (horas / 8.0) + 0.35 * (asistencia / 100.0) + 0.15 * (tareas / 8.0) + 0.1 * (promedio / 20.0)
        resultado = "Aprobará" if score >= 0.5 else "Desaprobará"
        note = 'Modelo heurístico (sin scikit-learn)'

    return render_template('index.html', resultado=resultado, metrics=metrics, note=note, model_name=MODEL_NAME, feature_importances=FEATURE_IMPORTANCES)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)