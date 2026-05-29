import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle


def make_synthetic_data(n=500, random_state=42):
    rng = np.random.RandomState(random_state)
    horas = rng.normal(loc=5, scale=2, size=n).clip(0, 12)
    asistencia = rng.normal(loc=75, scale=15, size=n).clip(0, 100)
    tareas = rng.poisson(lam=5, size=n).clip(0, 12)
    promedio = rng.normal(loc=11, scale=2.5, size=n).clip(0,20)

    # create a latent score and threshold to determine label
    score = 0.4 * (horas / 12.0) + 0.35 * (asistencia / 100.0) + 0.15 * (tareas / 12.0) + 0.1 * (promedio / 20.0)
    # add noise
    score += rng.normal(scale=0.08, size=n)
    resultado = (score > 0.5).astype(int)

    df = pd.DataFrame({
        'horas': horas,
        'asistencia': asistencia,
        'tareas': tareas,
        'promedio': promedio,
        'resultado': resultado
    })
    return df


FEATURES = ['horas', 'asistencia', 'tareas', 'promedio']

df = make_synthetic_data(n=500)

X = df[FEATURES]
y = df['resultado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define pipelines
pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
pipe_rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])

# Train both
pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

# Evaluate
def eval_pipe(pipeline, X_t, y_t):
    y_pred = pipeline.predict(X_t)
    return {
        'accuracy': float(accuracy_score(y_t, y_pred)),
        'precision': float(precision_score(y_t, y_pred, zero_division=0)),
        'recall': float(recall_score(y_t, y_pred, zero_division=0))
    }

metrics_lr = eval_pipe(pipe_lr, X_test, y_test)
metrics_rf = eval_pipe(pipe_rf, X_test, y_test)

# Choose best model by accuracy
best_model = ('LogisticRegression', pipe_lr, metrics_lr) if metrics_lr['accuracy'] >= metrics_rf['accuracy'] else ('RandomForest', pipe_rf, metrics_rf)
model_name, model_pipeline, chosen_metrics = best_model

# Feature importances: for LR use absolute coef, for RF use feature_importances_
importances = {}
try:
    if model_name == 'RandomForest':
        imps = model_pipeline.named_steps['clf'].feature_importances_
    else:
        coefs = model_pipeline.named_steps['clf'].coef_[0]
        imps = np.abs(coefs)
    # normalize to sum 1
    imps = imps / imps.sum() if imps.sum() > 0 else imps
    importances = {f: float(round(float(v), 4)) for f, v in zip(FEATURES, imps)}
except Exception:
    importances = {f: None for f in FEATURES}

metrics_to_save = {
    'model': model_name,
    'metrics': {k: round(v, 4) for k, v in chosen_metrics.items()},
    'feature_importances': importances
}

# Save pipeline and metrics
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

with open('metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)

print('Modelo entrenado:', model_name)
print('Metrics:', metrics_to_save['metrics'])
print('Importances:', metrics_to_save['feature_importances'])