# Predicción de aprobación de estudiantes

Proyecto simple que incluye:

- Modelo de Machine Learning (Logistic Regression)
- Aplicación web en Flask
- Frontend simple con Bootstrap
- CI/CD con GitHub Actions
- Despliegue objetivo: Render

Estructura:

```
app.py
model.pkl (generado por train.py)
train.py
test.py
requirements.txt
# Predicción de aprobación de estudiantes

Proyecto simple que incluye:

- Modelo de Machine Learning (Logistic Regression / RandomForest)
- Aplicación web en Flask
- Frontend simple con Bootstrap
- CI/CD con GitHub Actions
- Despliegue objetivo: Render

Estructura importante:

```
app.py
model.pkl (generado por train.py)
train.py
test.py
requirements.txt
templates/index.html
.github/workflows/main.yml
Procfile
```

Cómo ejecutar localmente:

1. Crear un entorno virtual e instalar dependencias:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Entrenar el modelo:

```powershell
python train.py
```

3. Ejecutar la app:

```powershell
python app.py
```

La app estará en http://127.0.0.1:5000

Despliegue en Render
-------------------

1. Crea una cuenta en https://render.com y conecta tu repositorio de GitHub.

2. Crea un nuevo Web Service en Render y selecciona el repo y la rama `main`.

3. Configura los comandos:

	 - Build Command:
		 ```bash
		 pip install -r requirements.txt
		 python train.py
		 ```

	 - Start Command (Render detectará el `Procfile` que hemos añadido):
		 ```bash
		 gunicorn app:app
		 ```

4. Deploy: Render hará build y despliegue. La URL pública será algo como `https://tu-app.onrender.com`.

Notas:

- Para acelerar builds puedes entrenar localmente y commitear `model.pkl` y `metrics.json` (archivo pequeño). Para producción profesional usa almacenamiento de modelos en S3 u otro servicio.
- Asegúrate de que `requirements.txt` incluya `gunicorn` (ya lo añadimos).
