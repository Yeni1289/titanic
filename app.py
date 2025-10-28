from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import requests
import time

# --- Archivos en Drive (usa los IDs que ya subiste) ---
URLS = {
    "modelo_titanic.pkl": "https://drive.google.com/uc?export=download&id=1jA2Bp_A0u-sv4hV4Rv34fu-Cd5ov8aLU",
    "titanic_rf_model.joblib": "https://drive.google.com/uc?export=download&id=1Qazikb3X9F1f5bCNOsDXAUUipgzbQKjm",
    "gender_submission.csv": "https://drive.google.com/uc?export=download&id=1D3MMkxYva40kdhQYvt9hmhEL39hduNH3",
    "test.csv": "https://drive.google.com/uc?export=download&id=1zDYntEh9QMMxeVUaKJyYpBkgyintWJIz",
    "train.csv": "https://drive.google.com/uc?export=download&id=1uR4Il5pQ8LwGgUXhv9a3u8az5fmvE8AD",
}

def download_file_if_missing(filename, url, max_tries=3):
    if os.path.exists(filename):
        print(f"[INFO] {filename} ya existe, no se descargará.")
        return True
    print(f"[INFO] Descargando {filename}...")
    for attempt in range(1, max_tries+1):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)
            print(f"[INFO] {filename} descargado correctamente.")
            return True
        except Exception as e:
            print(f"[WARN] intento {attempt} fallo: {e}")
            time.sleep(2)
    print(f"[ERROR] No se pudo descargar {filename}.")
    return False

# Descargar archivos necesarios al iniciar (solo si faltan)
for fname, furl in URLS.items():
    download_file_if_missing(fname, furl)

# Crear app Flask
app = Flask(__name__, static_folder='static', template_folder='templates')

# Cargar modelo (si existe)
MODEL_PATH = 'modelo_titanic.pkl'
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("[INFO] Modelo cargado correctamente.")
    except Exception as e:
        print(f"[ERROR] no se pudo cargar {MODEL_PATH}: {e}")
        model = None
else:
    print("[WARN] No existe el modelo en el filesystem.")

@app.route('/', methods=['GET'])
def index():
    modelo_ok = model is not None
    return render_template('index.html', modelo_ok=modelo_ok)

@app.route('/predecir', methods=['POST'])
def predecir():
    if model is None:
        return render_template('index.html', modelo_ok=False, error="No se encontró el modelo 'modelo_titanic.pkl'.")

    try:
        pclass = int(request.form.get('pclass'))
        sexo = request.form.get('sexo') or ''
        sex_num = 1 if sexo.lower() in ['femenino', 'female', 'f'] else 0
        edad = float(request.form.get('edad') or 0)
        sibsp = int(request.form.get('sibsp') or 0)
        parch = int(request.form.get('parch') or 0)
        tarifa = float(request.form.get('tarifa') or 0.0)

        embarcado = request.form.get('embarcado') or 'S'
        emb = str(embarcado).strip().upper()
        emb_num = 0 if emb == 'C' else (1 if emb == 'Q' else 2)

        # Formato de entrada que espera tu modelo
        x = np.array([[pclass, sex_num, edad, sibsp, parch, tarifa, emb_num]])
        pred = model.predict(x)[0]
        prob = model.predict_proba(x)[0][1] if hasattr(model, "predict_proba") else None

        resultado_text = "🟢 ¡Sobrevivirías!" if pred == 1 else "🔴 No sobrevivirías..."
        confianza = f"{prob*100:.1f}%" if prob is not None else "N/A"

        razones = []
        if sex_num == 1:
            razones.append("Sexo: mujer (aumenta probabilidad)")
        if pclass == 1:
            razones.append("Clase: 1ª (aumenta probabilidad)")
        if edad < 12:
            razones.append("Edad: niño (aumenta probabilidad)")
        explicacion = ", ".join(razones) if razones else "El modelo usa varias características para decidir."

        return render_template('index.html', modelo_ok=True, resultado=resultado_text, confianza=confianza, explicacion=explicacion)

    except Exception as e:
        return render_template('index.html', modelo_ok=True, error=f"Error al procesar los datos: {e}")

# Solo para ejecución local (no usado por gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
