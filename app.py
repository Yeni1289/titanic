from flask import Flask, render_template, request
import joblib
import numpy as np
import os

URLS = {
    "archivo1.pkl": "https://drive.google.com/uc?export=download&id=1Qu55sPYYveUGBcinzxgk3HTu7QEtXQl0",
    "modelo_titanic.pkl": "https://drive.google.com/uc?export=download&id=1jA2Bp_A0u-sv4hV4Rv34fu-Cd5ov8aLU",
    "titanic_rf_model.joblib": "https://drive.google.com/uc?export=download&id=1Qazikb3X9F1f5bCNOsDXAUUipgzbQKjm",
    "gender_submission.csv": "https://drive.google.com/uc?export=download&id=1D3MMkxYva40kdhQYvt9hmhEL39hduNH3",
    "test.csv": "https://drive.google.com/uc?export=download&id=1zDYntEh9QMMxeVUaKJyYpBkgyintWJIz",
    "train.csv": "https://drive.google.com/uc?export=download&id=1uR4Il5pQ8LwGgUXhv9a3u8az5fmvE8AD",
}

# Crear app Flask indicando dónde están templates y static
app = Flask(__name__, static_folder='static', template_folder='templates')

# Cargar el modelo solo una vez
MODEL_PATH = 'modelo_titanic.pkl'
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Ruta principal
@app.route('/', methods=['GET'])
def index():
    modelo_ok = model is not None
    return render_template('index.html', modelo_ok=modelo_ok)

# Ruta de predicción
@app.route('/predecir', methods=['POST'])
def predecir():
    if model is None:
        return render_template('index.html', modelo_ok=False, error="No se encontró el modelo 'modelo_titanic.pkl'.")

    try:
        pclass = int(request.form.get('pclass'))
        sexo = request.form.get('sexo')
        sex_num = 1 if sexo.lower() in ['femenino', 'mujer', 'female'] else 0
        edad = float(request.form.get('edad'))
        sibsp = int(request.form.get('sibsp'))
        parch = int(request.form.get('parch'))
        tarifa = float(request.form.get('tarifa'))

        embarcado = request.form.get('embarcado')
        emb = str(embarcado).strip().upper()
        if emb in ['C', 'CHERBURGO', 'CHERBOURGH']:
            emb_num = 0
        elif emb in ['Q', 'QUEENSTOWN']:
            emb_num = 1
        else:
            emb_num = 2  # Southampton por defecto

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


