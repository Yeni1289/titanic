from flask import Flask, render_template, request 
import joblib
import numpy as np
import os

app = Flask(__name__)

# Intentar cargar el modelo y si no existe, avisar claramente
MODEL_PATH = 'modelo_titanic.pkl'
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/', methods=['GET'])
def index():
    modelo_ok = model is not None
    return render_template('index.html', modelo_ok=modelo_ok)

@app.route('/predecir', methods=['POST'])
def predecir():
    if model is None:
        return render_template('index.html', modelo_ok=False, error="No se encontró el modelo. Ejecuta el notebook para generar 'modelo_titanic.pkl'.")

    try:
        # Leer formulario
        pclass_str = request.form.get('pclass')
        if not pclass_str:
            raise ValueError("Debes seleccionar la clase de billete")
        pclass = int(pclass_str)

        sexo = request.form.get('sexo')
        if not sexo:
            raise ValueError("Debes seleccionar el sexo")
        sex_num = 1 if sexo.lower() in ['femenino', 'mujer', 'female'] else 0

        edad = float(request.form.get('edad'))
        sibsp = int(request.form.get('sibsp'))
        parch = int(request.form.get('parch'))
        tarifa = float(request.form.get('tarifa'))

        embarcado = request.form.get('embarcado')
        if not embarcado:
            raise ValueError("Debes seleccionar el puerto de embarque")

        emb = str(embarcado).strip().upper()
        if emb in ['C', 'CHERBURGO', 'CHERBOURGH']:
            emb_num = 0
        elif emb in ['Q', 'QUEENSTOWN']:
            emb_num = 1
        else:
            emb_num = 2  # Southampton por defecto

        # ✅ Usar solo 7 columnas (las mismas del modelo)
        x = np.array([[pclass, sex_num, edad, sibsp, parch, tarifa, emb_num]])

        # Predecir
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True, ssl_context=('cert.pem', 'key.pem'))


#https://127.0.0.1:5050/

