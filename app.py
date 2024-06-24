from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('model_KR_dt.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        Fine_Aggr = float(request.form['Fine_Aggr'])
        Coarse_Aggr = float(request.form['Coarse_Aggr'])
        Slag = float(request.form['Slag'])
        Water = float(request.form['Water'])
        Cement = float(request.form['Cement'])
        Fly_ash = float(request.form['Fly_ash'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[Fine_Aggr, Coarse_Aggr, Slag, Water, Cement, Fly_ash]],
         columns=['Fine_Aggr', 'Coarse_Aggr', 'Slag', 'Water', 'Cement', 'Fly_ash'])

        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Convertir la predicción a un tipo de datos nativo de Python
        numericPrediction = float(prediction[0])

        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': numericPrediction})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

