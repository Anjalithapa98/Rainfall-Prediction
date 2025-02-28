from flask import Flask, request, jsonify
import predict

app = Flask(__name__)

@app.route('/')
def data():
    data_columns = predict.column_names
    data_values = list(predict.weather_data)
    data_columns.append("rainfallChance")
    data_values.append(round(predict.probability(),2))
    print(data_columns)
    print(data_values)
    data=dict(zip(data_columns, data_values))
    print(data)

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)