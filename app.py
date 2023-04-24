import pickle
import pandas as pd
import xgboost as xgb
from flask import Flask, request, render_template
import keras
from sklearn.preprocessing import StandardScaler




app = Flask(__name__)


# load the model from the h5 file
model = keras.models.load_model('app/model_part1.h5')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the user input data as a dictionary
        input_data = request.form.to_dict()

        user_input_df = pd.read_csv('UpsampledDF.csv')

        new_row = {'f1': input_data['Feature1'], 
                   'f2': input_data['Feature2'], 
                   'f3': input_data['Feature3'],
                   'f4': input_data['Feature4'],
                   'f5': input_data['Feature5'],
                   'f6': input_data['Feature6'],
                   'f7': input_data['Feature7'],
                   }

        user_input_df = user_input_df.append(new_row, ignore_index=True)
        last_row = user_input_df.iloc[-1]

        X = user_input_df.drop('target', axis=1)

        scaler = StandardScaler()

        numcols=scaler.fit_transform(X)

        resdf = pd.DataFrame(numcols, columns=X.columns)
        tofitrow = resdf.iloc[-1]

        prediction = model.predict(tofitrow)
        # Format the prediction as a string and display it in the HTML
        prediction_str = 'The predicted flight delay is {:.2f} minutes.'.format(prediction[0])
        return render_template('result.html', prediction=prediction_str)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
