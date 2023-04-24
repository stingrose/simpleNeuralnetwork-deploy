import pandas as pd
from flask import Flask, request, render_template
import keras
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
# load the model from the h5 file
model = keras.models.load_model('model_part1.h5',compile=False)

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
        user_input_df=user_input_df.drop(['Unnamed: 0','target'],axis=1)


        scaler = StandardScaler()
        numcols=scaler.fit_transform(user_input_df)
        resdf = pd.DataFrame(numcols, columns=user_input_df.columns)

        tofitrow = resdf.iloc[-1]

        data = tofitrow.values

        reshaped_data = data.reshape((1, 7))

        prediction = model.predict(reshaped_data)

        threshold = 0.5

        if prediction >= threshold:
            predicted_class = 1
        else:
            predicted_class = 0


        # Format the prediction as a string and display it in the HTML
        prediction_str = "The predicted class label is: " + str(predicted_class)

        return render_template('result.html', prediction=prediction_str)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
