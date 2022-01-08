import joblib
import json
import pickle
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from peewee import (SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
                    FloatField, TextField, IntegrityError)
from playhouse.shortcuts import model_to_dict

########################################
# Begin database stuff
import os
from playhouse.db_url import connect

# the connect function checks if there is a DATABASE_URL env var
# if it exists, it uses it to connect to a remote postgres db
# otherwise, it connects to a local sqlite db stored in the predictions.db file
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model

with open('columns.json') as fh:
    columns = json.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('validations.json') as fh:
    valid_columns = json.load(fh)

# End model un-pickling
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    request = request.get_json()
    try:
        _id = request['observation_id']
    except:
        _id = None
        error = 'observation_id'
        return {"observation_id": _id, "error": error}

    try:
        obs = request['data']
    except:
        error = 'data'
        return {"observation_id": _id, "error": error}

    for column, values in valid_columns.items():
        if column in obs:
            value = obs[column]
            if value not in values:
                error = column + ':' + str(value)
                return {"observation_id": _id, "error": error}
        else:
            error = column
            return {"observation_id": _id, "error": error}

    for column in obs.keys():
        if column not in valid_columns.keys():
            error = column
            return {"observation_id": _id, "error": error}

    df_obs = pd.DataFrame([obs], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(df_obs)[0, 1]
    prediction = pipeline.predict(df_obs)[0]

    response = {
        "observation_id": _id,
        "prediction": bool(prediction),
        "probability": proba
    }

    return response


    # What is this code doing? When we receive a new prediction request,
    # we want to store such request in our database (to keep track of our model performance).
    # With peewee, we save a new Prediction (basically a new row in our table) with the save()
    # method, which is very neat and convenient.

    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )

    # However, because our table has a unique constraint
    # (no two rows can have the same observation_id, which is a unique field),
    # if we perform the same prediction request twice (with the same id)
    # the system will crash because pewee can't save again an already saved observation_id;
    # it will throw an IntegrityError (as in, we would be asking pewee to violate the integrity
    # of the table's unique id requirement if we saved a duplicated id, right?).

    # To avoid that, we do a simple try/except block:
    # if we try a request with the same observation_id,
    # peewee will raise the integrity error and we'll catch it,
    # print a nice error message, and do a database rollback
    # (to close the current save transaction that has failed).

    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})

# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True)
