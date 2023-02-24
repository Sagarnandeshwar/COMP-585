from flask import Flask, jsonify, request, Response
# from data_collection.rest_api_utils import get_user_info
from rsmodel import load_model, predict
from model.recbole.config import Config
from model.recbole.data import create_dataset, data_preparation
import pandas as pd
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)

metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.3')

model_path = "./model/saved/BPR1.0.pth"
current_version = ""
with open("model/saved/current_version.txt", "r") as f:
    current_version = f.read()
if current_version:
    model_path = "./model/saved/BPR" + current_version +".pth"
print("[API] loading model " + model_path)
model = load_model(model_path)
config = Config(model='MultiVAE', dataset='movie', config_file_list=['./model/movie.yaml'])
dataset = create_dataset(config)
_, _, test_data = data_preparation(config, dataset)
inter = pd.read_csv('./model/movie/movie.inter', delimiter='\t')
item = inter[["tmdb_id:token","movie_id:token_seq"]].drop_duplicates('tmdb_id:token')
user = inter["user_id:token"].to_list()
high_rating_movie = inter[inter["grade:float"]>=4].drop_duplicates(["tmdb_id:token","grade:float"])
high_rating_movie = high_rating_movie["movie_id:token_seq"].to_list()

@app.route('/recommend/<userid>')
def get_recommendations_for_user(userid):
    # checking we can access the util functions in data_collection
    userid = str(userid)
    r = predict(userid=userid, high_rating=high_rating_movie, item=item, user=user, 
                model=model, config=config, test_data=test_data, dataset=dataset)
    result = ','.join(r)
    result = Response(result)
    result.headers["X-Model-Version"] = current_version
    return result


# read file
with open('./data/model_performance.json', 'r') as myfile:
    data = myfile.read()

@app.route("/performance")
def index():
    return data

# driver function
if __name__ == '__main__':
    app.run(debug = False, host="0.0.0.0", port=8082)

    print("API READY")
