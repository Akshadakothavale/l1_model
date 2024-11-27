from flask import Flask
from flask_cors import CORS
from routes.category_controller import l1_model_blueprint
from utils.logging import setup_logging
setup_logging()

app = Flask(__name__)
CORS(app)


app.register_blueprint(l1_model_blueprint)

@app.route('/', methods=['GET'])
def getHome():
    return {"message": "Taxonomy server is running..."}

if __name__ == '__main__':
    app.run(debug=True, port=5005,host="0.0.0.0")
