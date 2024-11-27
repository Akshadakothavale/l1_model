from flask import Blueprint, request, jsonify
from service.categories import classify_event 
from service.l1_model import train_l1_model
import os
import utils.logging as logger 

fetch_categories_blueprint = Blueprint('fetch_categories', __name__)

@fetch_categories_blueprint.route('/api/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        event_name = data.get('event_name')
        event_desc = data.get('event_description')

        if not event_name or not event_desc:
            return jsonify({"error": "Both 'eventName' and 'eventDesc' are required."}), 400
        
        description=f"{event_name} {event_desc}"
        result = classify_event(description)
        return jsonify({"data":result,"error":None })
    except Exception as e:
        logger.log_message(message=f"Exception while getting L1/L2 tags {e}", level="error")
        return jsonify({"data":None,"error":f"Exception while getting L1/L2 tags {e}" })


l1_model_blueprint = Blueprint('l1_model', __name__)

@l1_model_blueprint.route('/api/l1-train-model', methods=['POST'])
def L1_train_model_endpoint():
    try:
        # Check if a file is included in the request
        if 'file' not in request.files:
            logger.log_message("No file part in the request", level="error")
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        # Ensure a file is selected
        if file.filename == '':
            logger.log_message("No file selected", level="error")
            return jsonify({"error": "No file selected"}), 400

        # Save the file to a temporary directory
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)  # Directly use the filename
        file.save(file_path)
        logger.log_message(f"File {file.filename} saved successfully for training", level="info")

        # Call the service to train the model
        train_l1_model(file_path)

        # Clean up the uploaded file
        os.remove(file_path)
        logger.log_message(f"File {file.filename} removed after training", level="info")

        return jsonify({"message": "L1 model trained and saved successfully"}), 200

    except Exception as e:
        logger.log_message(f"Error training L1 model: {str(e)}", level="error")
        return jsonify({"error": str(e)}), 500