from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Load the trained YOLOv8 model
model = YOLO("best.pt")

# Define issue-to-solution mapping
solutions = {
    "Drainage water": "Clean and maintain drainage systems regularly.",
    "Garbage Dumping": "Implement waste disposal regulations and promote recycling.",
    "Muddy Roads": "Use gravel or proper drainage to prevent muddy road conditions.",
    "Road Broken": "Repair roads promptly with quality asphalt or concrete.",
    "Road Damage": "Regular inspections and road maintenance are necessary.",
    "Soil Erosion": "Plant trees and install erosion barriers.",
    "Street Light": "Regular maintenance of street lights is needed.",
    "Street Light not working": "Fix faulty street lights for safety.",
    "Chemical Waste": "Dispose of hazardous chemicals following safety regulations.",
    "Road Water Leakage": "Fix water pipeline leaks to prevent damage.",
    "Water Leakage": "Inspect water pipelines regularly to detect leaks.",
    "Water Pollution": "Implement wastewater treatment and pollution control measures.",
    "road water leakage": "Repair water tanks to prevent water loss."
}

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Run YOLOv8 model on the image
    results = model(file_path)

    detected_problems = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            problem_name = model.names[class_id]
            detected_problems.add(problem_name)

    # Prepare response
    response = []
    for problem in detected_problems:
        response.append({
            "problem": problem,
            "solution": solutions.get(problem, "No solution available.")
        })

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
