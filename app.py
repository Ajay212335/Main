from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from flask import session
from werkzeug.security import generate_password_hash, check_password_hash
import os
import jwt
import datetime
from random import randint
from flask_mail import Mail, Message
import random
from werkzeug.utils import secure_filename
from bson import ObjectId
import uuid
import requests 
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, render_template
from PIL import Image
from model import create_model
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'ajaiks2005@gmail.com'  
app.config['MAIL_PASSWORD'] = 'lijk wygx arsw ghna' 


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "ajaiks2005@gmail.com"
EMAIL_PASSWORD = "lijk wygx arsw ghna"

mail = Mail(app)
CORS(app)

CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

MONGO_URI = "mongodb+srv://ddarn3681:eyl349H2RkqaraZb@cluster0.ezhvpef.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

try:
    # Connect to MongoDB Atlas
    client = MongoClient(MONGO_URI)

    # Select Database and Collections
    db = client['MainProject']
    contacts_collection = db['contacts']
    users_collection = db['users']
    officers_collection = db['officers']
    complaints_collection = db['complaints']

    print("Connected to MongoDB Atlas successfully!")

except Exception as e:
    print("Error connecting to MongoDB:", e)

otp_storage = {}
def save_file(file, directory):
    if file:
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, file.filename)
        file.save(filepath)
        return filepath
    return None
API_KEY = "AIzaSyCvIOVnZUf7M-0d3IoHpVzy6zFfR5PTKW4"
chatbot_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
# Define rural-related keywords
rural_keywords = [
    "rural", "village", "farm", "agriculture", "countryside", "remote", 
    "water", "hi", "powercut", "power", "problem"
]

# Define quick response for simple greetings
QUICK_RESPONSES = {
    "hi": "Hello!,How can I help you?",
    "hello": "Hi!,How can I help you?",
    "hey": "Hey!,How can I help you?",
    "thanks": "You're welcome!",
    "bye": "Goodbye!"
}

def get_gemini_response(user_query, src_lang):
    query_lower = user_query.lower()

    # Check for quick responses
    if query_lower in QUICK_RESPONSES:
        return QUICK_RESPONSES[query_lower]

    # Handle rural problems using Gemini API
    if any(keyword in query_lower for keyword in rural_keywords):
        headers = {"Content-Type": "application/json"}
        prompt = f"Identify the rural problem and provide a practical solution. Query: {user_query}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        try:
            response = requests.post(f"{chatbot_url}?key={API_KEY}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            gemini_response = parse_response(data)

            # Translate response back to original language
            translated_response = GoogleTranslator(source='en', target=src_lang).translate(gemini_response)
            return translated_response
        
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
    else:
        return "This bot only handles rural area problems."



def parse_response(data):
    try:

        response = data["candidates"][0]["content"]["parts"][0]["text"]
        # Remove "*" from the response
        cleaned_response = response.replace("*", "")
        return cleaned_response
    except (KeyError, IndexError):
        return "Sorry, I couldn't process your request."

# Flask route for chatbot
@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"response": "Please enter a valid query."})

    # Detect input language using langdetect
    try:
        detected_lang = detect(user_query)
    except Exception as e:
        return jsonify({"response": f"Error detecting language: {e}"})

    # Translate user query to English
    translated_query = GoogleTranslator(source=detected_lang, target='en').translate(user_query)

    # Get chatbot response
    response = get_gemini_response(translated_query, detected_lang)

    return jsonify({"response": response})

CLASS_LABELS = {
    0: "Water Scarcity",
    1: "Electricity Issues",
    2: "Poor Road Conditions",
    3: "Garbage Disposal Problems",
    4: "Healthcare Access Problems"
}

# Load trained model
NUM_CLASSES = len(CLASS_LABELS)  # Ensure this matches training setup
MODEL_PATH = "rural_problem_model.pth"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first.")

# Load model
model = create_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = YOLO("best.pt")

# Define issue-to-solution mapping
solutions = {
    "Drainage water": "Clean and maintain drainage systems regularly.call : 8610904744",
    "Garbage Dumping": "Implement waste disposal regulations and promote recycling.call : 8610904744",
    "Muddy Roads": "Use gravel or proper drainage to prevent muddy road conditions.call : 8610904744",
    "Road Broken": "Repair roads promptly with quality asphalt or concrete.call : 8610904744",
    "road damage": "Regular inspections and road maintenance are necessary.",
    "Soil Erosion": "Plant trees and install erosion barriers.call : 8610904744",
    "Street Ligth": "Regular maintenance of street lights is needed.call : 8610904744",
    "Street Light not working": "Fix faulty street lights for safety.call : 8610904744",
    "chemical waste": "Dispose of hazardous chemicals following safety regulations.call : 8610904744",
    "Road Water Leakage": "Fix water pipeline leaks to prevent damage. call : 8610904744",
    "Water Leakage": "Inspect water pipelines regularly to detect leaks.call : 8610904744",
    "Water Pollution": "Implement wastewater treatment and pollution control measures.call : 8610904744",
    "road water leakage": "Repair water tanks to prevent water loss.call : 8610904744"
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

@app.route("/api/sendOtp", methods=["POST"])
def send_otp():
    email = request.json.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400
    otp = randint(100000, 999999)
    otp_storage[email] = otp
    try:
        msg = Message('Your OTP Code', sender='ajaiks2005@gmail.com', recipients=[email])
        msg.body = f'Your OTP code is {otp}.'
        mail.send(msg)
        return jsonify({"message": "OTP sent successfully"}), 200
    except Exception as e:
        print(f"Error sending email: {e}")
        return jsonify({"error": "Failed to send OTP"}), 500


@app.route("/api/login/sendOtp", methods=["POST"])
def send_login_otp():
    email = request.json.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400

    otp = randint(100000, 999999)
    otp_storage[email] = otp

    try:
        msg = Message('Your OTP Code for Login', sender='ajaiks2005@gmail.com', recipients=[email])
        msg.body = f'Your OTP code is {otp}.'
        mail.send(msg)
        return jsonify({"message": "OTP sent successfully"}), 200
    except Exception as e:
        print(f"Error sending email: {e}")
        return jsonify({"error": "Failed to send OTP"}), 500
    
@app.route("/api/user/register", methods=["POST"])
def user_register():
    try:
        data = request.json
        email = data.get("email")
        otp = data.get("otp")

        if otp != str(otp_storage.get(email)):
            return jsonify({"error": "Invalid OTP"}), 400
        otp_storage.pop(email, None)
        hashed_password = generate_password_hash(data["password"], method="pbkdf2:sha256")
        user = {
            "name": data["name"],
            "dob": data["dob"],
            "gender": data["gender"],
            "phone": data["phone"],
            "email": data["email"],
            "address": data["address"],
            "pincode": data["pincode"],
            "password": hashed_password
        }
        users_collection.insert_one(user)

        return jsonify({"message": "User registered successfully."}), 201
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Internal Server Error. Please try again later."}), 500

@app.route("/api/user/login", methods=["POST"])
def user_login():
    try:
        data = request.json
        print("Received login request:", data)  # Debug log

        # Step 1: Find user in the database
        user = users_collection.find_one({"email": data["email"]})
        if not user:
            print("User not found")
            return jsonify({"error": "User not found."}), 401

        # Step 2: Check password
        if not check_password_hash(user["password"], data["password"]):
            print("Invalid password")
            return jsonify({"error": "Invalid password."}), 401

        # Step 3: Check OTP
        otp = data.get("otp")
        stored_otp = otp_storage.get(data["email"])
        print(f"Received OTP: {otp}, Stored OTP: {stored_otp}")  # Debug log

        if not otp or otp != str(stored_otp):
            print("Invalid OTP")
            return jsonify({"error": "Invalid or missing OTP."}), 401

        # Step 4: Remove OTP after use
        otp_storage.pop(data["email"], None)

        # Step 5: Generate JWT Token
        token_payload = {
            "user_id": str(user["_id"]),
            "name": user["name"], 
            "email": user["email"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
        }
        
        token = jwt.encode(token_payload, "your-secret-key", algorithm="HS256")
        print("Generated token:", token)  # Debug log

        return jsonify({"message": "Login successful.", "token": token}), 200

    except Exception as e:
        import traceback
        print("Error occurred during login:", e)
        print(traceback.format_exc())  # Print full error details
        return jsonify({"error": "Login failed."}), 500


@app.route('/user/dashboard')
def user_dashboard():
    return redirect(url_for('user/dashboard'))

@app.route("/api/officer/register", methods=["POST"])
def officer_register():
    try:
        data = request.form
        files = request.files

        email = data.get("email")
        otp = data.get("otp")

        # Validate OTP
        if otp != str(otp_storage.get(email)):
            return jsonify({"error": "Invalid OTP"}), 400
        
        # Remove OTP after verification
        otp_storage.pop(email, None)

        # Hash password
        hashed_password = generate_password_hash(data["password"], method="pbkdf2:sha256")

        # Handle file uploads
        gov_id_proof_path = save_file(files.get("govIdProof"), "gov_id_proofs")
        photo_path = save_file(files.get("photo"), "photos")
        certificate_photo_path = save_file(files.get("certificatePhoto"), "certificates")
        appointment_photo_path = save_file(files.get("appointmentPhoto"), "appointments")

        if not all([gov_id_proof_path, photo_path, certificate_photo_path, appointment_photo_path]):
            return jsonify({"error": "One or more files failed to upload."}), 400

        officer = {
            "fullName": data["fullName"],
            "designation": data["designation"],
            "department": data["department"],
            "officeAddress": data["officeAddress"],
            "phone": data["phone"],
            "email": data["email"],
            "password": hashed_password,
            "employeeId": data["employeeId"],
            "govIdProof": gov_id_proof_path,
            "photo": photo_path,
            "region": data["region"],
            "subdivision": data["subdivision"],
            "village": data["village"],
            "experience": data["experience"],
            "specialization": data.get("specialization", ""),
            "supervisor": data["supervisor"],
            "certificatePhoto": certificate_photo_path,
            "certificateNumber": data["certificateNumber"],
            "certificateName": data["certificateName"],
            "appointmentPhoto": appointment_photo_path,
            "yearOfJoining": data["yearOfJoining"]
        }

        # Insert officer data into the database (make sure to use your own MongoDB collection)
        officers_collection.insert_one(officer)

        return jsonify({"message": "Officer registered successfully."}), 201

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Registration failed."}), 500  





@app.route("/api/officer/login/sendOtp", methods=["POST"])
def send_officer_login_otp():
    email = request.json.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400
    
    # Generate OTP
    otp = random.randint(100000, 999999)
    otp_storage[email] = otp

    # Send OTP via email using Flask-Mail
    try:
        msg = Message('Your OTP Code for Officer Login', sender='ajaiks2005@gmail.com', recipients=[email])
        msg.body = f'Your OTP code is {otp}.'
        mail.send(msg)
        return jsonify({"message": "OTP sent successfully"}), 200
    except Exception as e:
        print(f"Error sending OTP: {e}")
        return jsonify({"error": "Failed to send OTP"}), 500
    
    
@app.route("/api/officer/login", methods=["POST"])
def officer_login():
    try:
        data = request.json
        officer = officers_collection.find_one({"email": data["email"]})

        if not officer:
            return jsonify({"error": "Officer not found."}), 401

        # Validate password
        if not check_password_hash(officer["password"], data["password"]):
            return jsonify({"error": "Invalid password."}), 401

        # Validate OTP
        otp = data.get("otp")
        stored_otp = otp_storage.get(data["email"])

        # Compare OTPs
        if not otp or int(otp) != stored_otp:
            return jsonify({"error": "Invalid or missing OTP."}), 401

        # Remove OTP after successful verification
        otp_storage.pop(data["email"], None)

        # Generate JWT token
        token = jwt.encode(
            {
                "officer_id": str(officer["_id"]),
                "fullName": officer["fullName"],
                "email": officer["email"],
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
            },
            "your-secret-key",  # Replace with your actual secret key
            algorithm="HS256",
        )

        return jsonify({"message": "Login successful.", "token": token, "fullName": officer["fullName"]}), 200

    except Exception as e:
        print(f"Error occurred during officer login: {e}")
        return jsonify({"error": "Login failed."}), 500


@app.route('/officer/dashboard')
def officer_dashboard():
    return redirect(url_for('officer/dashboard'))

# Existing contact route
@app.route('/api/contact', methods=['POST'])
def contact():
    try:
        data = request.get_json()
        contact_data = {
            "fullName": data.get("fullName"),
            "phone": data.get("phone"),
            "email": data.get("email"),
            "message": data.get("message")
        }
        contacts_collection.insert_one(contact_data)
        return jsonify({"message": "Your Request Submitted Successfully!"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "There was an error submitting the form"}), 500



# Load NLP model (GPT-based)







UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Route to submit a new complaint
@app.route("/submit_complaint", methods=["POST"])
def submit_complaint():
    name = request.form.get("name")
    phone = request.form.get("phone")
    aadhar = request.form.get("aadhar")
    address = request.form.get("address")
    district = request.form.get("district")
    pincode = request.form.get("pincode")
    complaint = request.form.get("complaint")
    location = request.form.get("location")
    ministry = request.form.get("ministry")
    file = request.files.get("image")

    complaint_id = str(uuid.uuid4())  # Generate a unique complaint ID
    image_filename = None

    if file:
        image_filename = f"{complaint_id}.jpg"
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], image_filename))

    complaint_data = {
        "id": complaint_id,
        "name": name,
        "phone": phone,
        "aadhar": aadhar,
        "address": address,
        "district": district,
        "pincode": pincode,
        "complaint": complaint,
        "location": location,
        "ministry": ministry,
        "image": image_filename,
        "status": "Pending"
    }

    complaints_collection.insert_one(complaint_data)

    # Send email notification
    recipient_email = "ajaisha2021@gmail.com"
    subject = "New Complaint Submission"
    message_body = f"""
    A new complaint has been submitted:
    
    Complaint ID: {complaint_id}
    Name: {name}
    Phone: {phone}
    Aadhaar: {aadhar}
    Address: {address}
    District: {district}
    Pincode: {pincode}
    Complaint: {complaint}
    Location: {location}
    Ministry: {ministry}
    Status: Pending
    """

    send_email(recipient_email, subject, message_body)

    return jsonify({"message": "Complaint submitted successfully", "id": complaint_id}), 201


def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully")

    except Exception as e:
        print(f"Failed to send email: {e}")
# Route to get all complaints
@app.route("/get_complaints", methods=["GET"])
def get_complaints():
    complaints = list(complaints_collection.find({}, {"_id": 0}))  # Exclude MongoDB _id
    return jsonify(complaints)

# Route to update complaint status to "Processing"
@app.route("/process_complaint/<complaint_id>", methods=["PUT"])
def process_complaint(complaint_id):
    complaints_collection.update_one({"id": complaint_id}, {"$set": {"status": "Processing"}})
    return jsonify({"message": "Complaint moved to Processing"}), 200

# Route to upload image and mark complaint as Completed
@app.route("/upload_image", methods=["POST"])
def upload_image():
    complaint_id = request.form["complaint_id"]
    file = request.files["image"]

    if file:
        filename = f"{complaint_id}.jpg"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Update database with image and set status to "Completed"
        complaints_collection.update_one(
            {"id": complaint_id}, 
            {"$set": {"status": "Completed", "image": filename}}
        )

        return jsonify({"message": "Image uploaded and status updated"}), 200
    return jsonify({"error": "No image provided"}), 400

# Route to serve uploaded images
@app.route("/uploads/<filename>")
def get_uploaded_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)



    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5002)
