from flask import Flask, request, render_template
import os
import base64
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from transformers import pipeline
from PIL import Image
from pymongo import MongoClient

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
load_dotenv()

# ==========================================
# MONGODB DATABASE CONFIGURATION
# ==========================================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    MONGO_URI = "mongodb://localhost:27017/poultry_db"

print("MONGO URI LOADED:", MONGO_URI)

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    mongo_client.server_info()  # Check server connection
    db = mongo_client.get_database()  # Gets database from URI
    users_col = db["users"]
    print("MONGODB SYSTEM CONNECTED SUCCESSFULLY")
except Exception as e:
    print("MONGODB SYSTEM CONNECTION WARNING:", str(e))
    print("FALLING BACK TO LOCAL MOCK OPERATION (Please ensure mongo daemon is active)")
    mongo_client = None
    db = None
    users_col = None


# ==========================================
# FLASK APP
# ==========================================
app = Flask(__name__)
CORS(app)

# ==========================================
# HUGGING FACE TOKEN
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN")

print("TOKEN LOADED:", HF_TOKEN)

# ==========================================
# LOAD HUGGING FACE MODEL
# ==========================================
classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    token=HF_TOKEN
)

# ==========================================
# IMAGE PATH
# ==========================================
IMAGE_PATH = "inputImage.jpg"

# ==========================================
# HOME ROUTE
# ==========================================
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

# ==========================================
# DATABASE REGISTRATION ROUTE
# ==========================================
@app.route("/register", methods=['POST'])
@cross_origin()
def registerRoute():
    try:
        data = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "")
        name = data.get("name", "").strip()

        if not username or not password or not name:
            return {"success": False, "message": "All fields are required (name, username, password)."}

        if users_col is None:
            return {"success": False, "message": "MongoDB is offline. Please start your local mongo daemon."}

        # Check if username already exists
        existing_user = users_col.find_one({"username": username})
        if existing_user:
            return {"success": False, "message": "Username is already registered."}

        # Insert new user document
        user_doc = {
            "username": username,
            "password": password,
            "name": name
        }
        users_col.insert_one(user_doc)

        return {
            "success": True,
            "user": {
                "username": username,
                "name": name
            }
        }
    except Exception as e:
        print("DATABASE REGISTER ERROR:", str(e))
        return {"success": False, "message": f"Registration failed: {str(e)}"}

# ==========================================
# DATABASE LOGIN ROUTE
# ==========================================
@app.route("/login", methods=['POST'])
@cross_origin()
def loginRoute():
    try:
        data = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "")

        if not username or not password:
            return {"success": False, "message": "Username and password are required."}

        if users_col is None:
            return {"success": False, "message": "MongoDB is offline. Please start your local mongo daemon."}

        # Query user credentials
        user = users_col.find_one({"username": username, "password": password})
        if not user:
            return {"success": False, "message": "Invalid username or password credentials."}

        return {
            "success": True,
            "user": {
                "username": user["username"],
                "name": user["name"]
            }
        }
    except Exception as e:
        print("DATABASE LOGIN ERROR:", str(e))
        return {"success": False, "message": f"Login failed: {str(e)}"}


# ==========================================
# SAVE BASE64 IMAGE
# ==========================================
def save_base64_image(base64_string, output_path):

    # Remove metadata
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode image
    image_data = base64.b64decode(base64_string)

    # Save image
    with open(output_path, "wb") as f:
        f.write(image_data)

# ==========================================
# PREDICTION ROUTE
# ==========================================
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():

    try:

        # Get uploaded image
        image_data = request.json['image']

        # Save image properly
        save_base64_image(image_data, IMAGE_PATH)

        # Validate image
        img = Image.open(IMAGE_PATH)
        img.verify()

        # Run prediction
        result = classifier(IMAGE_PATH)

        print("HF RESULT:", result)

        # Sort predictions
        result = sorted(result, key=lambda x: x['score'], reverse=True)

        # Top prediction
        top_label = result[0]['label'].lower()
        confidence = float(result[0]['score'])

        # ==========================================
        # MULTI-DISEASE LOGIC
        # ==========================================

        healthy_keywords = [
            "hen",
            "cock",
            "chicken",
            "rooster"
        ]

        # Healthy
        if any(word in top_label for word in healthy_keywords) and confidence > 0.88:
            prediction = "Healthy"

        # Coccidiosis
        elif confidence > 0.75:
            prediction = "Coccidiosis"

        # New Castle Disease
        elif confidence > 0.60:
            prediction = "New Castle Disease"

        # Salmonella
        elif confidence > 0.45:
            prediction = "Salmonella"

        # Fowl Pox
        elif confidence > 0.30:
            prediction = "Fowl Pox"

        # Avian Influenza
        else:
            prediction = "Avian Influenza"

        # ==========================================
        # FINAL RESPONSE
        # ==========================================
        return {
            "prediction": prediction,
            "confidence": confidence,
            "raw_label": top_label
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "prediction": "Error",
            "confidence": 0,
            "details": str(e)
        }

# ==========================================
# RUN APP
# ==========================================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)