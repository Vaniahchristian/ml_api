from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from PIL import Image
import numpy as np
import io
import time
import os
from flask_cors import CORS
import google.generativeai as genai
import tempfile
import sys
from dotenv import load_dotenv
load_dotenv()  # This loads variables from the .env file into the environment

# Gemini API configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

# Class mappings
eczema_class_names = {  
    0: 'Acne and Rosacea',
    1: 'Normal',
    2: 'Vitiligo',
    3: 'Fungal Infections',
    4: 'Melanoma',
    5: 'Eczema'
}

body_part_class_names = {
    0: 'Belly', 1: 'Ear', 2: 'Elbow', 3: 'Eye', 4: 'Foot',
    5: 'Hand', 6: 'Knee', 7: 'Neck', 8: 'Nose', 9: 'Shoulders'
}

# Load models with timing
start_load = time.time()
print("Loading models...")
try:
    vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
    for layer in vgg_model.layers:
        layer.trainable = False

    eczema_model = load_model('eczema.h5')

    # Load TensorFlow Lite model for body part classification
    interpreter = tf.lite.Interpreter(model_path="mobilenet_bodypart_model_quantized.tflite")
    interpreter.allocate_tensors()

    end_load = time.time()
    print(f"Models loaded successfully in {end_load - start_load:.2f} seconds")
except Exception as e:
    print("Error loading models:", str(e))
    raise

def preprocess_image_for_vgg(image_bytes):
    start_preprocess_vgg = time.time()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((180, 180))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        end_preprocess_vgg = time.time()
        print(f"Preprocessing for VGG19 took {end_preprocess_vgg - start_preprocess_vgg:.2f} seconds")
        return img_array
    except Exception as e:
        print("Error preprocessing image for VGG:", str(e))
        raise

def preprocess_image_for_bodypart(image_bytes):
    start_preprocess_bodypart = time.time()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        end_preprocess_bodypart = time.time()
        print(f"Preprocessing for body part model took {end_preprocess_bodypart - start_preprocess_bodypart:.2f} seconds")
        return img_array
    except Exception as e:
        print("Error preprocessing image for body part:", str(e))
        raise

def get_severity(confidence):
    start_severity = time.time()
    if confidence >= 0.8:
        severity = "Severe"
    elif confidence >= 0.5:
        severity = "Moderate"
    else:
        severity = "Mild"
    end_severity = time.time()
    print(f"Severity calculation took {end_severity - start_severity:.2f} seconds")
    return severity

def predict_with_tflite(model_interpreter, img_array):
    start_tflite = time.time()
    input_details = model_interpreter.get_input_details()
    output_details = model_interpreter.get_output_details()
    model_interpreter.set_tensor(input_details[0]['index'], img_array)
    model_interpreter.invoke()
    output_data = model_interpreter.get_tensor(output_details[0]['index'])
    end_tflite = time.time()
    print(f"TFLite prediction took {end_tflite - start_tflite:.2f} seconds")
    return output_data

def get_treatment_recommendations(severity, body_part):
    start_recommendations = time.time()
    
    # General tips for all eczema cases
    general_tips = [
        "Moisturize 2-3 times daily with fragrance-free creams (e.g., CeraVe, Vanicream)",
        "Take short, lukewarm showers (5-10 minutes) and pat skin dry",
        "Use hypoallergenic, fragrance-free soaps and detergents",
        "Avoid triggers (e.g., wool, fragrances, stress) through a trigger journal",
        "Wear loose, breathable cotton clothing"
    ]
    
    # Severity-specific treatments based on evidence-based guidelines
    severe_treatments = [
        "Consult a dermatologist for biologics (e.g., dupilumab/Dupixent) or oral JAK inhibitors (e.g., upadacitinib/Rinvoq)",
        "keep a trigger journal to identify and avoid irritants like harsh soaps or stress.", 
        "Use high-potency prescription corticosteroids (e.g., clobetasol) for short-term flares (1-2 weeks)",
        "Apply wet wrap therapy over moisturizers or topicals to enhance efficacy",
        "Consider phototherapy (narrowband UVB) under medical supervision",
        "Monitor for infections (redness, oozing) and seek antibiotics if needed"
    ]
    
    moderate_treatments = [
        "Use prescription non-steroidal topicals (e.g., tacrolimus/Protopic, crisaborole/Eucrisa)",
        "Apply medium-potency corticosteroids (e.g., triamcinolone) as prescribed",
        "Take sedating antihistamines (e.g., diphenhydramine/Benadryl) for nighttime itching",
        "Use bleach baths (1/4 cup bleach per 40-gallon tub) to reduce bacterial infections",
        "Apply cold compresses to soothe inflammation"
    ]
    
    mild_treatments = [
        "Apply OTC hydrocortisone 1% cream for mild flares (up to 2 weeks)",
        "Use ceramide-based moisturizers (e.g., CeraVe, Aveeno Eczema Therapy) after bathing",
        "Take colloidal oatmeal baths (e.g., Aveeno Soothing Bath Treatment) for itch relief",
        "Apply coconut oil or calamine lotion for mild dryness and itching",
        "Identify and avoid irritants (e.g., harsh soaps, dust mites)"
    ]
    
    # Body-part-specific recommendations, tailored for eczema treatment
    body_part_specific = {
        'Eye': [
            "Use non-steroidal topicals (e.g., tacrolimus/Protopic) to avoid steroid-related risks",
            "Avoid eye makeup and rubbing eyes to prevent irritation"
        ],
        'Nose': [
            "Use gentle, non-comedogenic moisturizers to avoid clogging pores",
            "Avoid nasal sprays with irritants"
        ],
        'Neck': [
            "Apply non-steroidal topicals for sensitive skin (e.g., pimecrolimus/Elidel)",
            "Avoid tight collars or jewelry that may irritate"
        ],
        'Hand': [
            "Apply thick ointments (e.g., Vaseline) after handwashing",
            "Wear cotton gloves at night to lock in moisturizer; use vinyl gloves for cleaning"
        ],
        'Elbow': [
            "Apply extra moisturizer to thick skin areas; consider occlusion with bandages",
            "Use medium-potency steroids for flares if prescribed"
        ],
        'Knee': [
            "Apply thick creams to prevent cracking; avoid kneeling on rough surfaces",
            "Wear loose pants to reduce friction"
        ],
        'Foot': [
            "Use antifungal creams if fungal infection is suspected; keep feet dry",
            "Wear breathable cotton socks and change frequently"
        ],
        'Belly': [
            "Apply ceramide-based creams to large areas; avoid tight clothing",
            "Keep skin dry to prevent bacterial growth"
        ],
        'Ear': [
            "Use non-steroidal topicals for sensitive ear skin; keep ears dry",
            "Avoid earbuds or piercings during flares"
        ],
        'Shoulders': [
            "Apply moisturizers after showers; avoid heavy backpacks with straps",
            "Use loose, cotton shirts to minimize irritation"
        ]
    }
    
    # Select treatments based on severity
    if severity == "Severe":
        recommendations = severe_treatments
    elif severity == "Moderate":
        recommendations = moderate_treatments
    else:
        recommendations = mild_treatments
    
    # Add body-part-specific recommendations if applicable (robust lookup)
    body_part_key = body_part.capitalize()
    if body_part_key in body_part_specific:
        recommendations.extend(body_part_specific[body_part_key])
    elif body_part_key.rstrip('s') in body_part_specific:
        recommendations.extend(body_part_specific[body_part_key.rstrip('s')])
    elif body_part_key.lower() in body_part_specific:
        recommendations.extend(body_part_specific[body_part_key.lower()])
    elif body_part_key.lower().rstrip('s') in body_part_specific:
        recommendations.extend(body_part_specific[body_part_key.lower().rstrip('s')])
    
    # Always include general tips
    recommendations.extend(general_tips)
    
    end_recommendations = time.time()
    print(f"Treatment recommendations took {end_recommendations - start_recommendations:.2f} seconds")
    return recommendations

def get_skincare_tips():
    start_skincare = time.time()
    tips = [
        "Maintain a consistent skincare routine",
        "Use gentle, fragrance-free cleansers",
        "Apply sunscreen daily",
        "Stay hydrated and maintain a balanced diet",
        "Get adequate sleep and manage stress",
        "Avoid hot showers and pat dry skin gently",
        "Use a humidifier in dry environments",
        "Consider using products with ceramides and hyaluronic acid"
    ]
    end_skincare = time.time()
    print(f"Skincare tips generation took {end_skincare - start_skincare:.2f} seconds")
    return tips

@app.route('/predict', methods=['POST'])
def predict():
    start_request = time.time()
    print("Received prediction request")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    print(f"Request files: {list(request.files.keys())}")

    if 'image' not in request.files:
        print("No image in request")
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']
        print(f"Image received: {image_file.filename}")
        image_bytes = image_file.read()
        print(f"Image size: {len(image_bytes)} bytes")

        # Save uploaded image to a temporary file for Gemini
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img:
            tmp_img.write(image_bytes)
            tmp_img_path = tmp_img.name

        # --- Gemini skin check ---
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        img_pil = Image.open(tmp_img_path)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        prompt_skin = "Does this image predominantly feature human skin? Respond with a single word: Yes or No."
        print(f"Sending image to Gemini for skin analysis...")
        skin_response = gemini_model.generate_content([prompt_skin, img_pil])
        skin_classification = skin_response.text.strip().lower()
        print(f"Gemini skin response: '{skin_classification}'")
        if skin_classification != 'yes':
            os.unlink(tmp_img_path)
            return jsonify({'error': 'The uploaded image does not predominantly feature human skin. Please upload a suitable image.'}), 400

        # --- Gemini body part classification (anatomical, single word) ---
        prompt_bodypart = (
            "What is the main human body part shown in this image? "
            "Respond with a single word, the anatomical name only (e.g., 'knee', 'ear', 'hand', 'foot'), "
            "and do not include any adjectives or descriptors."
        )
        print(f"Sending image to Gemini for body part classification...")
        bodypart_response = gemini_model.generate_content([prompt_bodypart, img_pil])
        body_label = bodypart_response.text.strip()
        print(f"Gemini body part response: '{body_label}'")
        os.unlink(tmp_img_path)
        body_confidence = None  # Gemini does not provide a confidence score

        # Continue with VGG/Eczema model as before
        print("Preprocessing for VGG19")
        img_array_vgg = preprocess_image_for_vgg(image_bytes)
        start_vgg = time.time()
        print("Running VGG prediction")
        vgg_features = vgg_model.predict(img_array_vgg, verbose=0)
        end_vgg = time.time()
        print(f"VGG prediction took {end_vgg - start_vgg:.2f} seconds")

        start_eczema = time.time()
        print("Running eczema model prediction")
        features_flat = vgg_features.reshape(1, -1)
        eczema_preds = eczema_model.predict(features_flat, verbose=0)
        eczema_class = int(np.argmax(eczema_preds[0]))
        eczema_label_raw = eczema_class_names[eczema_class]
        eczema_label = 'Eczema' if eczema_label_raw == 'Eczema' else 'No Eczema'
        eczema_confidence = float(eczema_preds[0][eczema_class])
        end_eczema = time.time()
        print(f"Eczema model prediction took {end_eczema - start_eczema:.2f} seconds")

        severity = get_severity(eczema_confidence) if eczema_label == 'Eczema' else None
        recommendations = get_treatment_recommendations(severity, body_label) if eczema_label == 'Eczema' else []
        skincare_tips = get_skincare_tips() if eczema_label != 'Eczema' else []

        # Disclaimer for all recommendations
        disclaimer = "Disclaimer: These AI-generated recommendations are for informational purposes only. Consult a dermatologist to confirm diagnosis and treatment, especially for severe or persistent symptoms."
        if eczema_label == 'Eczema':
            recommendations.append(disclaimer)

        response = {
            'eczemaPrediction': eczema_label,
            'eczemaConfidence': eczema_confidence,
            'eczemaSeverity': severity,
            'bodyPart': body_label,
            'bodyPartConfidence': body_confidence,
            'recommendations': recommendations,
            'skincareTips': skincare_tips
        }

        end_request = time.time()
        print(f"Total request processing took {end_request - start_request:.2f} seconds")
        print("Prediction successful:", response)
        return jsonify(response)

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)