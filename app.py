import os

# Keep CPU thread usage predictable on small Render instances.
# These defaults are applied before TensorFlow is imported in load_model().
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sqlite3
import tempfile
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')
app = Flask(__name__, static_folder='statics', static_url_path='/statics', template_folder='templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'smart-weed-development-key-change-before-deployment')
DATABASE = BASE_DIR / 'smart_weed.db'
UPLOAD_FOLDER = BASE_DIR / 'statics' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_PATH = BASE_DIR / 'models' / 'smart_weed_unet.tflite'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
MAX_FILE_SIZE = 8 * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
MAX_IMAGE_PIXELS = 16000000
FREE_PREDICTION_LIMIT = 5
MODEL_SIZE = (256, 256)
CLASS_NAMES = {0: 'Background/Soil', 1: 'Crop', 2: 'Weed'}
WEED_CLASS_INDEX = 2
LOW_WEED_PERCENT = 5.0
MEDIUM_WEED_PERCENT = 20.0

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def normalize_field_name(value):
    """Normalize spacing while preserving the user's chosen capitalization."""
    return ' '.join(str(value or '').strip().split())

def field_key(value):
    """Case-insensitive key used to avoid duplicate field records."""
    return normalize_field_name(value).casefold()

def save_or_get_field(conn, customer_id, field_name):
    """Save a field once per customer and return its canonical display name."""
    display_name = normalize_field_name(field_name)
    normalized = field_key(display_name)
    if not display_name:
        raise ValueError('Field name is required.')
    existing = conn.execute('SELECT field_name FROM fields WHERE customer_id = ? AND normalized_name = ?', (customer_id, normalized)).fetchone()
    if existing:
        return existing['field_name']
    conn.execute('INSERT OR IGNORE INTO fields (customer_id, field_name, normalized_name, created_at)\n           VALUES (?, ?, ?, ?)', (customer_id, display_name, normalized, get_current_time()))
    saved = conn.execute('SELECT field_name FROM fields WHERE customer_id = ? AND normalized_name = ?', (customer_id, normalized)).fetchone()
    return saved['field_name'] if saved else display_name

def get_saved_fields(customer_id):
    if not customer_id:
        return []
    conn = get_db()
    try:
        rows = conn.execute('SELECT field_name FROM fields WHERE customer_id = ? ORDER BY field_name COLLATE NOCASE', (customer_id,)).fetchall()
        return [row['field_name'] for row in rows]
    finally:
        conn.close()

def column_exists(conn, table_name, column_name):
    columns = conn.execute(f'PRAGMA table_info({table_name})').fetchall()
    return any((row['name'] == column_name for row in columns))

def add_column_if_missing(conn, table_name, column_name, column_definition):
    if not column_exists(conn, table_name, column_name):
        conn.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}')

def init_database():
    conn = get_db()
    conn.execute('\n        CREATE TABLE IF NOT EXISTS customers (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            name TEXT NOT NULL,\n            email TEXT UNIQUE NOT NULL,\n            phone TEXT,\n            farm_location TEXT,\n            password_hash TEXT NOT NULL,\n            created_at TEXT NOT NULL,\n            last_login TEXT\n        )\n    ')
    conn.execute('\n        CREATE TABLE IF NOT EXISTS login_history (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            customer_id INTEGER NOT NULL,\n            login_at TEXT NOT NULL,\n            logout_at TEXT,\n            ip_address TEXT,\n            FOREIGN KEY(customer_id) REFERENCES customers(id)\n        )\n    ')
    conn.execute('\n        CREATE TABLE IF NOT EXISTS analyses (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            customer_id INTEGER NOT NULL,\n            field_name TEXT,\n            crop_percentage REAL,\n            soil_percentage REAL,\n            original_filename TEXT,\n            weed_percentage REAL,\n            weed_severity TEXT,\n            weed_location TEXT,\n            weed_pixel_count INTEGER,\n            confidence REAL,\n            weed_bbox TEXT,\n            source_filename TEXT,\n            overlay_filename TEXT,\n            weed_mask_filename TEXT,\n            suppression_status TEXT,\n            suppression_recommendation TEXT,\n            post_treatment_status TEXT,\n            workflow_status TEXT,\n            created_at TEXT NOT NULL,\n            FOREIGN KEY(customer_id) REFERENCES customers(id)\n        )\n    ')
    conn.execute('\n        CREATE TABLE IF NOT EXISTS fields (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            customer_id INTEGER NOT NULL,\n            field_name TEXT NOT NULL,\n            normalized_name TEXT NOT NULL,\n            created_at TEXT NOT NULL,\n            UNIQUE(customer_id, normalized_name),\n            FOREIGN KEY(customer_id) REFERENCES customers(id)\n        )\n    ')
    customer_columns = {'name': 'TEXT', 'email': 'TEXT', 'phone': 'TEXT', 'farm_location': 'TEXT', 'password_hash': 'TEXT', 'created_at': 'TEXT', 'last_login': 'TEXT'}
    for column, definition in customer_columns.items():
        add_column_if_missing(conn, 'customers', column, definition)
    login_columns = {'customer_id': 'INTEGER', 'login_at': 'TEXT', 'logout_at': 'TEXT', 'ip_address': 'TEXT'}
    for column, definition in login_columns.items():
        add_column_if_missing(conn, 'login_history', column, definition)
    analysis_columns = {'customer_id': 'INTEGER', 'field_name': 'TEXT', 'crop_percentage': 'REAL', 'soil_percentage': 'REAL', 'original_filename': 'TEXT', 'weed_percentage': 'REAL', 'weed_severity': 'TEXT', 'weed_location': 'TEXT', 'weed_pixel_count': 'INTEGER', 'confidence': 'REAL', 'weed_bbox': 'TEXT', 'source_filename': 'TEXT', 'overlay_filename': 'TEXT', 'weed_mask_filename': 'TEXT', 'suppression_status': 'TEXT', 'suppression_recommendation': 'TEXT', 'post_treatment_status': 'TEXT', 'workflow_status': 'TEXT', 'created_at': 'TEXT'}
    for column, definition in analysis_columns.items():
        add_column_if_missing(conn, 'analyses', column, definition)
    old_fields = conn.execute("\n        SELECT DISTINCT customer_id, TRIM(field_name) AS field_name\n        FROM analyses\n        WHERE customer_id IS NOT NULL AND field_name IS NOT NULL AND TRIM(field_name) <> ''\n    ").fetchall()
    for row in old_fields:
        display_name = normalize_field_name(row['field_name'])
        if display_name:
            conn.execute('\n                INSERT OR IGNORE INTO fields (customer_id, field_name, normalized_name, created_at)\n                VALUES (?, ?, ?, ?)\n            ', (row['customer_id'], display_name, field_key(display_name), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()
init_database()
SUPPORTED_LANGUAGES = {'en': 'English', 'te': 'Telugu', 'ta': 'Tamil', 'kn': 'Kannada', 'ml': 'Malayalam', 'hi': 'Hindi'}

@app.before_request
def load_preferred_language():
    requested_language = request.args.get('lang', '').strip().lower()
    if requested_language in SUPPORTED_LANGUAGES:
        session['language'] = requested_language

TRANSLATIONS = {
    'en': {
        'app_name': 'Smart Weed Detection System',
        'tagline': 'AI-Powered Smart Agriculture',
        'welcome': 'Welcome',
        'home': 'Home',
        'dashboard': 'Dashboard',
        'history': 'History',
        'profile': 'Profile',
        'logout': 'Logout',
        'login': 'Login',
        'register': 'Register',
        'language': 'Language',
        'select_language': 'Select Language',
        'dark_mode': 'Dark Mode',
        'light_mode': 'Light Mode',
        'assistant': 'AI Assistant',
        'assistant_title': 'Smart Agriculture Assistant',
        'assistant_welcome': 'Hello! How can I help you with weed detection?',
        'assistant_greeting': 'Hello! Ask a question about weeds, field images, or crop-safe weed management.',
        'assistant_ready': 'Ready',
        'ask_question': 'Ask something...',
        'ask_question_long': 'Ask a question...',
        'speak': 'Speak',
        'send': 'Send',
        'close': 'Close assistant',
        'open_assistant': 'Open AI assistant',
        'assistant_language': 'Assistant language',
        'auto_detect': 'Auto detect',
        'upload_image': 'Upload Field Image',
        'upload_instructions': 'Upload a clear image of your agricultural field to identify crops and weeds.',
        'supported_formats': 'Supported formats: JPG, JPEG, PNG, WEBP',
        'analyze': 'Analyze Field Image',
        'analyzing': 'Analyzing Image...',
        'prediction_result': 'Field Analysis Results',
        'weed_coverage': 'Weed Coverage',
        'crop_coverage': 'Crop Coverage',
        'soil_background': 'Soil / Background',
        'weed_severity': 'Weed Severity',
        'confidence': 'Confidence',
        'weed_pixels': 'Weed Pixels',
        'source_image': 'Original Image',
        'overlay_image': 'Segmentation Overlay',
        'weed_mask': 'Weed Mask',
        'recommendation': 'Recommendation',
        'recent_analyses': 'Recent Analyses',
        'date': 'Date',
        'full_name': 'Full Name',
        'email': 'Email',
        'phone': 'Phone',
        'password': 'Password',
        'confirm_password': 'Confirm Password',
        'create_account': 'Create Account',
        'total_analyses': 'Total Analyses',
        'average_weed': 'Average Weed Coverage',
        'high': 'High',
        'medium': 'Medium',
        'low': 'Low',
        'no_analyses': 'No analyses found',
        'farm_location': 'Farm Location',
        'save_changes': 'Save Changes',
        'update_profile': 'Update Profile',
        'guest_limit': 'Free Predictions',
        'remaining': 'Remaining',
        'used': 'Used',
        'guest_limit_used': 'Your five free predictions have been used.',
        'login_to_continue': 'Please log in or register to continue using the system.',
        'select_field': 'Select Agricultural Field',
        'existing_field': 'Select an existing field',
        'new_field_prompt': 'Or enter a new field name below.',
        'new_field_name': 'Enter new field name',
        'field_name': 'Agricultural Field Name',
        'field_name_placeholder': 'Enter field name (e.g. North Field)',
        'field_name_help': 'Your field name helps organize and compare your saved analyses.',
        'select_image': 'Select an Image',
        'browse_image': 'Click here to browse and select your field image',
        'image_preview': 'Selected image preview',
        'saved_history_note': 'Your analyses are saved to your account history.',
        'guest_history_note': 'Create an account to access your saved analysis history.',
        'register_here': 'Register here',
        'original_image': 'Original Image',
        'segmentation_overlay': 'Segmentation Overlay',
        'weed_mask_image': 'Weed Mask',
        'image_unavailable': 'Image is not available.',
        'image_failed': 'Unable to load image.',
        'detection_details': 'Detection Details',
        'field_name_label': 'Field Name',
        'weed_location': 'Weed Location',
        'weed_pixel_count': 'Weed Pixel Count',
        'suppression_status': 'Suppression Status',
        'post_treatment_status': 'Post-treatment Status',
        'conceptual_stage_status': 'Conceptual stage: detected weed regions can be targeted for thermal suppression. Dead weed biomass can remain in place for natural decomposition.',
        'no_treatment_required': 'No treatment required',
        'class_distribution': 'Class Distribution',
        'weed': 'Weed',
        'crop': 'Crop',
        'background_soil': 'Background / Soil',
        'total': 'Total',
        'natural_management': 'Natural Weed Management Recommendations',
        'low_coverage': 'Low Weed Coverage',
        'moderate_coverage': 'Moderate Weed Coverage',
        'high_coverage': 'High Weed Coverage',
        'low_coverage_text': 'Only a small amount of weed coverage was detected. Monitor the affected area and use targeted control methods where needed.',
        'moderate_coverage_text': 'A moderate amount of weed coverage was detected. Plan careful targeted management while protecting nearby crops.',
        'high_coverage_text': 'A high amount of weed coverage was detected. Inspect the field and plan weed management carefully to avoid crop damage.',
        'low_priority': 'Priority: Low',
        'medium_priority': 'Priority: Medium',
        'high_priority': 'Priority: High',
        'treatment_note': 'Any treatment should be crop-safe and appropriate for the weed type and field conditions.',
        'analyze_another': 'Analyze Another Image',
        'view_history': 'View Analysis History',
        'register_save_history': 'Register to Save History',
        'invalid_format': 'Please select a JPG, JPEG, PNG or WEBP image.',
        'file_too_large': 'Image size must not exceed 8 MB.',
        'image_dimensions_large': 'Image dimensions are too large. Please select a smaller image.',
        'invalid_image': 'The selected file is not a valid image.',
        'read_image_failed': 'Unable to read the selected image.',
        'field_required': 'Please select an existing field or enter an agricultural field name.',
        'field_too_long': 'Field name cannot exceed 80 characters.',
        'select_image_first': 'Please select a field image first.',
        'invalid_image_format': 'Invalid image format. Select JPG, PNG or WEBP.',
        'total_analyses_label': 'Total Analyses',
        'average_weed_label': 'Average Weed Coverage',
        'high_priority': 'High Priority',
        'medium_priority': 'Medium Priority',
        'low_priority': 'Low Priority',
        'field_comparison': 'Field-wise Comparison',
        'field_comparison_description': 'Compare weed and good plant coverage across your agricultural fields.',
        'field_analyses_count': 'analyses',
        'chart_aria': 'Pie chart showing weed and crop coverage',
        'chart_note': 'The pie chart shows the relative weed and crop coverage and totals 100%.',
        'good_plants': 'Good Plants (Crop)',
        'no_chart_data': 'No weed or crop pixels recorded.',
        'no_field_comparison': 'No field comparison data available yet. Upload and analyse field images to generate graphs.',
        'individual_field_analysis': 'Individual Field Analysis',
        'individual_field_description': 'Average coverage statistics for each saved field. No prediction images are displayed on this dashboard.',
        'unassigned_field': 'Unassigned Field',
        'average_good_plants': 'Average Good Plants',
        'no_field_analyses': 'No field analyses have been saved yet. Analyse a field image to view field-wise coverage statistics here.',
        'dashboard_description': 'Overview of your saved field analyses. Compare weed and good plant coverage across your fields using individual field pie charts.',
        'footer_text': 'Smart Weed Detection System | U-Net',
        'invalid_image_format_message': 'Please select a JPG, JPEG, PNG or WEBP image.',
        'image_size_error': 'Image size must not exceed 8 MB.',
        'image_dimensions_error': 'Image dimensions are too large. Please select a smaller image.',
        'invalid_selected_image': 'The selected file is not a valid image.',
        'image_read_error': 'Unable to read the selected image.',
        'field_selection_error': 'Please select an existing field or enter an agricultural field name.',
        'field_length_error': 'Field name cannot exceed 80 characters.',
        'no_image_selected': 'Please select a field image first.',
        'pie_no_data': 'No weed or crop pixels recorded.',
        'recommendations': 'Weed Control Recommendations',
        'recommendations_description': 'Recommended methods based on the estimated weed coverage.',
        'low_weed_priority': 'LOW WEED PRIORITY',
        'low_method_1_title': 'Manual Weed Removal',
        'low_method_1': 'Remove small weeds by hand when the soil is slightly moist. Pull them gently from the base and remove their roots wherever possible to reduce regrowth.',
        'low_method_2_title': 'Mulching',
        'low_method_2': 'Cover the exposed soil around crops with dry leaves, straw or other suitable organic materials. This blocks sunlight from reaching weed seeds and helps prevent new weed growth.',
        'low_method_3_title': 'Regular Field Inspection',
        'low_method_3': 'Inspect the field regularly, especially after rainfall or irrigation. Identify newly growing weeds and remove them early before they spread or produce seeds.',
        'medium_weed_priority': 'MEDIUM WEED PRIORITY',
        'medium_method_1_title': 'Shallow Hoeing and Mechanical Removal',
        'medium_method_1': 'Use a hand hoe or suitable mechanical weeder to remove weeds between crop rows. Work close to the soil surface to avoid damaging crop roots, and collect the removed weeds.',
        'medium_method_2_title': 'Section-Wise Weed Control',
        'medium_method_2': 'Divide the affected field into smaller sections and control weeds one section at a time. Remove weeds near crops manually and clear open spaces using suitable tools to make the work manageable.',
        'medium_method_3_title': 'Mulching and Cover Crops',
        'medium_method_3': 'Apply organic mulch after removing existing weeds to reduce further growth. During suitable crop cycles, grow cover crops to occupy exposed soil and compete with weeds for space and resources.',
        'high_weed_priority': 'HIGH WEED PRIORITY',
        'high_method_1_title': 'Intensive Mechanical Weed Control',
        'high_method_1': 'Use suitable mechanical weeders or repeated shallow cultivation to control dense weed growth between crop rows. Work carefully around crop plants and remove remaining weeds before they produce seeds.',
        'high_method_2_title': 'Safe and Targeted Herbicide Use',
        'high_method_2': 'Where mechanical and manual methods are insufficient, consider a locally approved herbicide suitable for the crop and target weeds. Follow the product label for dosage, application timing and safety precautions to avoid crop damage.',
        'high_method_3_title': 'Integrated Weed Management',
        'high_method_3': 'Combine manual removal, mechanical control and mulching to manage heavy weed growth. Remove large weeds, cover exposed soil and regularly inspect treated areas to control new growth and prevent weeds from spreading.',
        'important_note': 'Important Note',
        'weed_safety_note': 'These recommendations are based on the estimated weed coverage in the uploaded image. Select suitable methods according to the crop and field conditions, and always take precautions to protect crops.',
    },

    'te': {
        'app_name': 'స్మార్ట్ కలుపు మొక్కల గుర్తింపు వ్యవస్థ',
        'tagline': 'కృత్రిమ మేధస్సుతో స్మార్ట్ వ్యవసాయం',
        'welcome': 'స్వాగతం',
        'home': 'హోమ్',
        'dashboard': 'డాష్‌బోర్డ్',
        'history': 'చరిత్ర',
        'profile': 'ప్రొఫైల్',
        'logout': 'లాగ్అవుట్',
        'login': 'లాగిన్',
        'register': 'నమోదు',
        'language': 'భాష',
        'select_language': 'భాషను ఎంచుకోండి',
        'dark_mode': 'డార్క్ మోడ్',
        'light_mode': 'లైట్ మోడ్',
        'assistant': 'AI సహాయకుడు',
        'assistant_title': 'స్మార్ట్ వ్యవసాయ సహాయకుడు',
        'assistant_welcome': 'హలో! కలుపు మొక్కల గుర్తింపులో నేను మీకు ఎలా సహాయపడగలను?',
        'assistant_greeting': 'హలో! కలుపు మొక్కలు, పొలం చిత్రాలు లేదా పంటలకు సురక్షితమైన కలుపు నియంత్రణ గురించి ప్రశ్న అడగండి.',
        'assistant_ready': 'సిద్ధంగా ఉంది',
        'ask_question': 'ఏదైనా అడగండి...',
        'ask_question_long': 'ప్రశ్న అడగండి...',
        'speak': 'మాట్లాడండి',
        'send': 'పంపండి',
        'close': 'సహాయకుడిని మూసివేయండి',
        'open_assistant': 'AI సహాయకుడిని తెరవండి',
        'assistant_language': 'సహాయకుడి భాష',
        'auto_detect': 'స్వయంచాలకంగా గుర్తించు',
        'upload_image': 'పొలం చిత్రాన్ని అప్‌లోడ్ చేయండి',
        'upload_instructions': 'పంటలు మరియు కలుపు మొక్కలను గుర్తించడానికి మీ వ్యవసాయ క్షేత్రం యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి.',
        'supported_formats': 'అనుమతించబడే ఫార్మాట్‌లు: JPG, JPEG, PNG, WEBP',
        'analyze': 'పొలం చిత్రాన్ని విశ్లేషించండి',
        'analyzing': 'చిత్రాన్ని విశ్లేషిస్తోంది...',
        'prediction_result': 'పొలం విశ్లేషణ ఫలితాలు',
        'weed_coverage': 'కలుపు మొక్కల విస్తీర్ణం',
        'crop_coverage': 'పంటల విస్తీర్ణం',
        'soil_background': 'నేల / నేపథ్యం',
        'weed_severity': 'కలుపు తీవ్రత',
        'confidence': 'నమ్మక స్థాయి',
        'weed_pixels': 'కలుపు పిక్సెల్‌లు',
        'source_image': 'అసలు చిత్రం',
        'overlay_image': 'విభజన చిత్రం',
        'weed_mask': 'కలుపు మాస్క్',
        'recommendation': 'సిఫార్సు',
        'recent_analyses': 'ఇటీవలి విశ్లేషణలు',
        'date': 'తేదీ',
        'full_name': 'పూర్తి పేరు',
        'email': 'ఇమెయిల్',
        'phone': 'ఫోన్',
        'password': 'పాస్‌వర్డ్',
        'confirm_password': 'పాస్‌వర్డ్ నిర్ధారించండి',
        'create_account': 'ఖాతా సృష్టించండి',
        'total_analyses': 'మొత్తం విశ్లేషణలు',
        'average_weed': 'సగటు కలుపు విస్తీర్ణం',
        'high': 'అధికం',
        'medium': 'మధ్యస్థం',
        'low': 'తక్కువ',
        'no_analyses': 'విశ్లేషణలు కనుగొనబడలేదు',
        'farm_location': 'వ్యవసాయ క్షేత్ర స్థానం',
        'save_changes': 'మార్పులను సేవ్ చేయండి',
        'update_profile': 'ప్రొఫైల్‌ను నవీకరించండి',
        'guest_limit': 'ఉచిత అంచనాలు',
        'remaining': 'మిగిలినవి',
        'used': 'ఉపయోగించినవి',
        'guest_limit_used': 'మీ ఐదు ఉచిత అంచనాలు పూర్తయ్యాయి.',
        'login_to_continue': 'వ్యవస్థను కొనసాగించడానికి దయచేసి లాగిన్ అవ్వండి లేదా నమోదు చేసుకోండి.',
        'select_field': 'వ్యవసాయ క్షేత్రాన్ని ఎంచుకోండి',
        'existing_field': 'ఇప్పటికే ఉన్న క్షేత్రాన్ని ఎంచుకోండి',
        'new_field_prompt': 'లేదా క్రింద కొత్త క్షేత్రం పేరును నమోదు చేయండి.',
        'new_field_name': 'కొత్త క్షేత్రం పేరును నమోదు చేయండి',
        'field_name': 'వ్యవసాయ క్షేత్రం పేరు',
        'field_name_placeholder': 'క్షేత్రం పేరు నమోదు చేయండి (ఉదా: ఉత్తర క్షేత్రం)',
        'field_name_help': 'మీ సేవ్ చేసిన విశ్లేషణలను క్రమబద్ధీకరించడానికి క్షేత్రం పేరు సహాయపడుతుంది.',
        'select_image': 'చిత్రాన్ని ఎంచుకోండి',
        'browse_image': 'మీ పొలం చిత్రాన్ని ఎంచుకోవడానికి ఇక్కడ క్లిక్ చేయండి',
        'image_preview': 'ఎంచుకున్న చిత్రం ప్రివ్యూ',
        'saved_history_note': 'మీ విశ్లేషణలు మీ ఖాతా చరిత్రలో సేవ్ అవుతాయి.',
        'guest_history_note': 'మీ విశ్లేషణ చరిత్రను పొందడానికి ఖాతాను సృష్టించండి.',
        'register_here': 'ఇక్కడ నమోదు చేసుకోండి',
        'original_image': 'అసలు చిత్రం',
        'segmentation_overlay': 'విభజన చిత్రం',
        'weed_mask_image': 'కలుపు మాస్క్',
        'image_unavailable': 'చిత్రం అందుబాటులో లేదు.',
        'image_failed': 'చిత్రాన్ని లోడ్ చేయడం సాధ్యపడలేదు.',
        'detection_details': 'గుర్తింపు వివరాలు',
        'field_name_label': 'క్షేత్రం పేరు',
        'weed_location': 'కలుపు స్థానం',
        'weed_pixel_count': 'కలుపు పిక్సెల్‌ల సంఖ్య',
        'suppression_status': 'నియంత్రణ స్థితి',
        'post_treatment_status': 'చికిత్స తర్వాత స్థితి',
        'conceptual_stage_status': 'ఇది భావనాత్మక దశ: గుర్తించిన కలుపు ప్రాంతాలపై లక్ష్యిత ఉష్ణ నియంత్రణ చేయవచ్చు. చనిపోయిన కలుపు మొక్కల అవశేషాలను సహజంగా కుళ్లేందుకు అదే ప్రదేశంలో ఉంచవచ్చు.',
        'no_treatment_required': 'చికిత్స అవసరం లేదు',
        'class_distribution': 'తరగతుల పంపిణీ',
        'weed': 'కలుపు',
        'crop': 'పంట',
        'background_soil': 'నేపథ్యం / నేల',
        'total': 'మొత్తం',
        'natural_management': 'సహజ కలుపు నిర్వహణ సిఫార్సులు',
        'low_coverage': 'తక్కువ కలుపు విస్తీర్ణం',
        'moderate_coverage': 'మధ్యస్థ కలుపు విస్తీర్ణం',
        'high_coverage': 'అధిక కలుపు విస్తీర్ణం',
        'low_coverage_text': 'తక్కువ మొత్తంలో కలుపు కనిపించింది. ప్రభావిత ప్రాంతాన్ని గమనించి అవసరమైన చోట లక్ష్యిత నియంత్రణ పద్ధతులను ఉపయోగించండి.',
        'moderate_coverage_text': 'మధ్యస్థ మొత్తంలో కలుపు కనిపించింది. సమీపంలోని పంటలకు హాని కలగకుండా జాగ్రత్తగా నియంత్రణ ప్రణాళిక రూపొందించండి.',
        'high_coverage_text': 'అధిక మొత్తంలో కలుపు కనిపించింది. పంటలకు నష్టం కలగకుండా పొలాన్ని పరిశీలించి కలుపు నిర్వహణను జాగ్రత్తగా ప్రణాళిక చేయండి.',
        'low_priority': 'ప్రాధాన్యత: తక్కువ',
        'medium_priority': 'ప్రాధాన్యత: మధ్యస్థం',
        'high_priority': 'ప్రాధాన్యత: అధికం',
        'treatment_note': 'ఏ చికిత్స చేసినా అది పంటలకు సురక్షితంగా ఉండాలి మరియు కలుపు రకం, పొలం పరిస్థితులకు అనుగుణంగా ఉండాలి.',
        'analyze_another': 'మరో చిత్రాన్ని విశ్లేషించండి',
        'view_history': 'విశ్లేషణ చరిత్రను చూడండి',
        'register_save_history': 'చరిత్రను సేవ్ చేయడానికి నమోదు చేసుకోండి',
        'invalid_format': 'దయచేసి JPG, JPEG, PNG లేదా WEBP చిత్రాన్ని ఎంచుకోండి.',
        'file_too_large': 'చిత్ర పరిమాణం 8 MB మించకూడదు.',
        'image_dimensions_large': 'చిత్ర కొలతలు చాలా పెద్దవిగా ఉన్నాయి. చిన్న చిత్రాన్ని ఎంచుకోండి.',
        'invalid_image': 'ఎంచుకున్న ఫైల్ సరైన చిత్రం కాదు.',
        'read_image_failed': 'ఎంచుకున్న చిత్రాన్ని చదవడం సాధ్యపడలేదు.',
        'field_required': 'ఇప్పటికే ఉన్న క్షేత్రాన్ని ఎంచుకోండి లేదా కొత్త క్షేత్రం పేరును నమోదు చేయండి.',
        'field_too_long': 'క్షేత్రం పేరు 80 అక్షరాలకు మించకూడదు.',
        'select_image_first': 'దయచేసి ముందుగా పొలం చిత్రాన్ని ఎంచుకోండి.',
        'invalid_image_format': 'చిత్ర ఫార్మాట్ చెల్లదు. JPG, PNG లేదా WEBP ఎంచుకోండి.',
        'total_analyses_label': 'మొత్తం విశ్లేషణలు',
        'average_weed_label': 'సగటు కలుపు విస్తీర్ణం',
        'high_priority': 'అధిక ప్రాధాన్యత',
        'medium_priority': 'మధ్యస్థ ప్రాధాన్యత',
        'low_priority': 'తక్కువ ప్రాధాన్యత',
        'field_comparison': 'క్షేత్రాల వారీ పోలిక',
        'field_comparison_description': 'మీ వ్యవసాయ క్షేత్రాలలో కలుపు మరియు మంచి పంటల విస్తీర్ణాన్ని పోల్చండి.',
        'field_analyses_count': 'విశ్లేషణలు',
        'chart_aria': 'కలుపు మరియు పంటల విస్తీర్ణాన్ని చూపించే పై చార్ట్',
        'chart_note': 'పై చార్ట్ కలుపు మరియు పంటల సాపేక్ష విస్తీర్ణాన్ని చూపిస్తుంది. మొత్తం 100%.',
        'good_plants': 'మంచి మొక్కలు (పంట)',
        'no_chart_data': 'కలుపు లేదా పంట పిక్సెల్‌లు నమోదు కాలేదు.',
        'no_field_comparison': 'ఇంకా క్షేత్ర పోలిక సమాచారం లేదు. గ్రాఫ్‌ల కోసం పొలం చిత్రాలను అప్‌లోడ్ చేసి విశ్లేషించండి.',
        'individual_field_analysis': 'వ్యక్తిగత క్షేత్ర విశ్లేషణ',
        'individual_field_description': 'సేవ్ చేసిన ప్రతి క్షేత్రానికి సగటు విస్తీర్ణ గణాంకాలు. ఈ డాష్‌బోర్డ్‌లో అంచనా చిత్రాలు చూపబడవు.',
        'unassigned_field': 'కేటాయించని క్షేత్రం',
        'average_good_plants': 'సగటు మంచి పంటల విస్తీర్ణం',
        'no_field_analyses': 'ఇంకా క్షేత్ర విశ్లేషణలు సేవ్ చేయబడలేదు. ఇక్కడ క్షేత్రాల వారీ గణాంకాలను చూడటానికి పొలం చిత్రాన్ని విశ్లేషించండి.',
        'dashboard_description': 'మీ సేవ్ చేసిన పొలం విశ్లేషణల వివరాలు. ప్రతి క్షేత్రం పై చార్ట్ ద్వారా కలుపు మరియు మంచి పంటల విస్తీర్ణాన్ని పోల్చండి.',
        'footer_text': 'స్మార్ట్ కలుపు మొక్కల గుర్తింపు వ్యవస్థ | U-Net',
        'invalid_image_format_message': 'దయచేసి JPG, JPEG, PNG లేదా WEBP చిత్రాన్ని ఎంచుకోండి.',
        'image_size_error': 'చిత్ర పరిమాణం 8 MB మించకూడదు.',
        'image_dimensions_error': 'చిత్ర కొలతలు చాలా పెద్దవిగా ఉన్నాయి. చిన్న చిత్రాన్ని ఎంచుకోండి.',
        'invalid_selected_image': 'ఎంచుకున్న ఫైల్ సరైన చిత్రం కాదు.',
        'image_read_error': 'ఎంచుకున్న చిత్రాన్ని చదవడం సాధ్యపడలేదు.',
        'field_selection_error': 'ఇప్పటికే ఉన్న క్షేత్రాన్ని ఎంచుకోండి లేదా కొత్త క్షేత్రం పేరును నమోదు చేయండి.',
        'field_length_error': 'క్షేత్రం పేరు 80 అక్షరాలకు మించకూడదు.',
        'no_image_selected': 'దయచేసి ముందుగా పొలం చిత్రాన్ని ఎంచుకోండి.',
        'pie_no_data': 'కలుపు లేదా పంట పిక్సెల్‌లు నమోదు కాలేదు.',
        'recommendations': 'కలుపు మొక్కల నియంత్రణ సూచనలు',
        'recommendations_description': 'అంచనా వేసిన కలుపు మొక్కల విస్తీర్ణం ఆధారంగా సిఫార్సు చేసిన నియంత్రణ పద్ధతులు.',
        'low_weed_priority': 'తక్కువ కలుపు మొక్కల స్థాయి',
        'low_method_1_title': 'చేతితో కలుపు మొక్కలను తొలగించడం',
        'low_method_1': 'నేల కొద్దిగా తేమగా ఉన్నప్పుడు చిన్న కలుపు మొక్కలను చేతితో తొలగించండి. మొక్క అడుగు భాగాన్ని పట్టుకుని మెల్లగా లాగి, తిరిగి పెరగకుండా సాధ్యమైనంతవరకు వేర్లతో సహా తొలగించండి.',
        'low_method_2_title': 'మల్చింగ్ (నేలపై కవచం వేయడం)',
        'low_method_2': 'పంటల చుట్టూ ఉన్న ఖాళీ నేలపై ఎండిన ఆకులు, గడ్డి లేదా తగిన సేంద్రియ పదార్థాలను పరచండి. ఇది కలుపు విత్తనాలకు సూర్యకాంతి అందకుండా చేసి కొత్త కలుపు మొక్కలు పెరగకుండా సహాయపడుతుంది.',
        'low_method_3_title': 'పొలాన్ని క్రమం తప్పకుండా పరిశీలించడం',
        'low_method_3': 'ముఖ్యంగా వర్షం లేదా నీటిపారుదల తర్వాత పొలాన్ని క్రమం తప్పకుండా పరిశీలించండి. కొత్తగా మొలిచే కలుపు మొక్కలను గుర్తించి, అవి వ్యాపించకముందే లేదా విత్తనాలు ఉత్పత్తి చేయకముందే తొలగించండి.',
        'medium_weed_priority': 'మధ్యస్థ కలుపు మొక్కల స్థాయి',
        'medium_method_1_title': 'పైపొరలో తవ్వడం మరియు యంత్రాలతో తొలగించడం',
        'medium_method_1': 'పంట వరుసల మధ్య కలుపు మొక్కలను తొలగించడానికి చేతి గొడ్డలి లేదా తగిన యాంత్రిక కలుపు తొలగింపు పరికరాన్ని ఉపయోగించండి. పంట వేర్లు దెబ్బతినకుండా నేల పైపొర వద్ద మాత్రమే పని చేసి, తొలగించిన కలుపు మొక్కలను సేకరించండి.',
        'medium_method_2_title': 'విభాగాల వారీగా కలుపు నియంత్రణ',
        'medium_method_2': 'కలుపు ఎక్కువగా ఉన్న పొలాన్ని చిన్న భాగాలుగా విభజించి, ఒక్కో భాగంలో కలుపును తొలగించండి. పంటలకు దగ్గరగా ఉన్న కలుపును చేతితో తొలగించి, ఖాళీ ప్రదేశాల్లో తగిన పనిముట్లను ఉపయోగించండి.',
        'medium_method_3_title': 'మల్చింగ్ మరియు కవర్ పంటలు',
        'medium_method_3': 'ఇప్పటికే ఉన్న కలుపు మొక్కలను తొలగించిన తర్వాత నేలపై సేంద్రియ కవచాన్ని పరచండి. తగిన పంట కాలాల్లో కవర్ పంటలను పెంచడం ద్వారా ఖాళీ నేలను కప్పి, కలుపు మొక్కల పెరుగుదలను తగ్గించవచ్చు.',
        'high_weed_priority': 'అధిక కలుపు మొక్కల స్థాయి',
        'high_method_1_title': 'తీవ్ర యాంత్రిక కలుపు నియంత్రణ',
        'high_method_1': 'పంట వరుసల మధ్య ఎక్కువగా పెరిగిన కలుపును నియంత్రించడానికి తగిన యాంత్రిక పరికరాలు లేదా పునరావృతంగా నేల పైపొరను దున్నే పద్ధతిని ఉపయోగించండి. పంట మొక్కల దగ్గర జాగ్రత్తగా పని చేసి, విత్తనాలు ఏర్పడకముందే మిగిలిన కలుపును తొలగించండి.',
        'high_method_2_title': 'సురక్షితమైన మరియు లక్ష్యిత కలుపు మందుల వినియోగం',
        'high_method_2': 'చేతితో లేదా యంత్రాలతో కలుపును నియంత్రించడం సరిపోనప్పుడు, పంటకు మరియు లక్ష్య కలుపు మొక్కలకు అనుకూలమైన స్థానికంగా ఆమోదించబడిన కలుపు మందును మాత్రమే పరిగణించండి. పంటలకు నష్టం జరగకుండా మందు లేబుల్\u200cలోని మోతాదు, వాడే సమయం మరియు భద్రతా సూచనలను పాటించండి.',
        'high_method_3_title': 'సమగ్ర కలుపు నిర్వహణ',
        'high_method_3': 'ఎక్కువగా పెరిగిన కలుపును నియంత్రించడానికి చేతితో తొలగించడం, యాంత్రిక నియంత్రణ మరియు మల్చింగ్ పద్ధతులను కలిపి ఉపయోగించండి. పెద్ద కలుపు మొక్కలను తొలగించి, ఖాళీ నేలను కప్పి, తిరిగి పెరిగే కలుపును నియంత్రించడానికి తరచుగా పరిశీలించండి.',
        'important_note': 'ముఖ్యమైన గమనిక',
        'weed_safety_note': 'ఈ సూచనలు అప్\u200cలోడ్ చేసిన చిత్రంలో అంచనా వేసిన కలుపు మొక్కల విస్తీర్ణం ఆధారంగా రూపొందించబడ్డాయి. పంట మరియు పొల పరిస్థితులకు అనుగుణంగా సరైన పద్ధతులను ఎంచుకోండి. పంటలకు నష్టం జరగకుండా ఎల్లప్పుడూ జాగ్రత్తలు తీసుకోండి.',
    },

    'ta': {
        'app_name': 'ஸ்மார்ட் களை கண்டறிதல் அமைப்பு',
        'tagline': 'செயற்கை நுண்ணறிவு சார்ந்த ஸ்மார்ட் விவசாயம்',
        'welcome': 'வரவேற்கிறோம்',
        'home': 'முகப்பு',
        'dashboard': 'டாஷ்போர்டு',
        'history': 'வரலாறு',
        'profile': 'சுயவிவரம்',
        'logout': 'வெளியேறு',
        'login': 'உள்நுழை',
        'register': 'பதிவு',
        'language': 'மொழி',
        'select_language': 'மொழியைத் தேர்ந்தெடுக்கவும்',
        'dark_mode': 'இருண்ட பயன்முறை',
        'light_mode': 'ஒளி பயன்முறை',
        'assistant': 'AI உதவியாளர்',
        'assistant_title': 'ஸ்மார்ட் விவசாய உதவியாளர்',
        'assistant_welcome': 'வணக்கம்! களை கண்டறிதலில் நான் உங்களுக்கு எவ்வாறு உதவலாம்?',
        'assistant_greeting': 'வணக்கம்! களைகள், வயல் படங்கள் அல்லது பயிர்களுக்கு பாதுகாப்பான களை மேலாண்மை பற்றி கேளுங்கள்.',
        'assistant_ready': 'தயார்',
        'ask_question': 'ஏதாவது கேளுங்கள்...',
        'ask_question_long': 'ஒரு கேள்வியைக் கேளுங்கள்...',
        'speak': 'பேசுங்கள்',
        'send': 'அனுப்பு',
        'close': 'உதவியாளரை மூடு',
        'open_assistant': 'AI உதவியாளரைத் திற',
        'assistant_language': 'உதவியாளர் மொழி',
        'auto_detect': 'தானாக கண்டறி',
        'upload_image': 'வயல் படத்தை பதிவேற்றவும்',
        'upload_instructions': 'பயிர்கள் மற்றும் களைகளை அடையாளம் காண உங்கள் விவசாய வயலின் தெளிவான படத்தை பதிவேற்றவும்.',
        'supported_formats': 'ஆதரிக்கப்படும் வடிவங்கள்: JPG, JPEG, PNG, WEBP',
        'analyze': 'வயல் படத்தை பகுப்பாய்வு செய்க',
        'analyzing': 'படம் பகுப்பாய்வு செய்யப்படுகிறது...',
        'prediction_result': 'வயல் பகுப்பாய்வு முடிவுகள்',
        'weed_coverage': 'களை பரப்பளவு',
        'crop_coverage': 'பயிர் பரப்பளவு',
        'soil_background': 'மண் / பின்னணி',
        'weed_severity': 'களையின் தீவிரம்',
        'confidence': 'நம்பகத்தன்மை',
        'weed_pixels': 'களை பிக்சல்கள்',
        'source_image': 'அசல் படம்',
        'overlay_image': 'பிரிவு மேலடுக்கு படம்',
        'weed_mask': 'களை மாஸ்க்',
        'recommendation': 'பரிந்துரை',
        'recent_analyses': 'சமீபத்திய பகுப்பாய்வுகள்',
        'date': 'தேதி',
        'full_name': 'முழு பெயர்',
        'email': 'மின்னஞ்சல்',
        'phone': 'தொலைபேசி',
        'password': 'கடவுச்சொல்',
        'confirm_password': 'கடவுச்சொல்லை உறுதிப்படுத்தவும்',
        'create_account': 'கணக்கை உருவாக்கவும்',
        'total_analyses': 'மொத்த பகுப்பாய்வுகள்',
        'average_weed': 'சராசரி களை பரப்பளவு',
        'high': 'அதிகம்',
        'medium': 'நடுத்தரம்',
        'low': 'குறைவு',
        'no_analyses': 'பகுப்பாய்வுகள் எதுவும் இல்லை',
        'farm_location': 'பண்ணை இடம்',
        'save_changes': 'மாற்றங்களைச் சேமிக்கவும்',
        'update_profile': 'சுயவிவரத்தைப் புதுப்பிக்கவும்',
        'guest_limit': 'இலவச கணிப்புகள்',
        'remaining': 'மீதமுள்ளது',
        'used': 'பயன்படுத்தப்பட்டது',
        'guest_limit_used': 'உங்களுக்கான ஐந்து இலவச கணிப்புகளும் பயன்படுத்தப்பட்டுவிட்டன.',
        'login_to_continue': 'தொடர்ந்து பயன்படுத்த உள்நுழையவும் அல்லது பதிவு செய்யவும்.',
        'select_field': 'விவசாய வயலைத் தேர்ந்தெடுக்கவும்',
        'existing_field': 'ஏற்கனவே உள்ள வயலைத் தேர்ந்தெடுக்கவும்',
        'new_field_prompt': 'அல்லது கீழே புதிய வயல் பெயரை உள்ளிடவும்.',
        'new_field_name': 'புதிய வயல் பெயரை உள்ளிடவும்',
        'field_name': 'விவசாய வயல் பெயர்',
        'field_name_placeholder': 'வயல் பெயரை உள்ளிடவும் (எ.கா. வடக்கு வயல்)',
        'field_name_help': 'சேமிக்கப்பட்ட பகுப்பாய்வுகளை ஒழுங்குபடுத்த வயல் பெயர் உதவும்.',
        'select_image': 'ஒரு படத்தைத் தேர்ந்தெடுக்கவும்',
        'browse_image': 'வயல் படத்தைத் தேர்ந்தெடுக்க இங்கே கிளிக் செய்யவும்',
        'image_preview': 'தேர்ந்தெடுக்கப்பட்ட படத்தின் முன்னோட்டம்',
        'saved_history_note': 'உங்கள் பகுப்பாய்வுகள் கணக்கு வரலாற்றில் சேமிக்கப்படும்.',
        'guest_history_note': 'சேமிக்கப்பட்ட பகுப்பாய்வு வரலாற்றைப் பெற கணக்கை உருவாக்கவும்.',
        'register_here': 'இங்கே பதிவு செய்யவும்',
        'original_image': 'அசல் படம்',
        'segmentation_overlay': 'பிரிவு மேலடுக்கு படம்',
        'weed_mask_image': 'களை மாஸ்க்',
        'image_unavailable': 'படம் கிடைக்கவில்லை.',
        'image_failed': 'படத்தை ஏற்ற முடியவில்லை.',
        'detection_details': 'கண்டறிதல் விவரங்கள்',
        'field_name_label': 'வயல் பெயர்',
        'weed_location': 'களையின் இடம்',
        'weed_pixel_count': 'களை பிக்சல் எண்ணிக்கை',
        'suppression_status': 'கட்டுப்பாட்டு நிலை',
        'post_treatment_status': 'சிகிச்சைக்குப் பிந்தைய நிலை',
        'conceptual_stage_status': 'இது கருத்தாக்க நிலை: கண்டறியப்பட்ட களைகள் உள்ள பகுதிகளில் இலக்கு வெப்பக் கட்டுப்பாடு செய்யலாம். இறந்த களைகளின் உயிர்த்தொகையை இயற்கையாக மட்குவதற்காக அதே இடத்தில் விடலாம்.',
        'no_treatment_required': 'சிகிச்சை தேவையில்லை',
        'class_distribution': 'வகைப் பகிர்வு',
        'weed': 'களை',
        'crop': 'பயிர்',
        'background_soil': 'பின்னணி / மண்',
        'total': 'மொத்தம்',
        'natural_management': 'இயற்கையான களை மேலாண்மை பரிந்துரைகள்',
        'low_coverage': 'குறைந்த களை பரப்பளவு',
        'moderate_coverage': 'மிதமான களை பரப்பளவு',
        'high_coverage': 'அதிக களை பரப்பளவு',
        'low_coverage_text': 'குறைந்த அளவு களைகள் கண்டறியப்பட்டுள்ளன. பாதிக்கப்பட்ட பகுதியைக் கண்காணித்து தேவையான இடங்களில் இலக்கு சார்ந்த கட்டுப்பாட்டு முறைகளைப் பயன்படுத்தவும்.',
        'moderate_coverage_text': 'மிதமான அளவு களைகள் கண்டறியப்பட்டுள்ளன. அருகிலுள்ள பயிர்களைப் பாதுகாத்து கவனமாக மேலாண்மை செய்யவும்.',
        'high_coverage_text': 'அதிக அளவு களைகள் கண்டறியப்பட்டுள்ளன. பயிர்களுக்கு சேதம் ஏற்படாமல் வயலை ஆய்வு செய்து மேலாண்மை திட்டமிடவும்.',
        'low_priority': 'முன்னுரிமை: குறைவு',
        'medium_priority': 'முன்னுரிமை: நடுத்தரம்',
        'high_priority': 'முன்னுரிமை: அதிகம்',
        'treatment_note': 'எந்த சிகிச்சையும் பயிர்களுக்கு பாதுகாப்பானதாகவும் களை வகை மற்றும் வயல் நிலைமைக்கு ஏற்றதாகவும் இருக்க வேண்டும்.',
        'analyze_another': 'மற்றொரு படத்தைப் பகுப்பாய்வு செய்க',
        'view_history': 'பகுப்பாய்வு வரலாற்றைக் காண்க',
        'register_save_history': 'வரலாற்றைச் சேமிக்க பதிவு செய்யவும்',
        'invalid_format': 'JPG, JPEG, PNG அல்லது WEBP படத்தைத் தேர்ந்தெடுக்கவும்.',
        'file_too_large': 'படத்தின் அளவு 8 MB-ஐத் தாண்டக்கூடாது.',
        'image_dimensions_large': 'படத்தின் அளவுகள் மிகப் பெரியவை. சிறிய படத்தைத் தேர்ந்தெடுக்கவும்.',
        'invalid_image': 'தேர்ந்தெடுக்கப்பட்ட கோப்பு சரியான படம் அல்ல.',
        'read_image_failed': 'தேர்ந்தெடுக்கப்பட்ட படத்தைப் படிக்க முடியவில்லை.',
        'field_required': 'ஏற்கனவே உள்ள வயலைத் தேர்ந்தெடுக்கவும் அல்லது புதிய வயல் பெயரை உள்ளிடவும்.',
        'field_too_long': 'வயல் பெயர் 80 எழுத்துகளுக்கு மிகாமல் இருக்க வேண்டும்.',
        'select_image_first': 'முதலில் வயல் படத்தைத் தேர்ந்தெடுக்கவும்.',
        'invalid_image_format': 'தவறான பட வடிவம். JPG, PNG அல்லது WEBP தேர்ந்தெடுக்கவும்.',
        'total_analyses_label': 'மொத்த பகுப்பாய்வுகள்',
        'average_weed_label': 'சராசரி களை பரப்பளவு',
        'high_priority': 'அதிக முன்னுரிமை',
        'medium_priority': 'நடுத்தர முன்னுரிமை',
        'low_priority': 'குறைந்த முன்னுரிமை',
        'field_comparison': 'வயல் வாரியான ஒப்பீடு',
        'field_comparison_description': 'உங்கள் வயல்களில் களை மற்றும் நல்ல பயிர்களின் பரப்பளவை ஒப்பிடுங்கள்.',
        'field_analyses_count': 'பகுப்பாய்வுகள்',
        'chart_aria': 'களை மற்றும் பயிர் பரப்பளவைக் காட்டும் வட்ட விளக்கப்படம்',
        'chart_note': 'வட்ட விளக்கப்படம் களை மற்றும் பயிர்களின் ஒப்பீட்டு பரப்பளவைக் காட்டுகிறது. மொத்தம் 100%.',
        'good_plants': 'நல்ல தாவரங்கள் (பயிர்)',
        'no_chart_data': 'களை அல்லது பயிர் பிக்சல்கள் பதிவு செய்யப்படவில்லை.',
        'no_field_comparison': 'வயல் ஒப்பீட்டுத் தரவு இன்னும் இல்லை. வரைபடங்களை உருவாக்க வயல் படங்களை பதிவேற்றி பகுப்பாய்வு செய்யவும்.',
        'individual_field_analysis': 'தனிப்பட்ட வயல் பகுப்பாய்வு',
        'individual_field_description': 'சேமிக்கப்பட்ட ஒவ்வொரு வயலுக்குமான சராசரி பரப்பளவு புள்ளிவிவரங்கள். இந்த டாஷ்போர்டில் கணிப்பு படங்கள் காட்டப்படாது.',
        'unassigned_field': 'ஒதுக்கப்படாத வயல்',
        'average_good_plants': 'சராசரி நல்ல பயிர்கள்',
        'no_field_analyses': 'வயல் பகுப்பாய்வுகள் எதுவும் சேமிக்கப்படவில்லை. இங்கு வயல் வாரியான விவரங்களைக் காண ஒரு வயல் படத்தைப் பகுப்பாய்வு செய்யவும்.',
        'dashboard_description': 'சேமிக்கப்பட்ட வயல் பகுப்பாய்வுகளின் கண்ணோட்டம். ஒவ்வொரு வயலின் வட்ட விளக்கப்படங்கள் மூலம் களை மற்றும் நல்ல பயிர் பரப்பளவை ஒப்பிடுங்கள்.',
        'footer_text': 'ஸ்மார்ட் களை கண்டறிதல் அமைப்பு | U-Net',
        'invalid_image_format_message': 'JPG, JPEG, PNG அல்லது WEBP படத்தைத் தேர்ந்தெடுக்கவும்.',
        'image_size_error': 'படத்தின் அளவு 8 MB-ஐத் தாண்டக்கூடாது.',
        'image_dimensions_error': 'படத்தின் அளவுகள் மிகப் பெரியவை. சிறிய படத்தைத் தேர்ந்தெடுக்கவும்.',
        'invalid_selected_image': 'தேர்ந்தெடுக்கப்பட்ட கோப்பு சரியான படம் அல்ல.',
        'image_read_error': 'தேர்ந்தெடுக்கப்பட்ட படத்தைப் படிக்க முடியவில்லை.',
        'field_selection_error': 'ஏற்கனவே உள்ள வயலைத் தேர்ந்தெடுக்கவும் அல்லது புதிய வயல் பெயரை உள்ளிடவும்.',
        'field_length_error': 'வயல் பெயர் 80 எழுத்துகளுக்கு மிகாமல் இருக்க வேண்டும்.',
        'no_image_selected': 'முதலில் வயல் படத்தைத் தேர்ந்தெடுக்கவும்.',
        'pie_no_data': 'களை அல்லது பயிர் பிக்சல்கள் பதிவு செய்யப்படவில்லை.',
        'recommendations': 'களை கட்டுப்பாட்டு பரிந்துரைகள்',
        'recommendations_description': 'மதிப்பிடப்பட்ட களைகளின் பரப்பளவை அடிப்படையாகக் கொண்ட பரிந்துரைக்கப்பட்ட கட்டுப்பாட்டு முறைகள்.',
        'low_weed_priority': 'குறைந்த அளவு களைகள்',
        'low_method_1_title': 'கைகளால் களைகளை அகற்றுதல்',
        'low_method_1': 'மண் சிறிது ஈரமாக இருக்கும்போது சிறிய களைகளை கைகளால் அகற்றவும். களையின் அடிப்பகுதியைப் பிடித்து மெதுவாக இழுத்து, மீண்டும் வளர்வதைத் தடுக்க முடிந்தவரை வேருடன் அகற்றவும்.',
        'low_method_2_title': 'மூடாக்கு அமைத்தல்',
        'low_method_2': 'பயிர்களைச் சுற்றியுள்ள வெளிப்பட்ட மண்ணில் உலர்ந்த இலைகள், வைக்கோல் அல்லது பொருத்தமான இயற்கை பொருட்களைப் பரப்பவும். இது களை விதைகளுக்கு சூரிய ஒளி கிடைப்பதைத் தடுத்து, புதிய களைகள் வளர்வதைக் குறைக்கிறது.',
        'low_method_3_title': 'வயலைத் தொடர்ந்து கண்காணித்தல்',
        'low_method_3': 'குறிப்பாக மழை பெய்த பிறகும் அல்லது நீர்ப்பாசனத்திற்குப் பிறகும் வயலைத் தொடர்ந்து கண்காணிக்கவும். புதிதாக முளைக்கும் களைகளைக் கண்டறிந்து, அவை பரவுவதற்கு அல்லது விதைகளை உருவாக்குவதற்கு முன்பே அகற்றவும்.',
        'medium_weed_priority': 'மிதமான அளவு களைகள்',
        'medium_method_1_title': 'மேலோட்டமான மண்வெட்டி மற்றும் இயந்திர அகற்றுதல்',
        'medium_method_1': 'பயிர் வரிசைகளுக்கு இடையில் உள்ள களைகளை அகற்ற கை மண்வெட்டி அல்லது பொருத்தமான இயந்திரக் களை அகற்றியைப் பயன்படுத்தவும். பயிர்களின் வேர்கள் சேதமடையாமல் மண்ணின் மேற்பரப்பிற்கு அருகில் வேலை செய்து, அகற்றிய களைகளைச் சேகரிக்கவும்.',
        'medium_method_2_title': 'பகுதிவாரியான களை கட்டுப்பாடு',
        'medium_method_2': 'களைகள் அதிகம் உள்ள வயலைச் சிறிய பகுதிகளாகப் பிரித்து, ஒவ்வொரு பகுதியிலும் தனித்தனியாக களைகளை அகற்றவும். பயிர்களுக்கு அருகிலுள்ள களைகளை கைகளால் அகற்றி, திறந்த இடங்களில் பொருத்தமான கருவிகளைப் பயன்படுத்தவும்.',
        'medium_method_3_title': 'மூடாக்கு மற்றும் மூடுபயிர்கள்',
        'medium_method_3': 'ஏற்கனவே உள்ள களைகளை அகற்றிய பிறகு மண்ணில் இயற்கை மூடாக்கைப் பரப்பவும். பொருத்தமான பயிர் சுழற்சிக் காலங்களில் மூடுபயிர்களை வளர்ப்பதன் மூலம் காலியான மண்ணை மூடி, களைகளின் வளர்ச்சியைக் குறைக்கலாம்.',
        'high_weed_priority': 'அதிக அளவு களைகள்',
        'high_method_1_title': 'தீவிர இயந்திர களை கட்டுப்பாடு',
        'high_method_1': 'பயிர் வரிசைகளுக்கு இடையில் அடர்த்தியாக வளர்ந்துள்ள களைகளைக் கட்டுப்படுத்த பொருத்தமான இயந்திரக் களை அகற்றிகளை அல்லது மீண்டும் மீண்டும் மேலோட்டமாக உழும் முறையைப் பயன்படுத்தவும். பயிர்களுக்கு அருகில் கவனமாகச் செயல்பட்டு, விதைகள் உருவாகும் முன்பே மீதமுள்ள களைகளை அகற்றவும்.',
        'high_method_2_title': 'பாதுகாப்பான மற்றும் இலக்கு சார்ந்த களைக்கொல்லி பயன்பாடு',
        'high_method_2': 'கைகளாலும் இயந்திரங்களாலும் களைகளைக் கட்டுப்படுத்துவது போதுமானதாக இல்லாதபோது, பயிருக்கும் குறிப்பிட்ட களைகளுக்கும் பொருத்தமான உள்ளூரில் அங்கீகரிக்கப்பட்ட களைக்கொல்லியை மட்டுமே பரிசீலிக்கவும். பயிர்கள் பாதிக்கப்படாமல் இருக்க அதன் லேபிளில் குறிப்பிடப்பட்ட அளவு, பயன்பாட்டு நேரம் மற்றும் பாதுகாப்பு வழிமுறைகளைப் பின்பற்றவும்.',
        'high_method_3_title': 'ஒருங்கிணைந்த களை மேலாண்மை',
        'high_method_3': 'அதிக அளவு களைகளைக் கட்டுப்படுத்த கைகளால் அகற்றுதல், இயந்திரக் கட்டுப்பாடு மற்றும் மூடாக்கு ஆகிய முறைகளை இணைத்துப் பயன்படுத்தவும். பெரிய களைகளை அகற்றி, காலியான மண்ணை மூடி, மீண்டும் வளரும் களைகளைக் கட்டுப்படுத்த தொடர்ந்து கண்காணிக்கவும்.',
        'important_note': 'முக்கிய குறிப்பு',
        'weed_safety_note': 'இந்த பரிந்துரைகள் பதிவேற்றப்பட்ட படத்தில் கணிக்கப்பட்ட களைகளின் பரப்பளவை அடிப்படையாகக் கொண்டவை. பயிர் மற்றும் வயல் நிலைமைகளுக்கு ஏற்ப பொருத்தமான முறைகளைத் தேர்ந்தெடுக்கவும். பயிர்களுக்கு சேதம் ஏற்படாமல் எப்போதும் முன்னெச்சரிக்கை நடவடிக்கைகளை மேற்கொள்ளவும்.',
    },

    'kn': {
        'app_name': 'ಸ್ಮಾರ್ಟ್ ಕಳೆ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ',
        'tagline': 'ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಆಧಾರಿತ ಸ್ಮಾರ್ಟ್ ಕೃಷಿ',
        'welcome': 'ಸ್ವಾಗತ',
        'home': 'ಮುಖಪುಟ',
        'dashboard': 'ಡ್ಯಾಶ್‌ಬೋರ್ಡ್',
        'history': 'ಇತಿಹಾಸ',
        'profile': 'ಪ್ರೊಫೈಲ್',
        'logout': 'ಲಾಗ್ ಔಟ್',
        'login': 'ಲಾಗಿನ್',
        'register': 'ನೋಂದಣಿ',
        'language': 'ಭಾಷೆ',
        'select_language': 'ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'dark_mode': 'ಡಾರ್ಕ್ ಮೋಡ್',
        'light_mode': 'ಲೈಟ್ ಮೋಡ್',
        'assistant': 'AI ಸಹಾಯಕ',
        'assistant_title': 'ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಹಾಯಕ',
        'assistant_welcome': 'ನಮಸ್ಕಾರ! ಕಳೆ ಪತ್ತೆಯಲ್ಲಿ ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?',
        'assistant_greeting': 'ನಮಸ್ಕಾರ! ಕಳೆಗಳು, ಹೊಲದ ಚಿತ್ರಗಳು ಅಥವಾ ಬೆಳೆಗಳಿಗೆ ಸುರಕ್ಷಿತವಾದ ಕಳೆ ನಿರ್ವಹಣೆ ಕುರಿತು ಪ್ರಶ್ನೆ ಕೇಳಿ.',
        'assistant_ready': 'ಸಿದ್ಧವಾಗಿದೆ',
        'ask_question': 'ಏನಾದರೂ ಕೇಳಿ...',
        'ask_question_long': 'ಪ್ರಶ್ನೆ ಕೇಳಿ...',
        'speak': 'ಮಾತನಾಡಿ',
        'send': 'ಕಳುಹಿಸಿ',
        'close': 'ಸಹಾಯಕನನ್ನು ಮುಚ್ಚಿ',
        'open_assistant': 'AI ಸಹಾಯಕನನ್ನು ತೆರೆಯಿರಿ',
        'assistant_language': 'ಸಹಾಯಕನ ಭಾಷೆ',
        'auto_detect': 'ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಗುರುತಿಸಿ',
        'upload_image': 'ಹೊಲದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ',
        'upload_instructions': 'ಬೆಳೆಗಳು ಮತ್ತು ಕಳೆಗಳನ್ನು ಗುರುತಿಸಲು ನಿಮ್ಮ ಕೃಷಿ ಹೊಲದ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.',
        'supported_formats': 'ಬೆಂಬಲಿತ ಸ್ವರೂಪಗಳು: JPG, JPEG, PNG, WEBP',
        'analyze': 'ಹೊಲದ ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ',
        'analyzing': 'ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...',
        'prediction_result': 'ಹೊಲದ ವಿಶ್ಲೇಷಣೆ ಫಲಿತಾಂಶಗಳು',
        'weed_coverage': 'ಕಳೆ ವ್ಯಾಪ್ತಿ',
        'crop_coverage': 'ಬೆಳೆ ವ್ಯಾಪ್ತಿ',
        'soil_background': 'ಮಣ್ಣು / ಹಿನ್ನೆಲೆ',
        'weed_severity': 'ಕಳೆಯ ತೀವ್ರತೆ',
        'confidence': 'ನಿಖರತೆಯ ಮಟ್ಟ',
        'weed_pixels': 'ಕಳೆ ಪಿಕ್ಸೆಲ್‌ಗಳು',
        'source_image': 'ಮೂಲ ಚಿತ್ರ',
        'overlay_image': 'ವಿಭಜನಾ ಓವರ್‌ಲೇ',
        'weed_mask': 'ಕಳೆ ಮಾಸ್ಕ್',
        'recommendation': 'ಶಿಫಾರಸು',
        'recent_analyses': 'ಇತ್ತೀಚಿನ ವಿಶ್ಲೇಷಣೆಗಳು',
        'date': 'ದಿನಾಂಕ',
        'full_name': 'ಪೂರ್ಣ ಹೆಸರು',
        'email': 'ಇಮೇಲ್',
        'phone': 'ದೂರವಾಣಿ',
        'password': 'ಪಾಸ್‌ವರ್ಡ್',
        'confirm_password': 'ಪಾಸ್‌ವರ್ಡ್ ದೃಢೀಕರಿಸಿ',
        'create_account': 'ಖಾತೆ ರಚಿಸಿ',
        'total_analyses': 'ಒಟ್ಟು ವಿಶ್ಲೇಷಣೆಗಳು',
        'average_weed': 'ಸರಾಸರಿ ಕಳೆ ವ್ಯಾಪ್ತಿ',
        'high': 'ಹೆಚ್ಚು',
        'medium': 'ಮಧ್ಯಮ',
        'low': 'ಕಡಿಮೆ',
        'no_analyses': 'ಯಾವುದೇ ವಿಶ್ಲೇಷಣೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ',
        'farm_location': 'ಕೃಷಿ ಸ್ಥಳ',
        'save_changes': 'ಬದಲಾವಣೆಗಳನ್ನು ಉಳಿಸಿ',
        'update_profile': 'ಪ್ರೊಫೈಲ್ ನವೀಕರಿಸಿ',
        'guest_limit': 'ಉಚಿತ ಭವಿಷ್ಯವಾಣಿಗಳು',
        'remaining': 'ಉಳಿದಿವೆ',
        'used': 'ಬಳಸಲಾಗಿದೆ',
        'guest_limit_used': 'ನಿಮ್ಮ ಐದು ಉಚಿತ ಭವಿಷ್ಯವಾಣಿಗಳನ್ನು ಬಳಸಲಾಗಿದೆ.',
        'login_to_continue': 'ಮುಂದುವರಿಸಲು ದಯವಿಟ್ಟು ಲಾಗಿನ್ ಆಗಿ ಅಥವಾ ನೋಂದಾಯಿಸಿ.',
        'select_field': 'ಕೃಷಿ ಹೊಲವನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'existing_field': 'ಈಗಾಗಲೇ ಇರುವ ಹೊಲವನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'new_field_prompt': 'ಅಥವಾ ಕೆಳಗೆ ಹೊಸ ಹೊಲದ ಹೆಸರನ್ನು ನಮೂದಿಸಿ.',
        'new_field_name': 'ಹೊಸ ಹೊಲದ ಹೆಸರನ್ನು ನಮೂದಿಸಿ',
        'field_name': 'ಕೃಷಿ ಹೊಲದ ಹೆಸರು',
        'field_name_placeholder': 'ಹೊಲದ ಹೆಸರನ್ನು ನಮೂದಿಸಿ (ಉದಾ: ಉತ್ತರದ ಹೊಲ)',
        'field_name_help': 'ನಿಮ್ಮ ಉಳಿಸಿದ ವಿಶ್ಲೇಷಣೆಗಳನ್ನು ವ್ಯವಸ್ಥಿತಗೊಳಿಸಲು ಹೊಲದ ಹೆಸರು ಸಹಾಯ ಮಾಡುತ್ತದೆ.',
        'select_image': 'ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'browse_image': 'ಹೊಲದ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಲು ಇಲ್ಲಿ ಕ್ಲಿಕ್ ಮಾಡಿ',
        'image_preview': 'ಆಯ್ಕೆಮಾಡಿದ ಚಿತ್ರದ ಪೂರ್ವವೀಕ್ಷಣೆ',
        'saved_history_note': 'ನಿಮ್ಮ ವಿಶ್ಲೇಷಣೆಗಳು ಖಾತೆಯ ಇತಿಹಾಸದಲ್ಲಿ ಉಳಿಸಲಾಗುತ್ತವೆ.',
        'guest_history_note': 'ಉಳಿಸಿದ ವಿಶ್ಲೇಷಣೆಗಳ ಇತಿಹಾಸವನ್ನು ಪಡೆಯಲು ಖಾತೆ ರಚಿಸಿ.',
        'register_here': 'ಇಲ್ಲಿ ನೋಂದಾಯಿಸಿ',
        'original_image': 'ಮೂಲ ಚಿತ್ರ',
        'segmentation_overlay': 'ವಿಭಜನಾ ಓವರ್‌ಲೇ',
        'weed_mask_image': 'ಕಳೆ ಮಾಸ್ಕ್',
        'image_unavailable': 'ಚಿತ್ರ ಲಭ್ಯವಿಲ್ಲ.',
        'image_failed': 'ಚಿತ್ರವನ್ನು ಲೋಡ್ ಮಾಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.',
        'detection_details': 'ಪತ್ತೆಹಚ್ಚುವಿಕೆಯ ವಿವರಗಳು',
        'field_name_label': 'ಹೊಲದ ಹೆಸರು',
        'weed_location': 'ಕಳೆಯ ಸ್ಥಳ',
        'weed_pixel_count': 'ಕಳೆ ಪಿಕ್ಸೆಲ್‌ಗಳ ಸಂಖ್ಯೆ',
        'suppression_status': 'ನಿಯಂತ್ರಣ ಸ್ಥಿತಿ',
        'post_treatment_status': 'ಚಿಕಿತ್ಸೆಯ ನಂತರದ ಸ್ಥಿತಿ',
        'conceptual_stage_status': 'ಇದು ಪರಿಕಲ್ಪನಾ ಹಂತ: ಪತ್ತೆಯಾದ ಕಳೆ ಪ್ರದೇಶಗಳನ್ನು ಗುರಿಯಾಗಿಸಿಕೊಂಡು ಉಷ್ಣ ನಿಯಂತ್ರಣ ಮಾಡಬಹುದು. ಸತ್ತ ಕಳೆ ಜೈವಿಕ ಅವಶೇಷಗಳನ್ನು ನೈಸರ್ಗಿಕವಾಗಿ ಕೊಳೆಯಲು ಅದೇ ಸ್ಥಳದಲ್ಲಿ ಬಿಡಬಹುದು.',
        'no_treatment_required': 'ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ',
        'class_distribution': 'ವರ್ಗಗಳ ಹಂಚಿಕೆ',
        'weed': 'ಕಳೆ',
        'crop': 'ಬೆಳೆ',
        'background_soil': 'ಹಿನ್ನೆಲೆ / ಮಣ್ಣು',
        'total': 'ಒಟ್ಟು',
        'natural_management': 'ನೈಸರ್ಗಿಕ ಕಳೆ ನಿರ್ವಹಣೆಯ ಶಿಫಾರಸುಗಳು',
        'low_coverage': 'ಕಡಿಮೆ ಕಳೆ ವ್ಯಾಪ್ತಿ',
        'moderate_coverage': 'ಮಧ್ಯಮ ಕಳೆ ವ್ಯಾಪ್ತಿ',
        'high_coverage': 'ಹೆಚ್ಚಿನ ಕಳೆ ವ್ಯಾಪ್ತಿ',
        'low_coverage_text': 'ಕಡಿಮೆ ಪ್ರಮಾಣದ ಕಳೆ ಪತ್ತೆಯಾಗಿದೆ. ಬಾಧಿತ ಪ್ರದೇಶವನ್ನು ಗಮನಿಸಿ ಅಗತ್ಯವಿದ್ದಲ್ಲಿ ನಿರ್ದಿಷ್ಟ ನಿಯಂತ್ರಣ ವಿಧಾನಗಳನ್ನು ಬಳಸಿ.',
        'moderate_coverage_text': 'ಮಧ್ಯಮ ಪ್ರಮಾಣದ ಕಳೆ ಪತ್ತೆಯಾಗಿದೆ. ಹತ್ತಿರದ ಬೆಳೆಗಳಿಗೆ ಹಾನಿಯಾಗದಂತೆ ಎಚ್ಚರಿಕೆಯಿಂದ ನಿರ್ವಹಣಾ ಯೋಜನೆ ರೂಪಿಸಿ.',
        'high_coverage_text': 'ಹೆಚ್ಚಿನ ಪ್ರಮಾಣದ ಕಳೆ ಪತ್ತೆಯಾಗಿದೆ. ಬೆಳೆಗಳಿಗೆ ಹಾನಿಯಾಗದಂತೆ ಹೊಲವನ್ನು ಪರಿಶೀಲಿಸಿ ನಿರ್ವಹಣೆಯನ್ನು ಯೋಜಿಸಿ.',
        'low_priority': 'ಆದ್ಯತೆ: ಕಡಿಮೆ',
        'medium_priority': 'ಆದ್ಯತೆ: ಮಧ್ಯಮ',
        'high_priority': 'ಆದ್ಯತೆ: ಹೆಚ್ಚು',
        'treatment_note': 'ಯಾವುದೇ ಚಿಕಿತ್ಸೆಯು ಬೆಳೆಗಳಿಗೆ ಸುರಕ್ಷಿತವಾಗಿರಬೇಕು ಮತ್ತು ಕಳೆಯ ವಿಧ ಹಾಗೂ ಹೊಲದ ಪರಿಸ್ಥಿತಿಗೆ ಸೂಕ್ತವಾಗಿರಬೇಕು.',
        'analyze_another': 'ಮತ್ತೊಂದು ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ',
        'view_history': 'ವಿಶ್ಲೇಷಣೆ ಇತಿಹಾಸವನ್ನು ವೀಕ್ಷಿಸಿ',
        'register_save_history': 'ಇತಿಹಾಸ ಉಳಿಸಲು ನೋಂದಾಯಿಸಿ',
        'invalid_format': 'ದಯವಿಟ್ಟು JPG, JPEG, PNG ಅಥವಾ WEBP ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ.',
        'file_too_large': 'ಚಿತ್ರದ ಗಾತ್ರ 8 MB ಮೀರಬಾರದು.',
        'image_dimensions_large': 'ಚಿತ್ರದ ಆಯಾಮಗಳು ತುಂಬಾ ದೊಡ್ಡದಾಗಿವೆ. ಚಿಕ್ಕ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ.',
        'invalid_image': 'ಆಯ್ಕೆಮಾಡಿದ ಫೈಲ್ ಮಾನ್ಯ ಚಿತ್ರವಲ್ಲ.',
        'read_image_failed': 'ಆಯ್ಕೆಮಾಡಿದ ಚಿತ್ರವನ್ನು ಓದಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.',
        'field_required': 'ಈಗಿರುವ ಹೊಲವನ್ನು ಆಯ್ಕೆಮಾಡಿ ಅಥವಾ ಹೊಸ ಹೊಲದ ಹೆಸರನ್ನು ನಮೂದಿಸಿ.',
        'field_too_long': 'ಹೊಲದ ಹೆಸರು 80 ಅಕ್ಷರಗಳನ್ನು ಮೀರಬಾರದು.',
        'select_image_first': 'ದಯವಿಟ್ಟು ಮೊದಲು ಹೊಲದ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ.',
        'invalid_image_format': 'ಅಮಾನ್ಯ ಚಿತ್ರ ಸ್ವರೂಪ. JPG, PNG ಅಥವಾ WEBP ಆಯ್ಕೆಮಾಡಿ.',
        'total_analyses_label': 'ಒಟ್ಟು ವಿಶ್ಲೇಷಣೆಗಳು',
        'average_weed_label': 'ಸರಾಸರಿ ಕಳೆ ವ್ಯಾಪ್ತಿ',
        'high_priority': 'ಹೆಚ್ಚಿನ ಆದ್ಯತೆ',
        'medium_priority': 'ಮಧ್ಯಮ ಆದ್ಯತೆ',
        'low_priority': 'ಕಡಿಮೆ ಆದ್ಯತೆ',
        'field_comparison': 'ಹೊಲವಾರು ಹೋಲಿಕೆ',
        'field_comparison_description': 'ನಿಮ್ಮ ಕೃಷಿ ಹೊಲಗಳಲ್ಲಿನ ಕಳೆ ಮತ್ತು ಉತ್ತಮ ಬೆಳೆಗಳ ವ್ಯಾಪ್ತಿಯನ್ನು ಹೋಲಿಸಿ.',
        'field_analyses_count': 'ವಿಶ್ಲೇಷಣೆಗಳು',
        'chart_aria': 'ಕಳೆ ಮತ್ತು ಬೆಳೆ ವ್ಯಾಪ್ತಿಯನ್ನು ತೋರಿಸುವ ವೃತ್ತ ಚಾರ್ಟ್',
        'chart_note': 'ವೃತ್ತ ಚಾರ್ಟ್ ಕಳೆ ಮತ್ತು ಬೆಳೆಗಳ ಸಾಪೇಕ್ಷ ವ್ಯಾಪ್ತಿಯನ್ನು ತೋರಿಸುತ್ತದೆ. ಒಟ್ಟು 100%.',
        'good_plants': 'ಉತ್ತಮ ಸಸ್ಯಗಳು (ಬೆಳೆ)',
        'no_chart_data': 'ಕಳೆ ಅಥವಾ ಬೆಳೆ ಪಿಕ್ಸೆಲ್‌ಗಳು ದಾಖಲಾಗಿಲ್ಲ.',
        'no_field_comparison': 'ಇನ್ನೂ ಹೊಲ ಹೋಲಿಕೆ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ಗ್ರಾಫ್‌ಗಳನ್ನು ರಚಿಸಲು ಹೊಲದ ಚಿತ್ರಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ವಿಶ್ಲೇಷಿಸಿ.',
        'individual_field_analysis': 'ಪ್ರತ್ಯೇಕ ಹೊಲದ ವಿಶ್ಲೇಷಣೆ',
        'individual_field_description': 'ಉಳಿಸಿದ ಪ್ರತಿಯೊಂದು ಹೊಲದ ಸರಾಸರಿ ವ್ಯಾಪ್ತಿ ಅಂಕಿಅಂಶಗಳು. ಈ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್‌ನಲ್ಲಿ ಭವಿಷ್ಯವಾಣಿ ಚಿತ್ರಗಳನ್ನು ತೋರಿಸಲಾಗುವುದಿಲ್ಲ.',
        'unassigned_field': 'ನಿಯೋಜಿಸದ ಹೊಲ',
        'average_good_plants': 'ಸರಾಸರಿ ಉತ್ತಮ ಬೆಳೆಗಳು',
        'no_field_analyses': 'ಯಾವುದೇ ಹೊಲದ ವಿಶ್ಲೇಷಣೆಗಳನ್ನು ಉಳಿಸಲಾಗಿಲ್ಲ. ಹೊಲವಾರು ಅಂಕಿಅಂಶಗಳನ್ನು ನೋಡಲು ಹೊಲದ ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ.',
        'dashboard_description': 'ನಿಮ್ಮ ಉಳಿಸಿದ ಹೊಲದ ವಿಶ್ಲೇಷಣೆಗಳ ಅವಲೋಕನ. ಪ್ರತಿ ಹೊಲದ ವೃತ್ತ ಚಾರ್ಟ್‌ಗಳ ಮೂಲಕ ಕಳೆ ಮತ್ತು ಉತ್ತಮ ಬೆಳೆಗಳ ವ್ಯಾಪ್ತಿಯನ್ನು ಹೋಲಿಸಿ.',
        'footer_text': 'ಸ್ಮಾರ್ಟ್ ಕಳೆ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ | U-Net',
        'invalid_image_format_message': 'ದಯವಿಟ್ಟು JPG, JPEG, PNG ಅಥವಾ WEBP ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ.',
        'image_size_error': 'ಚಿತ್ರದ ಗಾತ್ರ 8 MB ಮೀರಬಾರದು.',
        'image_dimensions_error': 'ಚಿತ್ರದ ಆಯಾಮಗಳು ತುಂಬಾ ದೊಡ್ಡದಾಗಿವೆ. ಚಿಕ್ಕ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ.',
        'invalid_selected_image': 'ಆಯ್ಕೆಮಾಡಿದ ಫೈಲ್ ಮಾನ್ಯ ಚಿತ್ರವಲ್ಲ.',
        'image_read_error': 'ಆಯ್ಕೆಮಾಡಿದ ಚಿತ್ರವನ್ನು ಓದಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.',
        'field_selection_error': 'ಈಗಿರುವ ಹೊಲವನ್ನು ಆಯ್ಕೆಮಾಡಿ ಅಥವಾ ಹೊಸ ಹೊಲದ ಹೆಸರನ್ನು ನಮೂದಿಸಿ.',
        'field_length_error': 'ಹೊಲದ ಹೆಸರು 80 ಅಕ್ಷರಗಳನ್ನು ಮೀರಬಾರದು.',
        'no_image_selected': 'ದಯವಿಟ್ಟು ಮೊದಲು ಹೊಲದ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ.',
        'pie_no_data': 'ಕಳೆ ಅಥವಾ ಬೆಳೆ ಪಿಕ್ಸೆಲ್‌ಗಳು ದಾಖಲಾಗಿಲ್ಲ.',
        'recommendations': 'ಕಳೆ ನಿಯಂತ್ರಣ ಸಲಹೆಗಳು',
        'recommendations_description': 'ಅಂದಾಜು ಮಾಡಿದ ಕಳೆ ಆವರಣದ ಆಧಾರದ ಮೇಲೆ ಶಿಫಾರಸು ಮಾಡಲಾದ ನಿಯಂತ್ರಣ ವಿಧಾನಗಳು.',
        'low_weed_priority': 'ಕಡಿಮೆ ಪ್ರಮಾಣದ ಕಳೆಗಳು',
        'low_method_1_title': 'ಕೈಯಿಂದ ಕಳೆ ತೆಗೆಯುವುದು',
        'low_method_1': 'ಮಣ್ಣು ಸ್ವಲ್ಪ ತೇವವಾಗಿರುವಾಗ ಸಣ್ಣ ಕಳೆಗಳನ್ನು ಕೈಯಿಂದ ತೆಗೆಯಿರಿ. ಕಳೆಯ ಬುಡವನ್ನು ಹಿಡಿದು ನಿಧಾನವಾಗಿ ಎಳೆಯಿರಿ ಮತ್ತು ಮತ್ತೆ ಬೆಳೆಯುವುದನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಸಾಧ್ಯವಾದಷ್ಟು ಬೇರುಗಳ ಸಮೇತ ತೆಗೆದುಹಾಕಿ.',
        'low_method_2_title': 'ಮಲ್ಚಿಂಗ್ (ಮಣ್ಣಿನ ಹೊದಿಕೆ)',
        'low_method_2': 'ಬೆಳೆಗಳ ಸುತ್ತಲಿನ ತೆರೆದ ಮಣ್ಣಿನ ಮೇಲೆ ಒಣ ಎಲೆಗಳು, ಹುಲ್ಲು ಅಥವಾ ಸೂಕ್ತ ಸಾವಯವ ವಸ್ತುಗಳನ್ನು ಹರಡಿ. ಇದು ಕಳೆ ಬೀಜಗಳಿಗೆ ಸೂರ್ಯನ ಬೆಳಕು ತಲುಪುವುದನ್ನು ತಡೆದು ಹೊಸ ಕಳೆಗಳ ಬೆಳವಣಿಗೆಯನ್ನು ಕಡಿಮೆ ಮಾಡುತ್ತದೆ.',
        'low_method_3_title': 'ಹೊಲವನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸುವುದು',
        'low_method_3': 'ವಿಶೇಷವಾಗಿ ಮಳೆ ಅಥವಾ ನೀರಾವರಿಯ ನಂತರ ಹೊಲವನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ. ಹೊಸದಾಗಿ ಬೆಳೆಯುತ್ತಿರುವ ಕಳೆಗಳನ್ನು ಗುರುತಿಸಿ, ಅವು ಹರಡುವ ಅಥವಾ ಬೀಜ ಉತ್ಪಾದಿಸುವ ಮೊದಲು ತೆಗೆದುಹಾಕಿ.',
        'medium_weed_priority': 'ಮಧ್ಯಮ ಪ್ರಮಾಣದ ಕಳೆಗಳು',
        'medium_method_1_title': 'ಮೇಲ್ಮೈ ಗುದ್ದಲಿ ಮತ್ತು ಯಾಂತ್ರಿಕ ತೆರವು',
        'medium_method_1': 'ಬೆಳೆ ಸಾಲುಗಳ ನಡುವಿನ ಕಳೆಗಳನ್ನು ತೆಗೆಯಲು ಕೈ ಗುದ್ದಲಿ ಅಥವಾ ಸೂಕ್ತ ಯಾಂತ್ರಿಕ ಕಳೆ ತೆಗೆಯುವ ಸಾಧನವನ್ನು ಬಳಸಿ. ಬೆಳೆಗಳ ಬೇರುಗಳಿಗೆ ಹಾನಿಯಾಗದಂತೆ ಮಣ್ಣಿನ ಮೇಲ್ಮೈ ಬಳಿ ಕೆಲಸ ಮಾಡಿ ಮತ್ತು ತೆಗೆದ ಕಳೆಗಳನ್ನು ಸಂಗ್ರಹಿಸಿ.',
        'medium_method_2_title': 'ವಿಭಾಗವಾರು ಕಳೆ ನಿಯಂತ್ರಣ',
        'medium_method_2': 'ಕಳೆಗಳು ಹೆಚ್ಚಿರುವ ಹೊಲವನ್ನು ಸಣ್ಣ ವಿಭಾಗಗಳಾಗಿ ವಿಂಗಡಿಸಿ, ಒಂದೊಂದು ವಿಭಾಗದಲ್ಲಿ ಪ್ರತ್ಯೇಕವಾಗಿ ಕಳೆಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ. ಬೆಳೆಗಳ ಹತ್ತಿರವಿರುವ ಕಳೆಗಳನ್ನು ಕೈಯಿಂದ ತೆಗೆದುಹಾಕಿ ಮತ್ತು ತೆರೆದ ಜಾಗಗಳಲ್ಲಿ ಸೂಕ್ತ ಸಾಧನಗಳನ್ನು ಬಳಸಿ.',
        'medium_method_3_title': 'ಮಲ್ಚಿಂಗ್ ಮತ್ತು ಹೊದಿಕೆ ಬೆಳೆಗಳು',
        'medium_method_3': 'ಈಗಿರುವ ಕಳೆಗಳನ್ನು ತೆಗೆದ ನಂತರ ಮಣ್ಣಿನ ಮೇಲೆ ಸಾವಯವ ಹೊದಿಕೆಯನ್ನು ಹರಡಿ. ಸೂಕ್ತ ಬೆಳೆ ಚಕ್ರಗಳಲ್ಲಿ ಹೊದಿಕೆ ಬೆಳೆಗಳನ್ನು ಬೆಳೆಸುವುದರಿಂದ ತೆರೆದ ಮಣ್ಣು ಮುಚ್ಚಲ್ಪಟ್ಟು ಕಳೆಗಳ ಬೆಳವಣಿಗೆ ಕಡಿಮೆಯಾಗುತ್ತದೆ.',
        'high_weed_priority': 'ಹೆಚ್ಚಿನ ಪ್ರಮಾಣದ ಕಳೆಗಳು',
        'high_method_1_title': 'ತೀವ್ರ ಯಾಂತ್ರಿಕ ಕಳೆ ನಿಯಂತ್ರಣ',
        'high_method_1': 'ಬೆಳೆ ಸಾಲುಗಳ ನಡುವೆ ದಟ್ಟವಾಗಿ ಬೆಳೆದಿರುವ ಕಳೆಗಳನ್ನು ನಿಯಂತ್ರಿಸಲು ಸೂಕ್ತ ಯಾಂತ್ರಿಕ ಕಳೆ ತೆಗೆಯುವ ಸಾಧನಗಳನ್ನು ಅಥವಾ ಪುನರಾವರ್ತಿತ ಮೇಲ್ಮೈ ಕೃಷಿ ವಿಧಾನವನ್ನು ಬಳಸಿ. ಬೆಳೆಗಳ ಹತ್ತಿರ ಎಚ್ಚರಿಕೆಯಿಂದ ಕೆಲಸ ಮಾಡಿ ಮತ್ತು ಬೀಜ ಉತ್ಪತ್ತಿಯಾಗುವ ಮೊದಲು ಉಳಿದ ಕಳೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.',
        'high_method_2_title': 'ಸುರಕ್ಷಿತ ಮತ್ತು ಗುರಿಪಡಿಸಿದ ಕಳೆನಾಶಕ ಬಳಕೆ',
        'high_method_2': 'ಕೈಯಿಂದ ಮತ್ತು ಯಂತ್ರಗಳಿಂದ ಕಳೆ ನಿಯಂತ್ರಣ ಸಾಕಾಗದಿದ್ದರೆ, ಬೆಳೆಗೆ ಮತ್ತು ಗುರಿಪಡಿಸಿದ ಕಳೆಗಳಿಗೆ ಸೂಕ್ತವಾದ ಸ್ಥಳೀಯವಾಗಿ ಅನುಮೋದಿತ ಕಳೆನಾಶಕವನ್ನು ಮಾತ್ರ ಪರಿಗಣಿಸಿ. ಬೆಳೆಗಳಿಗೆ ಹಾನಿಯಾಗದಂತೆ ಲೇಬಲ್\u200cನಲ್ಲಿರುವ ಪ್ರಮಾಣ, ಬಳಕೆಯ ಸಮಯ ಮತ್ತು ಸುರಕ್ಷತಾ ಸೂಚನೆಗಳನ್ನು ಪಾಲಿಸಿ.',
        'high_method_3_title': 'ಸಮಗ್ರ ಕಳೆ ನಿರ್ವಹಣೆ',
        'high_method_3': 'ಹೆಚ್ಚಿನ ಕಳೆ ಬೆಳವಣಿಗೆಯನ್ನು ನಿಯಂತ್ರಿಸಲು ಕೈಯಿಂದ ತೆಗೆಯುವುದು, ಯಾಂತ್ರಿಕ ನಿಯಂತ್ರಣ ಮತ್ತು ಮಲ್ಚಿಂಗ್ ವಿಧಾನಗಳನ್ನು ಒಟ್ಟಿಗೆ ಬಳಸಿ. ದೊಡ್ಡ ಕಳೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ತೆರೆದ ಮಣ್ಣನ್ನು ಮುಚ್ಚಿ ಮತ್ತು ಮತ್ತೆ ಬೆಳೆಯುವ ಕಳೆಗಳನ್ನು ನಿಯಂತ್ರಿಸಲು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.',
        'important_note': 'ಪ್ರಮುಖ ಸೂಚನೆ',
        'weed_safety_note': 'ಈ ಸಲಹೆಗಳು ಅಪ್\u200cಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರದಲ್ಲಿನ ಅಂದಾಜು ಕಳೆ ಆವರಣದ ಆಧಾರದ ಮೇಲೆ ನೀಡಲಾಗಿದೆ. ಬೆಳೆ ಮತ್ತು ಹೊಲದ ಪರಿಸ್ಥಿತಿಗೆ ಅನುಗುಣವಾಗಿ ಸೂಕ್ತ ವಿಧಾನಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ. ಬೆಳೆಗಳಿಗೆ ಹಾನಿಯಾಗದಂತೆ ಯಾವಾಗಲೂ ಮುನ್ನೆಚ್ಚರಿಕೆ ವಹಿಸಿ.',
    },

    'ml': {
        'app_name': 'സ്മാർട്ട് കള തിരിച്ചറിയൽ സംവിധാനം',
        'tagline': 'കൃത്രിമ ബുദ്ധി അധിഷ്ഠിത സ്മാർട്ട് കൃഷി',
        'welcome': 'സ്വാഗതം',
        'home': 'ഹോം',
        'dashboard': 'ഡാഷ്ബോർഡ്',
        'history': 'ചരിത്രം',
        'profile': 'പ്രൊഫൈൽ',
        'logout': 'ലോഗ് ഔട്ട്',
        'login': 'ലോഗിൻ',
        'register': 'രജിസ്റ്റർ',
        'language': 'ഭാഷ',
        'select_language': 'ഭാഷ തിരഞ്ഞെടുക്കുക',
        'dark_mode': 'ഡാർക്ക് മോഡ്',
        'light_mode': 'ലൈറ്റ് മോഡ്',
        'assistant': 'AI സഹായി',
        'assistant_title': 'സ്മാർട്ട് കൃഷി സഹായി',
        'assistant_welcome': 'ഹലോ! കളകളെ തിരിച്ചറിയുന്നതിൽ എനിക്ക് നിങ്ങളെ എങ്ങനെ സഹായിക്കാനാകും?',
        'assistant_greeting': 'ഹലോ! കളകൾ, കൃഷിയിട ചിത്രങ്ങൾ, അല്ലെങ്കിൽ വിളകൾക്ക് സുരക്ഷിതമായ കള നിയന്ത്രണം എന്നിവയെക്കുറിച്ച് ചോദിക്കൂ.',
        'assistant_ready': 'തയ്യാർ',
        'ask_question': 'എന്തെങ്കിലും ചോദിക്കൂ...',
        'ask_question_long': 'ഒരു ചോദ്യം ചോദിക്കൂ...',
        'speak': 'സംസാരിക്കുക',
        'send': 'അയയ്ക്കുക',
        'close': 'സഹായി അടയ്ക്കുക',
        'open_assistant': 'AI സഹായി തുറക്കുക',
        'assistant_language': 'സഹായിയുടെ ഭാഷ',
        'auto_detect': 'സ്വയം കണ്ടെത്തുക',
        'upload_image': 'കൃഷിയിട ചിത്രം അപ്‌ലോഡ് ചെയ്യുക',
        'upload_instructions': 'വിളകളും കളകളും തിരിച്ചറിയാൻ നിങ്ങളുടെ കൃഷിയിടത്തിന്റെ വ്യക്തമായ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.',
        'supported_formats': 'പിന്തുണയ്ക്കുന്ന ഫോർമാറ്റുകൾ: JPG, JPEG, PNG, WEBP',
        'analyze': 'കൃഷിയിട ചിത്രം വിശകലനം ചെയ്യുക',
        'analyzing': 'ചിത്രം വിശകലനം ചെയ്യുന്നു...',
        'prediction_result': 'കൃഷിയിട വിശകലന ഫലങ്ങൾ',
        'weed_coverage': 'കളകളുടെ വ്യാപ്തി',
        'crop_coverage': 'വിളകളുടെ വ്യാപ്തി',
        'soil_background': 'മണ്ണ് / പശ്ചാത്തലം',
        'weed_severity': 'കളകളുടെ തീവ്രത',
        'confidence': 'വിശ്വാസ്യത',
        'weed_pixels': 'കള പിക്സലുകൾ',
        'source_image': 'യഥാർത്ഥ ചിത്രം',
        'overlay_image': 'സെഗ്മെന്റേഷൻ ഓവർലേ',
        'weed_mask': 'കള മാസ്ക്',
        'recommendation': 'ശുപാർശ',
        'recent_analyses': 'സമീപകാല വിശകലനങ്ങൾ',
        'date': 'തീയതി',
        'full_name': 'പൂർണ്ണ പേര്',
        'email': 'ഇമെയിൽ',
        'phone': 'ഫോൺ',
        'password': 'പാസ്‌വേഡ്',
        'confirm_password': 'പാസ്‌വേഡ് സ്ഥിരീകരിക്കുക',
        'create_account': 'അക്കൗണ്ട് സൃഷ്ടിക്കുക',
        'total_analyses': 'ആകെ വിശകലനങ്ങൾ',
        'average_weed': 'ശരാശരി കള വ്യാപ്തി',
        'high': 'ഉയർന്നത്',
        'medium': 'ഇടത്തരം',
        'low': 'കുറഞ്ഞത്',
        'no_analyses': 'വിശകലനങ്ങളൊന്നും കണ്ടെത്തിയില്ല',
        'farm_location': 'കൃഷിയിട സ്ഥലം',
        'save_changes': 'മാറ്റങ്ങൾ സംരക്ഷിക്കുക',
        'update_profile': 'പ്രൊഫൈൽ പുതുക്കുക',
        'guest_limit': 'സൗജന്യ പ്രവചനങ്ങൾ',
        'remaining': 'ശേഷിക്കുന്നത്',
        'used': 'ഉപയോഗിച്ചത്',
        'guest_limit_used': 'നിങ്ങളുടെ അഞ്ച് സൗജന്യ പ്രവചനങ്ങളും ഉപയോഗിച്ചു.',
        'login_to_continue': 'തുടരാൻ ലോഗിൻ ചെയ്യുകയോ രജിസ്റ്റർ ചെയ്യുകയോ ചെയ്യുക.',
        'select_field': 'കൃഷിയിടം തിരഞ്ഞെടുക്കുക',
        'existing_field': 'നിലവിലുള്ള കൃഷിയിടം തിരഞ്ഞെടുക്കുക',
        'new_field_prompt': 'അല്ലെങ്കിൽ താഴെ പുതിയ കൃഷിയിടത്തിന്റെ പേര് നൽകുക.',
        'new_field_name': 'പുതിയ കൃഷിയിടത്തിന്റെ പേര് നൽകുക',
        'field_name': 'കൃഷിയിടത്തിന്റെ പേര്',
        'field_name_placeholder': 'കൃഷിയിടത്തിന്റെ പേര് നൽകുക (ഉദാ: വടക്കൻ കൃഷിയിടം)',
        'field_name_help': 'സംരക്ഷിച്ച വിശകലനങ്ങൾ ക്രമീകരിക്കാൻ കൃഷിയിടത്തിന്റെ പേര് സഹായിക്കുന്നു.',
        'select_image': 'ഒരു ചിത്രം തിരഞ്ഞെടുക്കുക',
        'browse_image': 'കൃഷിയിട ചിത്രം തിരഞ്ഞെടുക്കാൻ ഇവിടെ ക്ലിക്ക് ചെയ്യുക',
        'image_preview': 'തിരഞ്ഞെടുത്ത ചിത്രത്തിന്റെ പ്രിവ്യൂ',
        'saved_history_note': 'നിങ്ങളുടെ വിശകലനങ്ങൾ അക്കൗണ്ട് ചരിത്രത്തിൽ സംരക്ഷിക്കും.',
        'guest_history_note': 'സംരക്ഷിച്ച വിശകലന ചരിത്രം ലഭിക്കാൻ അക്കൗണ്ട് സൃഷ്ടിക്കുക.',
        'register_here': 'ഇവിടെ രജിസ്റ്റർ ചെയ്യുക',
        'original_image': 'യഥാർത്ഥ ചിത്രം',
        'segmentation_overlay': 'സെഗ്മെന്റേഷൻ ഓവർലേ',
        'weed_mask_image': 'കള മാസ്ക്',
        'image_unavailable': 'ചിത്രം ലഭ്യമല്ല.',
        'image_failed': 'ചിത്രം ലോഡ് ചെയ്യാൻ കഴിഞ്ഞില്ല.',
        'detection_details': 'കണ്ടെത്തൽ വിശദാംശങ്ങൾ',
        'field_name_label': 'കൃഷിയിടത്തിന്റെ പേര്',
        'weed_location': 'കളയുടെ സ്ഥാനം',
        'weed_pixel_count': 'കള പിക്സൽ എണ്ണം',
        'suppression_status': 'നിയന്ത്രണ നില',
        'post_treatment_status': 'ചികിത്സയ്ക്കുശേഷമുള്ള നില',
        'conceptual_stage_status': 'ഇത് ആശയഘട്ടമാണ്: കണ്ടെത്തിയ കള പ്രദേശങ്ങളിൽ ലക്ഷ്യമിട്ട താപ നിയന്ത്രണം നടത്താം. ചത്ത കള ജൈവാവശിഷ്ടങ്ങൾ സ്വാഭാവികമായി വിഘടിക്കാൻ അതേ സ്ഥലത്ത് വിടാം.',
        'no_treatment_required': 'ചികിത്സ ആവശ്യമില്ല',
        'class_distribution': 'വിഭാഗങ്ങളുടെ വിതരണം',
        'weed': 'കള',
        'crop': 'വിള',
        'background_soil': 'പശ്ചാത്തലം / മണ്ണ്',
        'total': 'ആകെ',
        'natural_management': 'പ്രകൃതിദത്ത കള നിയന്ത്രണ ശുപാർശകൾ',
        'low_coverage': 'കുറഞ്ഞ കള വ്യാപ്തി',
        'moderate_coverage': 'മിതമായ കള വ്യാപ്തി',
        'high_coverage': 'ഉയർന്ന കള വ്യാപ്തി',
        'low_coverage_text': 'കുറച്ച് കളകൾ മാത്രമാണ് കണ്ടെത്തിയത്. ബാധിച്ച പ്രദേശം നിരീക്ഷിച്ച് ആവശ്യമായിടത്ത് ലക്ഷ്യമിട്ട നിയന്ത്രണ രീതികൾ ഉപയോഗിക്കുക.',
        'moderate_coverage_text': 'മിതമായ അളവിൽ കളകൾ കണ്ടെത്തി. സമീപ വിളകൾക്ക് കേടുപാടുകൾ വരുത്താതെ ശ്രദ്ധാപൂർവ്വം നിയന്ത്രണം ആസൂത്രണം ചെയ്യുക.',
        'high_coverage_text': 'കൂടുതൽ കളകൾ കണ്ടെത്തി. വിളകൾക്ക് കേടുപാടുകൾ വരുത്താതെ കൃഷിയിടം പരിശോധിച്ച് നിയന്ത്രണം ആസൂത്രണം ചെയ്യുക.',
        'low_priority': 'മുൻഗണന: കുറവ്',
        'medium_priority': 'മുൻഗണന: ഇടത്തരം',
        'high_priority': 'മുൻഗണന: ഉയർന്നത്',
        'treatment_note': 'ഏത് ചികിത്സയും വിളകൾക്ക് സുരക്ഷിതവും കളയുടെ തരത്തിനും കൃഷിയിട സാഹചര്യങ്ങൾക്കും അനുയോജ്യവുമായിരിക്കണം.',
        'analyze_another': 'മറ്റൊരു ചിത്രം വിശകലനം ചെയ്യുക',
        'view_history': 'വിശകലന ചരിത്രം കാണുക',
        'register_save_history': 'ചരിത്രം സംരക്ഷിക്കാൻ രജിസ്റ്റർ ചെയ്യുക',
        'invalid_format': 'JPG, JPEG, PNG അല്ലെങ്കിൽ WEBP ചിത്രം തിരഞ്ഞെടുക്കുക.',
        'file_too_large': 'ചിത്രത്തിന്റെ വലുപ്പം 8 MB കവിയരുത്.',
        'image_dimensions_large': 'ചിത്രത്തിന്റെ അളവുകൾ വളരെ വലുതാണ്. ചെറിയ ചിത്രം തിരഞ്ഞെടുക്കുക.',
        'invalid_image': 'തിരഞ്ഞെടുത്ത ഫയൽ സാധുവായ ചിത്രമല്ല.',
        'read_image_failed': 'തിരഞ്ഞെടുത്ത ചിത്രം വായിക്കാൻ കഴിഞ്ഞില്ല.',
        'field_required': 'നിലവിലുള്ള കൃഷിയിടം തിരഞ്ഞെടുക്കുകയോ പുതിയ കൃഷിയിടത്തിന്റെ പേര് നൽകുകയോ ചെയ്യുക.',
        'field_too_long': 'കൃഷിയിടത്തിന്റെ പേര് 80 അക്ഷരങ്ങളിൽ കവിയരുത്.',
        'select_image_first': 'ആദ്യം കൃഷിയിട ചിത്രം തിരഞ്ഞെടുക്കുക.',
        'invalid_image_format': 'ചിത്ര ഫോർമാറ്റ് അസാധുവാണ്. JPG, PNG അല്ലെങ്കിൽ WEBP തിരഞ്ഞെടുക്കുക.',
        'total_analyses_label': 'ആകെ വിശകലനങ്ങൾ',
        'average_weed_label': 'ശരാശരി കള വ്യാപ്തി',
        'high_priority': 'ഉയർന്ന മുൻഗണന',
        'medium_priority': 'ഇടത്തരം മുൻഗണന',
        'low_priority': 'കുറഞ്ഞ മുൻഗണന',
        'field_comparison': 'കൃഷിയിടം തിരിച്ചുള്ള താരതമ്യം',
        'field_comparison_description': 'നിങ്ങളുടെ കൃഷിയിടങ്ങളിലെ കളകളുടെയും നല്ല വിളകളുടെയും വ്യാപ്തി താരതമ്യം ചെയ്യുക.',
        'field_analyses_count': 'വിശകലനങ്ങൾ',
        'chart_aria': 'കളകളുടെയും വിളകളുടെയും വ്യാപ്തി കാണിക്കുന്ന പൈ ചാർട്ട്',
        'chart_note': 'പൈ ചാർട്ട് കളകളുടെയും വിളകളുടെയും ആപേക്ഷിക വ്യാപ്തി കാണിക്കുന്നു. ആകെ 100%.',
        'good_plants': 'നല്ല സസ്യങ്ങൾ (വിള)',
        'no_chart_data': 'കള അല്ലെങ്കിൽ വിള പിക്സലുകൾ രേഖപ്പെടുത്തിയിട്ടില്ല.',
        'no_field_comparison': 'കൃഷിയിട താരതമ്യ ഡാറ്റ ലഭ്യമല്ല. ഗ്രാഫുകൾ സൃഷ്ടിക്കാൻ കൃഷിയിട ചിത്രങ്ങൾ അപ്‌ലോഡ് ചെയ്ത് വിശകലനം ചെയ്യുക.',
        'individual_field_analysis': 'വ്യക്തിഗത കൃഷിയിട വിശകലനം',
        'individual_field_description': 'സംരക്ഷിച്ച ഓരോ കൃഷിയിടത്തിന്റെയും ശരാശരി വ്യാപ്തി വിവരങ്ങൾ. ഈ ഡാഷ്ബോർഡിൽ പ്രവചന ചിത്രങ്ങൾ കാണിക്കില്ല.',
        'unassigned_field': 'നിയോഗിക്കാത്ത കൃഷിയിടം',
        'average_good_plants': 'ശരാശരി നല്ല വിളകൾ',
        'no_field_analyses': 'കൃഷിയിട വിശകലനങ്ങളൊന്നും സംരക്ഷിച്ചിട്ടില്ല. ഇവിടെ കൃഷിയിടം തിരിച്ചുള്ള വിവരങ്ങൾ കാണാൻ ഒരു ചിത്രം വിശകലനം ചെയ്യുക.',
        'dashboard_description': 'നിങ്ങൾ സംരക്ഷിച്ച കൃഷിയിട വിശകലനങ്ങളുടെ അവലോകനം. ഓരോ കൃഷിയിടത്തിന്റെയും പൈ ചാർട്ടുകൾ ഉപയോഗിച്ച് കളകളുടെയും നല്ല വിളകളുടെയും വ്യാപ്തി താരതമ്യം ചെയ്യുക.',
        'footer_text': 'സ്മാർട്ട് കള തിരിച്ചറിയൽ സംവിധാനം | U-Net',
        'invalid_image_format_message': 'JPG, JPEG, PNG അല്ലെങ്കിൽ WEBP ചിത്രം തിരഞ്ഞെടുക്കുക.',
        'image_size_error': 'ചിത്രത്തിന്റെ വലുപ്പം 8 MB കവിയരുത്.',
        'image_dimensions_error': 'ചിത്രത്തിന്റെ അളവുകൾ വളരെ വലുതാണ്. ചെറിയ ചിത്രം തിരഞ്ഞെടുക്കുക.',
        'invalid_selected_image': 'തിരഞ്ഞെടുത്ത ഫയൽ സാധുവായ ചിത്രമല്ല.',
        'image_read_error': 'തിരഞ്ഞെടുത്ത ചിത്രം വായിക്കാൻ കഴിഞ്ഞില്ല.',
        'field_selection_error': 'നിലവിലുള്ള കൃഷിയിടം തിരഞ്ഞെടുക്കുകയോ പുതിയ കൃഷിയിടത്തിന്റെ പേര് നൽകുകയോ ചെയ്യുക.',
        'field_length_error': 'കൃഷിയിടത്തിന്റെ പേര് 80 അക്ഷരങ്ങളിൽ കവിയരുത്.',
        'no_image_selected': 'ആദ്യം കൃഷിയിട ചിത്രം തിരഞ്ഞെടുക്കുക.',
        'pie_no_data': 'കള അല്ലെങ്കിൽ വിള പിക്സലുകൾ രേഖപ്പെടുത്തിയിട്ടില്ല.',
        'recommendations': 'കള നിയന്ത്രണ നിർദ്ദേശങ്ങൾ',
        'recommendations_description': 'കണക്കാക്കിയ കളകളുടെ വിസ്തൃതിയുടെ അടിസ്ഥാനത്തിൽ ശുപാർശ ചെയ്യുന്ന നിയന്ത്രണ രീതികൾ.',
        'low_weed_priority': 'കുറഞ്ഞ തോതിലുള്ള കളകൾ',
        'low_method_1_title': 'കൈകൊണ്ട് കളകൾ നീക്കം ചെയ്യൽ',
        'low_method_1': 'മണ്ണ് അല്പം ഈർപ്പമുള്ളപ്പോൾ ചെറിയ കളകൾ കൈകൊണ്ട് നീക്കം ചെയ്യുക. ചെടിയുടെ അടിഭാഗത്ത് പിടിച്ച് പതുക്കെ വലിക്കുകയും വീണ്ടും വളരുന്നത് കുറയ്ക്കാൻ കഴിയുന്നത്ര വേരുകളോടെ നീക്കം ചെയ്യുകയും ചെയ്യുക.',
        'low_method_2_title': 'മൾച്ചിംഗ് (മണ്ണ് മൂടൽ)',
        'low_method_2': 'വിളകൾക്ക് ചുറ്റുമുള്ള തുറന്ന മണ്ണിൽ ഉണങ്ങിയ ഇലകൾ, വൈക്കോൽ അല്ലെങ്കിൽ അനുയോജ്യമായ ജൈവവസ്തുക്കൾ വിതറുക. ഇത് കളവിത്തുകളിലേക്ക് സൂര്യപ്രകാശം എത്തുന്നത് തടയുകയും പുതിയ കളകൾ വളരുന്നത് കുറയ്ക്കുകയും ചെയ്യുന്നു.',
        'low_method_3_title': 'വയൽ പതിവായി പരിശോധിക്കൽ',
        'low_method_3': 'പ്രത്യേകിച്ച് മഴയ്ക്കും ജലസേചനത്തിനും ശേഷം വയൽ പതിവായി പരിശോധിക്കുക. പുതുതായി വളരുന്ന കളകൾ തിരിച്ചറിഞ്ഞ് അവ പടരുന്നതിനോ വിത്തുണ്ടാക്കുന്നതിനോ മുമ്പ് നീക്കം ചെയ്യുക.',
        'medium_weed_priority': 'ഇടത്തരം തോതിലുള്ള കളകൾ',
        'medium_method_1_title': 'മേൽമണ്ണ് ഇളക്കലും യന്ത്രസഹായത്തോടെയുള്ള നീക്കലും',
        'medium_method_1': 'വിളവരികൾക്കിടയിലെ കളകൾ നീക്കം ചെയ്യാൻ കൈക്കൊത്തിയോ അനുയോജ്യമായ യന്ത്രക്കളനീക്കിയോ ഉപയോഗിക്കുക. വിളകളുടെ വേരുകൾക്ക് കേടുപാടുകൾ വരാതിരിക്കാൻ മണ്ണിന്റെ ഉപരിതലത്തോട് ചേർന്ന് പ്രവർത്തിക്കുകയും നീക്കം ചെയ്ത കളകൾ ശേഖരിക്കുകയും ചെയ്യുക.',
        'medium_method_2_title': 'വിഭാഗങ്ങളായി തിരിച്ചുള്ള കള നിയന്ത്രണം',
        'medium_method_2': 'കളകൾ കൂടുതലുള്ള വയലിനെ ചെറിയ ഭാഗങ്ങളായി വിഭജിച്ച് ഓരോ ഭാഗത്തെയും കളകൾ പ്രത്യേകം നിയന്ത്രിക്കുക. വിളകൾക്ക് സമീപമുള്ള കളകൾ കൈകൊണ്ട് നീക്കം ചെയ്യുകയും തുറന്ന സ്ഥലങ്ങളിൽ അനുയോജ്യമായ ഉപകരണങ്ങൾ ഉപയോഗിക്കുകയും ചെയ്യുക.',
        'medium_method_3_title': 'മൾച്ചിംഗും മൂടുവിളകളും',
        'medium_method_3': 'നിലവിലുള്ള കളകൾ നീക്കം ചെയ്തശേഷം മണ്ണിൽ ജൈവ മൾച്ച് വിതറുക. അനുയോജ്യമായ വിളചക്രങ്ങളിൽ മൂടുവിളകൾ വളർത്തുന്നത് തുറന്ന മണ്ണ് മൂടുകയും കളകളുടെ വളർച്ച കുറയ്ക്കുകയും ചെയ്യും.',
        'high_weed_priority': 'ഉയർന്ന തോതിലുള്ള കളകൾ',
        'high_method_1_title': 'തീവ്രമായ യന്ത്രവത്കൃത കള നിയന്ത്രണം',
        'high_method_1': 'വിളവരികൾക്കിടയിൽ ഇടതൂർന്ന് വളരുന്ന കളകൾ നിയന്ത്രിക്കാൻ അനുയോജ്യമായ യന്ത്രക്കളനീക്കികളോ ആവർത്തിച്ചുള്ള മേൽമണ്ണ് ഇളക്കലോ ഉപയോഗിക്കുക. വിളകൾക്ക് സമീപം ശ്രദ്ധയോടെ പ്രവർത്തിക്കുകയും വിത്തുണ്ടാകുന്നതിന് മുമ്പ് ശേഷിക്കുന്ന കളകൾ നീക്കം ചെയ്യുകയും ചെയ്യുക.',
        'high_method_2_title': 'സുരക്ഷിതവും ലക്ഷ്യബദ്ധവുമായ കളനാശിനി ഉപയോഗം',
        'high_method_2': 'കൈകൊണ്ടും യന്ത്രങ്ങൾ ഉപയോഗിച്ചും കളകൾ നിയന്ത്രിക്കുന്നത് മതിയാകാത്തപ്പോൾ, വിളയ്ക്കും ലക്ഷ്യമിടുന്ന കളകൾക്കും അനുയോജ്യമായ പ്രാദേശികമായി അംഗീകരിച്ച കളനാശിനി മാത്രം പരിഗണിക്കുക. വിളകൾക്ക് കേടുപാടുകൾ വരാതിരിക്കാൻ ഉൽപ്പന്ന ലേബലിലെ അളവ്, പ്രയോഗ സമയം, സുരക്ഷാ നിർദ്ദേശങ്ങൾ എന്നിവ പാലിക്കുക.',
        'high_method_3_title': 'സംയോജിത കള പരിപാലനം',
        'high_method_3': 'കൂടുതൽ കളവളർച്ച നിയന്ത്രിക്കാൻ കൈകൊണ്ടുള്ള നീക്കം, യന്ത്രനിയന്ത്രണം, മൾച്ചിംഗ് എന്നിവ ഒരുമിച്ച് ഉപയോഗിക്കുക. വലിയ കളകൾ നീക്കം ചെയ്യുക, തുറന്ന മണ്ണ് മൂടുക, വീണ്ടും വളരുന്ന കളകൾ നിയന്ത്രിക്കാൻ പ്രദേശം പതിവായി പരിശോധിക്കുക.',
        'important_note': 'പ്രധാന കുറിപ്പ്',
        'weed_safety_note': 'ഈ നിർദ്ദേശങ്ങൾ അപ്\u200cലോഡ് ചെയ്ത ചിത്രത്തിൽ കണക്കാക്കിയ കളകളുടെ വിസ്തൃതിയെ അടിസ്ഥാനമാക്കിയുള്ളതാണ്. വിളയ്ക്കും വയലിന്റെ സാഹചര്യങ്ങൾക്കും അനുയോജ്യമായ രീതികൾ തിരഞ്ഞെടുക്കുക. വിളകൾക്ക് കേടുപാടുകൾ സംഭവിക്കാതിരിക്കാൻ എല്ലായ്പ്പോഴും മുൻകരുതലുകൾ സ്വീകരിക്കുക.',
    },

    'hi': {
        'app_name': 'स्मार्ट खरपतवार पहचान प्रणाली',
        'tagline': 'कृत्रिम बुद्धिमत्ता आधारित स्मार्ट कृषि',
        'welcome': 'स्वागत है',
        'home': 'होम',
        'dashboard': 'डैशबोर्ड',
        'history': 'इतिहास',
        'profile': 'प्रोफ़ाइल',
        'logout': 'लॉग आउट',
        'login': 'लॉगिन',
        'register': 'पंजीकरण',
        'language': 'भाषा',
        'select_language': 'भाषा चुनें',
        'dark_mode': 'डार्क मोड',
        'light_mode': 'लाइट मोड',
        'assistant': 'AI सहायक',
        'assistant_title': 'स्मार्ट कृषि सहायक',
        'assistant_welcome': 'नमस्ते! खरपतवार की पहचान में मैं आपकी कैसे मदद कर सकता हूँ?',
        'assistant_greeting': 'नमस्ते! खरपतवार, खेत की तस्वीरों या फसलों के लिए सुरक्षित खरपतवार प्रबंधन के बारे में प्रश्न पूछें।',
        'assistant_ready': 'तैयार',
        'ask_question': 'कुछ पूछें...',
        'ask_question_long': 'एक प्रश्न पूछें...',
        'speak': 'बोलें',
        'send': 'भेजें',
        'close': 'सहायक बंद करें',
        'open_assistant': 'AI सहायक खोलें',
        'assistant_language': 'सहायक की भाषा',
        'auto_detect': 'स्वचालित पहचान',
        'upload_image': 'खेत की तस्वीर अपलोड करें',
        'upload_instructions': 'फसल और खरपतवार की पहचान के लिए अपने खेत की स्पष्ट तस्वीर अपलोड करें।',
        'supported_formats': 'समर्थित प्रारूप: JPG, JPEG, PNG, WEBP',
        'analyze': 'खेत की तस्वीर का विश्लेषण करें',
        'analyzing': 'तस्वीर का विश्लेषण हो रहा है...',
        'prediction_result': 'खेत विश्लेषण के परिणाम',
        'weed_coverage': 'खरपतवार का क्षेत्र',
        'crop_coverage': 'फसल का क्षेत्र',
        'soil_background': 'मिट्टी / पृष्ठभूमि',
        'weed_severity': 'खरपतवार की गंभीरता',
        'confidence': 'विश्वास स्तर',
        'weed_pixels': 'खरपतवार पिक्सेल',
        'source_image': 'मूल तस्वीर',
        'overlay_image': 'सेगमेंटेशन ओवरले',
        'weed_mask': 'खरपतवार मास्क',
        'recommendation': 'सुझाव',
        'recent_analyses': 'हाल के विश्लेषण',
        'date': 'तारीख',
        'full_name': 'पूरा नाम',
        'email': 'ईमेल',
        'phone': 'फोन',
        'password': 'पासवर्ड',
        'confirm_password': 'पासवर्ड की पुष्टि करें',
        'create_account': 'खाता बनाएं',
        'total_analyses': 'कुल विश्लेषण',
        'average_weed': 'औसत खरपतवार क्षेत्र',
        'high': 'उच्च',
        'medium': 'मध्यम',
        'low': 'कम',
        'no_analyses': 'कोई विश्लेषण नहीं मिला',
        'farm_location': 'खेत का स्थान',
        'save_changes': 'बदलाव सहेजें',
        'update_profile': 'प्रोफ़ाइल अपडेट करें',
        'guest_limit': 'मुफ़्त अनुमान',
        'remaining': 'शेष',
        'used': 'उपयोग किए गए',
        'guest_limit_used': 'आपके पाँच मुफ़्त अनुमान उपयोग हो चुके हैं।',
        'login_to_continue': 'सिस्टम का उपयोग जारी रखने के लिए लॉगिन करें या पंजीकरण करें।',
        'select_field': 'कृषि क्षेत्र चुनें',
        'existing_field': 'मौजूदा खेत चुनें',
        'new_field_prompt': 'या नीचे नए खेत का नाम दर्ज करें।',
        'new_field_name': 'नए खेत का नाम दर्ज करें',
        'field_name': 'कृषि क्षेत्र का नाम',
        'field_name_placeholder': 'खेत का नाम दर्ज करें (जैसे: उत्तरी खेत)',
        'field_name_help': 'खेत का नाम आपके सहेजे गए विश्लेषणों को व्यवस्थित करने में मदद करता है।',
        'select_image': 'तस्वीर चुनें',
        'browse_image': 'खेत की तस्वीर चुनने के लिए यहाँ क्लिक करें',
        'image_preview': 'चुनी गई तस्वीर का पूर्वावलोकन',
        'saved_history_note': 'आपके विश्लेषण आपके खाते के इतिहास में सहेजे जाते हैं।',
        'guest_history_note': 'अपना सहेजा गया विश्लेषण इतिहास देखने के लिए खाता बनाएं।',
        'register_here': 'यहाँ पंजीकरण करें',
        'original_image': 'मूल तस्वीर',
        'segmentation_overlay': 'सेगमेंटेशन ओवरले',
        'weed_mask_image': 'खरपतवार मास्क',
        'image_unavailable': 'तस्वीर उपलब्ध नहीं है।',
        'image_failed': 'तस्वीर लोड नहीं हो सकी।',
        'detection_details': 'पहचान का विवरण',
        'field_name_label': 'खेत का नाम',
        'weed_location': 'खरपतवार का स्थान',
        'weed_pixel_count': 'खरपतवार पिक्सेल की संख्या',
        'suppression_status': 'नियंत्रण की स्थिति',
        'post_treatment_status': 'उपचार के बाद की स्थिति',
        'conceptual_stage_status': 'यह अवधारणा चरण है: पहचाने गए खरपतवार क्षेत्रों पर लक्षित तापीय नियंत्रण किया जा सकता है। मृत खरपतवार जैव-द्रव्य को प्राकृतिक रूप से विघटित होने के लिए उसी स्थान पर छोड़ा जा सकता है।',
        'no_treatment_required': 'उपचार की आवश्यकता नहीं',
        'class_distribution': 'वर्ग वितरण',
        'weed': 'खरपतवार',
        'crop': 'फसल',
        'background_soil': 'पृष्ठभूमि / मिट्टी',
        'total': 'कुल',
        'natural_management': 'प्राकृतिक खरपतवार प्रबंधन सुझाव',
        'low_coverage': 'कम खरपतवार क्षेत्र',
        'moderate_coverage': 'मध्यम खरपतवार क्षेत्र',
        'high_coverage': 'अधिक खरपतवार क्षेत्र',
        'low_coverage_text': 'कम मात्रा में खरपतवार पाया गया है। प्रभावित क्षेत्र पर नज़र रखें और आवश्यकता होने पर लक्षित नियंत्रण विधियों का उपयोग करें।',
        'moderate_coverage_text': 'मध्यम मात्रा में खरपतवार पाया गया है। आसपास की फसलों को सुरक्षित रखते हुए सावधानीपूर्वक प्रबंधन की योजना बनाएं।',
        'high_coverage_text': 'अधिक मात्रा में खरपतवार पाया गया है। फसलों को नुकसान से बचाने के लिए खेत का निरीक्षण करें और सावधानी से प्रबंधन की योजना बनाएं।',
        'low_priority': 'प्राथमिकता: कम',
        'medium_priority': 'प्राथमिकता: मध्यम',
        'high_priority': 'प्राथमिकता: उच्च',
        'treatment_note': 'कोई भी उपचार फसल के लिए सुरक्षित और खरपतवार के प्रकार तथा खेत की परिस्थितियों के अनुसार होना चाहिए।',
        'analyze_another': 'दूसरी तस्वीर का विश्लेषण करें',
        'view_history': 'विश्लेषण इतिहास देखें',
        'register_save_history': 'इतिहास सहेजने के लिए पंजीकरण करें',
        'invalid_format': 'कृपया JPG, JPEG, PNG या WEBP तस्वीर चुनें।',
        'file_too_large': 'तस्वीर का आकार 8 MB से अधिक नहीं होना चाहिए।',
        'image_dimensions_large': 'तस्वीर के आयाम बहुत बड़े हैं। कृपया छोटी तस्वीर चुनें।',
        'invalid_image': 'चुनी गई फ़ाइल मान्य तस्वीर नहीं है।',
        'read_image_failed': 'चुनी गई तस्वीर पढ़ी नहीं जा सकी।',
        'field_required': 'मौजूदा खेत चुनें या नए कृषि क्षेत्र का नाम दर्ज करें।',
        'field_too_long': 'खेत का नाम 80 अक्षरों से अधिक नहीं होना चाहिए।',
        'select_image_first': 'कृपया पहले खेत की तस्वीर चुनें।',
        'invalid_image_format': 'अमान्य तस्वीर प्रारूप। JPG, PNG या WEBP चुनें।',
        'total_analyses_label': 'कुल विश्लेषण',
        'average_weed_label': 'औसत खरपतवार क्षेत्र',
        'high_priority': 'उच्च प्राथमिकता',
        'medium_priority': 'मध्यम प्राथमिकता',
        'low_priority': 'कम प्राथमिकता',
        'field_comparison': 'खेत के अनुसार तुलना',
        'field_comparison_description': 'अपने कृषि क्षेत्रों में खरपतवार और अच्छी फसलों के क्षेत्र की तुलना करें।',
        'field_analyses_count': 'विश्लेषण',
        'chart_aria': 'खरपतवार और फसल क्षेत्र दिखाने वाला पाई चार्ट',
        'chart_note': 'पाई चार्ट खरपतवार और फसल के सापेक्ष क्षेत्र को दिखाता है। कुल 100%.',
        'good_plants': 'अच्छे पौधे (फसल)',
        'no_chart_data': 'खरपतवार या फसल पिक्सेल दर्ज नहीं किए गए।',
        'no_field_comparison': 'अभी खेत की तुलना का डेटा उपलब्ध नहीं है। ग्राफ़ बनाने के लिए खेत की तस्वीरें अपलोड करके उनका विश्लेषण करें।',
        'individual_field_analysis': 'व्यक्तिगत खेत विश्लेषण',
        'individual_field_description': 'प्रत्येक सहेजे गए खेत के औसत क्षेत्र के आँकड़े। इस डैशबोर्ड पर अनुमान की तस्वीरें नहीं दिखाई जाती हैं।',
        'unassigned_field': 'बिना नाम वाला खेत',
        'average_good_plants': 'औसत अच्छी फसल',
        'no_field_analyses': 'अभी तक कोई खेत विश्लेषण सहेजा नहीं गया है। यहाँ खेत के अनुसार आँकड़े देखने के लिए खेत की तस्वीर का विश्लेषण करें।',
        'dashboard_description': 'आपके सहेजे गए खेत विश्लेषणों का विवरण। प्रत्येक खेत के पाई चार्ट से खरपतवार और अच्छी फसल के क्षेत्र की तुलना करें।',
        'footer_text': 'स्मार्ट खरपतवार पहचान प्रणाली | U-Net',
        'invalid_image_format_message': 'कृपया JPG, JPEG, PNG या WEBP तस्वीर चुनें।',
        'image_size_error': 'तस्वीर का आकार 8 MB से अधिक नहीं होना चाहिए।',
        'image_dimensions_error': 'तस्वीर के आयाम बहुत बड़े हैं। कृपया छोटी तस्वीर चुनें।',
        'invalid_selected_image': 'चुनी गई फ़ाइल मान्य तस्वीर नहीं है।',
        'image_read_error': 'चुनी गई तस्वीर पढ़ी नहीं जा सकी।',
        'field_selection_error': 'मौजूदा खेत चुनें या नए कृषि क्षेत्र का नाम दर्ज करें।',
        'field_length_error': 'खेत का नाम 80 अक्षरों से अधिक नहीं होना चाहिए।',
        'no_image_selected': 'कृपया पहले खेत की तस्वीर चुनें।',
        'pie_no_data': 'खरपतवार या फसल पिक्सेल दर्ज नहीं किए गए।',
        'recommendations': 'खरपतवार नियंत्रण के सुझाव',
        'recommendations_description': 'अनुमानित खरपतवार क्षेत्र के आधार पर सुझाए गए नियंत्रण तरीके।',
        'low_weed_priority': 'कम खरपतवार स्तर',
        'low_method_1_title': 'हाथ से खरपतवार हटाना',
        'low_method_1': 'मिट्टी थोड़ी नम होने पर छोटे खरपतवारों को हाथ से हटाएँ। उन्हें आधार के पास से धीरे-धीरे खींचें और दोबारा उगने की संभावना कम करने के लिए जहाँ तक संभव हो जड़ों सहित निकालें।',
        'low_method_2_title': 'मल्चिंग (मिट्टी को ढकना)',
        'low_method_2': 'फसलों के आसपास की खुली मिट्टी पर सूखी पत्तियाँ, भूसा या अन्य उपयुक्त जैविक सामग्री फैलाएँ। इससे खरपतवार के बीजों तक सूर्य का प्रकाश नहीं पहुँचता और नए खरपतवारों की वृद्धि कम होती है।',
        'low_method_3_title': 'खेत का नियमित निरीक्षण',
        'low_method_3': 'विशेष रूप से बारिश या सिंचाई के बाद खेत का नियमित निरीक्षण करें। नए उगने वाले खरपतवारों की पहचान करें और उनके फैलने या बीज बनने से पहले उन्हें हटा दें।',
        'medium_weed_priority': 'मध्यम खरपतवार स्तर',
        'medium_method_1_title': 'हल्की गुड़ाई और मशीन द्वारा खरपतवार हटाना',
        'medium_method_1': 'फसल की कतारों के बीच खरपतवार हटाने के लिए हाथ की कुदाल या उपयुक्त यांत्रिक खरपतवार हटाने वाले उपकरण का उपयोग करें। फसलों की जड़ों को नुकसान से बचाने के लिए मिट्टी की ऊपरी सतह के पास काम करें और हटाए गए खरपतवारों को इकट्ठा करें।',
        'medium_method_2_title': 'खेत के अलग-अलग हिस्सों में खरपतवार नियंत्रण',
        'medium_method_2': 'प्रभावित खेत को छोटे-छोटे हिस्सों में बाँटें और एक समय में एक हिस्से के खरपतवार हटाएँ। फसलों के पास के खरपतवार हाथ से निकालें और खुले स्थानों को साफ करने के लिए उपयुक्त उपकरणों का उपयोग करें।',
        'medium_method_3_title': 'मल्चिंग और आवरण फसलें',
        'medium_method_3': 'मौजूदा खरपतवार हटाने के बाद मिट्टी पर जैविक मल्च फैलाएँ, जिससे आगे खरपतवार उगने की संभावना कम हो। उपयुक्त फसल चक्र के दौरान आवरण फसलें उगाएँ, ताकि खुली मिट्टी ढकी रहे और खरपतवारों की वृद्धि कम हो।',
        'high_weed_priority': 'उच्च खरपतवार स्तर',
        'high_method_1_title': 'व्यापक यांत्रिक खरपतवार नियंत्रण',
        'high_method_1': 'फसल की कतारों के बीच घने खरपतवारों को नियंत्रित करने के लिए उपयुक्त यांत्रिक उपकरणों या बार-बार हल्की गुड़ाई का उपयोग करें। फसलों के आसपास सावधानी से काम करें और बीज बनने से पहले बचे हुए खरपतवारों को हटा दें।',
        'high_method_2_title': 'सुरक्षित और लक्षित खरपतवारनाशक का उपयोग',
        'high_method_2': 'जब हाथ से और मशीनों द्वारा खरपतवार नियंत्रण पर्याप्त न हो, तब फसल और लक्षित खरपतवारों के लिए उपयुक्त स्थानीय रूप से स्वीकृत खरपतवारनाशक पर ही विचार करें। फसल को नुकसान से बचाने के लिए उत्पाद के लेबल पर दिए गए निर्देशों, मात्रा, उपयोग के समय और सुरक्षा सावधानियों का पालन करें।',
        'high_method_3_title': 'एकीकृत खरपतवार प्रबंधन',
        'high_method_3': 'अधिक खरपतवारों को नियंत्रित करने के लिए हाथ से हटाने, यांत्रिक नियंत्रण और मल्चिंग विधियों को एक साथ अपनाएँ। बड़े खरपतवार हटाएँ, खुली मिट्टी को ढकें और दोबारा उगने वाले खरपतवारों को नियंत्रित करने के लिए नियमित निरीक्षण करें।',
        'important_note': 'महत्वपूर्ण सूचना',
        'weed_safety_note': 'ये सुझाव अपलोड की गई छवि में अनुमानित खरपतवार क्षेत्र के आधार पर दिए गए हैं। फसल और खेत की परिस्थितियों के अनुसार उपयुक्त तरीके चुनें। फसलों को नुकसान से बचाने के लिए हमेशा सावधानी बरतें.',
    }
}

def t(key):
    language = session.get('language', 'en')
    if language not in TRANSLATIONS:
        language = 'en'
    return TRANSLATIONS[language].get(key, TRANSLATIONS['en'].get(key, str(key).replace('_', ' ').title()))

@app.context_processor
def inject_language_settings():
    language = session.get('language', 'en')
    if language not in SUPPORTED_LANGUAGES:
        language = 'en'
    return {'current_language': language, 'supported_languages': SUPPORTED_LANGUAGES, 't': t}

@app.route('/set-language/<language>', methods=['GET', 'POST'])
def set_language(language):
    """Save language choice and return to the requested local page."""
    language = language.strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        language = 'en'
    session['language'] = language
    next_page = request.values.get('next', '')
    # Only allow a local relative path. Do not clear the session: it contains
    # the previous prediction and guest prediction counter.
    if next_page.startswith('/') and not next_page.startswith('//') and '\\' not in next_page:
        return redirect(next_page)
    if session.get('customer_id'):
        return redirect(url_for('home'))
    return redirect(url_for('login'))

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):
        if 'customer_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return function(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_client_ip():
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or 'Unknown'

def get_current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def render_home_error(message, status_code=400):
    """Render prediction-page errors for both visitors and logged-in users."""
    flash(message, 'danger')
    customer_id = session.get('customer_id')
    if customer_id:
        conn = get_db()
        customer = conn.execute('SELECT * FROM customers WHERE id = ?', (customer_id,)).fetchone()
        conn.close()
        if customer is None:
            session.pop('customer_id', None)
            session.pop('customer_name', None)
            session.pop('customer_email', None)
            customer_id = None
    else:
        customer = None
    used = min(int(session.get('free_predictions_used', 0)), FREE_PREDICTION_LIMIT)
    return (render_template('index.html', customer=customer, result=None, field_name=request.form.get('field_name', '').strip(), saved_fields=get_saved_fields(customer_id) if customer_id else [], is_guest=customer_id is None, free_predictions_used=used, free_predictions_remaining=max(0, FREE_PREDICTION_LIMIT - used)), status_code)
_interpreter = None
_input_details = None
_output_details = None
PROCESSING_DEVICE = "CPU"


def load_model():
    """Load the compact TensorFlow Lite model once, only when a prediction is requested."""
    global _interpreter, _input_details, _output_details, PROCESSING_DEVICE

    if _interpreter is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"TensorFlow Lite model not found: {MODEL_PATH}"
            )

        app.logger.info("PREDICTION: LiteRT interpreter import starting")
        import_started = time.perf_counter()

        # LiteRT is substantially lighter than importing the full TensorFlow
        # runtime. Install the ai-edge-litert package in the deployment.
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "LiteRT is not installed. Install it with: "
                "pip install ai-edge-litert"
            ) from exc

        app.logger.info(
            "PREDICTION: LiteRT import completed in %.2f seconds",
            time.perf_counter() - import_started
        )

        load_started = time.perf_counter()
        app.logger.info("PREDICTION: TFLite interpreter loading started")
        _interpreter = Interpreter(
            model_path=str(MODEL_PATH),
            num_threads=int(os.getenv("TFLITE_NUM_THREADS", "1"))
        )
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()

        if len(_input_details) != 1 or len(_output_details) != 1:
            raise RuntimeError(
                "Expected one model input and one model output."
            )

        expected_input = tuple(int(v) for v in _input_details[0]["shape"])
        expected_output = tuple(int(v) for v in _output_details[0]["shape"])

        if expected_input != (1, 256, 256, 3):
            raise RuntimeError(
                f"Unexpected TFLite input shape: {expected_input}"
            )
        if expected_output != (1, 256, 256, 3):
            raise RuntimeError(
                f"Unexpected TFLite output shape: {expected_output}"
            )

        PROCESSING_DEVICE = "CPU"
        app.logger.info(
            "PREDICTION: TFLite interpreter ready in %.2f seconds; "
            "input=%s output=%s device=%s",
            time.perf_counter() - load_started,
            expected_input,
            expected_output,
            PROCESSING_DEVICE
        )

    return _interpreter


def run_model(input_array):
    """Run one image through the cached TFLite interpreter."""
    interpreter = load_model()
    input_info = _input_details[0]
    output_info = _output_details[0]

    expected_dtype = input_info["dtype"]
    if input_array.dtype != expected_dtype:
        input_array = input_array.astype(expected_dtype, copy=False)

    interpreter.set_tensor(input_info["index"], input_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_info["index"])


def preprocess_image(image):
    image = image.convert('RGB')
    original_width, original_height = image.size
    resized = image.resize(MODEL_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    array = array / 255.0
    array = np.expand_dims(array, axis=0)
    return (image, array, original_width, original_height)

def is_likely_field_image(image, class_mask):
    """
    Reject obviously unrelated images using a conservative heuristic.
    A U-Net segmentation model is not itself a trained field/non-field classifier,
    so this combines its class predictions with field-like green/soil colors.
    """
    # Sample at no more than 512 x 512. The original image can be up to
    # 16 megapixels; converting all of it to HSV creates several large arrays
    # and can exhaust memory on a small Render instance.
    sample_image = image.convert('RGB')
    sample_image.thumbnail((512, 512), Image.Resampling.BILINEAR)
    hsv = np.asarray(sample_image.convert('HSV'), dtype=np.uint8)
    hue = hsv[:, :, 0].astype(np.float32) * (360.0 / 255.0)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    green = (hue >= 32) & (hue <= 155) & (saturation >= 35) & (value >= 35)
    earthy = (hue >= 5) & (hue <= 42) & (saturation >= 30) & (value >= 25)
    green_ratio = float(np.mean(green))
    earth_ratio = float(np.mean(earthy))
    total = max(1, int(class_mask.size))
    crop_ratio = float(np.count_nonzero(class_mask == 1)) / total
    weed_ratio = float(np.count_nonzero(class_mask == 2)) / total
    soil_ratio = float(np.count_nonzero(class_mask == 0)) / total
    vegetation_ratio = crop_ratio + weed_ratio
    if vegetation_ratio < 0.001 and green_ratio < 0.012 and (not (earth_ratio >= 0.18 and soil_ratio >= 0.25)):
        return False
    if vegetation_ratio < 0.0005 and green_ratio < 0.006 and (earth_ratio < 0.06):
        return False
    return True

def get_display_size(width, height, max_side=1280):
    """Limit generated preview images to reduce resize and PNG-save time."""
    longest_side = max(width, height)
    if longest_side <= max_side:
        return (width, height)
    scale = max_side / float(longest_side)
    return (max(1, int(width * scale)), max(1, int(height * scale)))

def normalize_prediction(prediction):
    prediction = np.asarray(prediction)
    if prediction.ndim == 4:
        prediction = prediction[0]
    if prediction.ndim != 3:
        raise ValueError(f'Unexpected U-Net output shape: {prediction.shape}')
    if prediction.min() >= 0 and prediction.max() <= 1.0:
        sums = prediction.sum(axis=-1)
        if np.allclose(sums, 1.0, atol=0.05):
            probabilities = prediction
        else:
            exp_values = np.exp(prediction - np.max(prediction, axis=-1, keepdims=True))
            probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    else:
        exp_values = np.exp(prediction - np.max(prediction, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    return probabilities

def calculate_severity(weed_percentage):
    if weed_percentage < LOW_WEED_PERCENT:
        return 'Low'
    elif weed_percentage < MEDIUM_WEED_PERCENT:
        return 'Medium'
    return 'High'

def calculate_location(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 'No weed detected'
    center_x = np.mean(xs)
    center_y = np.mean(ys)
    height, width = mask.shape
    if center_y < height / 3:
        vertical = 'Top'
    elif center_y < 2 * height / 3:
        vertical = 'Center'
    else:
        vertical = 'Bottom'
    if center_x < width / 3:
        horizontal = 'Left'
    elif center_x < 2 * width / 3:
        horizontal = 'Center'
    else:
        horizontal = 'Right'
    return f'{vertical}-{horizontal}'

def calculate_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 'No weed detected'
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return f'{x1},{y1},{x2},{y2}'

def create_target_mask(mask, original_size):
    width, height = original_size
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
    return mask_image

def create_color_mask(class_mask, original_size):
    """Create a transparent RGBA segmentation layer: green=crop, red=weed."""
    width, height = original_size
    rgba = np.zeros((class_mask.shape[0], class_mask.shape[1], 4), dtype=np.uint8)
    rgba[class_mask == 1] = [0, 255, 70, 180]
    rgba[class_mask == 2] = [255, 25, 25, 210]
    image = Image.fromarray(rgba, mode='RGBA')
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    return image

def create_overlay(original_image, color_mask):
    """Composite crop/weed colors over the source without blacking the image."""
    original = original_image.convert('RGBA')
    mask = color_mask.convert('RGBA')
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.NEAREST)
    return Image.alpha_composite(original, mask).convert('RGB')

def calculate_confidence(probabilities, class_mask):
    confidence_map = np.max(probabilities, axis=-1)
    if confidence_map.size == 0:
        return 0.0
    return float(np.mean(confidence_map) * 100)

def get_suppression_recommendation(weed_percentage):
    if weed_percentage <= 0:
        return {'status': 'No treatment required', 'recommendation': 'No weed region was detected.'}
    if weed_percentage < LOW_WEED_PERCENT:
        return {'status': 'Low weed coverage', 'recommendation': 'Targeted thermal suppression may be considered for the detected weed regions.'}
    if weed_percentage < MEDIUM_WEED_PERCENT:
        return {'status': 'Medium weed coverage', 'recommendation': 'Targeted thermal suppression is recommended for the detected weed regions, subject to crop-safety validation.'}
    return {'status': 'High weed coverage', 'recommendation': 'High weed coverage detected. Targeted thermal suppression should be planned carefully with crop-safety validation.'}

def get_suppression_status_key(status):
    """Return a stable translation key for a suppression status."""
    status_keys = {
        'No treatment required': 'no_treatment_required',
        'Low weed coverage': 'low_coverage',
        'Medium weed coverage': 'moderate_coverage',
        'High weed coverage': 'high_coverage',
    }
    return status_keys.get(status, 'no_treatment_required')

def get_post_treatment_status():
    return 'Conceptual stage: detected weed regions can be targeted for thermal suppression. Dead weed biomass can remain in place for natural decomposition.'

def get_workflow_status():
    return [{'step': 'Image uploaded', 'status': 'Completed'}, {'step': 'Image preprocessing', 'status': 'Completed'}, {'step': 'U-Net segmentation', 'status': 'Completed'}, {'step': 'Weed region identification', 'status': 'Completed'}, {'step': 'Targeted suppression', 'status': 'Proposed'}, {'step': 'Natural decomposition', 'status': 'Proposed'}]

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    request_started = time.perf_counter()
    app.logger.info(
        'PREDICTION: Request started; method=%s path=%s',
        request.method,
        request.path
    )
    customer_id = session.get('customer_id')
    customer = None
    if customer_id:
        conn = get_db()
        customer = conn.execute('SELECT * FROM customers WHERE id = ?', (customer_id,)).fetchone()
        conn.close()
        if customer is None:
            session.pop('customer_id', None)
            session.pop('customer_name', None)
            session.pop('customer_email', None)
            customer_id = None
    used = min(int(session.get('free_predictions_used', 0)), FREE_PREDICTION_LIMIT)
    if request.method == 'GET':
        result = session.get('last_prediction')
        if result:
            # Do not show stale/broken result cards if any generated image was
            # removed from statics/uploads.
            image_urls = (result.get('source_url'), result.get('overlay_url'), result.get('weed_mask_url'))
            image_names = [str(url or '').rsplit('/', 1)[-1] for url in image_urls]
            if not all(name and (UPLOAD_FOLDER / name).is_file() for name in image_names):
                session.pop('last_prediction', None)
                result = None
        return render_template('index.html', customer=customer, result=result, is_guest=customer_id is None, free_predictions_used=used, free_predictions_remaining=max(0, FREE_PREDICTION_LIMIT - used), saved_fields=get_saved_fields(customer_id) if customer_id else [])
    if customer_id is None and used >= FREE_PREDICTION_LIMIT:
        flash('You have used all 5 free predictions. Please login or register to continue.', 'warning')
        return redirect(url_for('login'))
    field_name = normalize_field_name(request.form.get('field_name', ''))
    if not field_name:
        return render_home_error('Please select or enter a field name before analyzing.', 400)
    if len(field_name) > 80:
        return render_home_error('Field name must be 80 characters or fewer.', 400)
    if customer_id is not None:
        field_conn = get_db()
        try:
            field_name = save_or_get_field(field_conn, customer_id, field_name)
            field_conn.commit()
        finally:
            field_conn.close()
    if 'image' not in request.files:
        return render_home_error('Please select an image.', 400)
    file = request.files['image']
    if file.filename == '':
        return render_home_error('No image selected.', 400)
    if not allowed_file(file.filename):
        return render_home_error('Allowed image formats: JPG, JPEG, PNG, WEBP.', 400)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_FILE_SIZE:
        return render_home_error('Image file is too large. Maximum size is 8 MB.', 400)
    original_filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    extension = Path(original_filename).suffix.lower()
    base_name = Path(original_filename).stem + '_' + timestamp
    source_filename = base_name + extension
    source_path = UPLOAD_FOLDER / source_filename
    app.logger.info(
        'PREDICTION: Saving upload; filename=%s size_bytes=%s',
        original_filename,
        file_size
    )
    file.save(source_path)
    app.logger.info('PREDICTION: Upload saved')

    try:
        image_started = time.perf_counter()
        app.logger.info('PREDICTION: Opening and validating image')
        with Image.open(source_path) as opened_image:
            opened_image.load()
            image_width, image_height = opened_image.size
            app.logger.info(
                'PREDICTION: Image opened; width=%s height=%s',
                image_width,
                image_height
            )
            if image_width * image_height > MAX_IMAGE_PIXELS:
                source_path.unlink(missing_ok=True)
                return render_home_error('Image contains too many pixels.', 400)
            image = opened_image.convert('RGB')

        preprocess_started = time.perf_counter()
        original_image, input_array, original_width, original_height = preprocess_image(image)
        app.logger.info(
            'PREDICTION: Image preprocessing completed in %.2f seconds; model_input_shape=%s',
            time.perf_counter() - preprocess_started,
            input_array.shape
        )
        app.logger.info(
            'PREDICTION: Image validation and preprocessing total %.2f seconds',
            time.perf_counter() - image_started
        )
    except Exception as e:
        app.logger.exception('PREDICTION: Image opening or preprocessing failed')
        source_path.unlink(missing_ok=True)
        return render_home_error(f'Invalid image: {str(e)}', 400)

    try:
        app.logger.info('PREDICTION: Requesting TFLite U-Net interpreter')
        load_model()
        app.logger.info('PREDICTION: TFLite interpreter is ready; starting inference')
        inference_started = time.perf_counter()
        prediction = run_model(input_array)
        app.logger.info(
            'PREDICTION: Raw TFLite output received; shape=%s',
            np.shape(prediction)
        )
        probabilities = normalize_prediction(prediction)
        app.logger.info(
            'PREDICTION: TFLite inference and normalization completed in %.2f seconds',
            time.perf_counter() - inference_started
        )
    except Exception:
        app.logger.exception('PREDICTION: U-Net prediction failed')
        source_path.unlink(missing_ok=True)
        return render_home_error('The U-Net model could not process this image. Check that the model file exists and is compatible.', 500)

    if probabilities.shape[-1] != 3:
        app.logger.error('PREDICTION: Expected 3 output classes; received shape=%s', probabilities.shape)
        source_path.unlink(missing_ok=True)
        return render_home_error('The loaded U-Net model must return exactly three classes: Background/Soil, Crop, and Weed.', 500)

    class_mask = np.argmax(probabilities, axis=-1).astype(np.uint8)
    validation_started = time.perf_counter()
    app.logger.info('PREDICTION: Checking whether uploaded image is an agricultural field image')
    is_field_image = class_mask.ndim == 2 and is_likely_field_image(original_image, class_mask)
    app.logger.info(
        'PREDICTION: Field-image validation completed in %.2f seconds; accepted=%s',
        time.perf_counter() - validation_started,
        is_field_image
    )
    if not is_field_image:
        source_path.unlink(missing_ok=True)
        return render_home_error('Invalid image. Please upload a clear agricultural field or crop image. Portraits, documents, objects, and unrelated photos cannot be analyzed.', 400)
    weed_mask = class_mask == WEED_CLASS_INDEX
    weed_pixel_count = int(np.sum(weed_mask))
    total_pixels = int(weed_mask.size)
    weed_percentage = weed_pixel_count / total_pixels * 100
    weed_percentage = round(float(weed_percentage), 2)
    soil_pixel_count = int(np.sum(class_mask == 0))
    crop_pixel_count = int(np.sum(class_mask == 1))
    weed_pixel_count = int(np.sum(class_mask == WEED_CLASS_INDEX))
    soil_percentage = round(soil_pixel_count / total_pixels * 100.0, 2)
    crop_percentage = round(crop_pixel_count / total_pixels * 100.0, 2)
    weed_percentage = round(100.0 - soil_percentage - crop_percentage, 2)
    severity = calculate_severity(weed_percentage)
    location = calculate_location(weed_mask)
    bbox = calculate_bbox(weed_mask)
    confidence = calculate_confidence(probabilities, class_mask)
    confidence = round(confidence, 2)
    suppression = get_suppression_recommendation(weed_percentage)
    post_treatment_status = get_post_treatment_status()
    workflow_status = get_workflow_status()
    display_size = get_display_size(original_width, original_height, max_side=1280)
    display_image = original_image
    if display_image.size != display_size:
        display_image = display_image.resize(display_size, Image.Resampling.LANCZOS)
    output_started = time.perf_counter()
    weed_mask_image = create_target_mask(weed_mask, display_size)
    weed_mask_filename = base_name + '_weed_mask.png'
    weed_mask_path = UPLOAD_FOLDER / weed_mask_filename
    weed_mask_image.save(weed_mask_path)
    color_mask = create_color_mask(class_mask, display_size)
    overlay = create_overlay(display_image, color_mask)
    overlay_filename = base_name + '_overlay.png'
    overlay_path = UPLOAD_FOLDER / overlay_filename
    overlay.save(overlay_path, optimize=False)
    required_output_files = (source_path, weed_mask_path, overlay_path)
    missing_outputs = [str(path) for path in required_output_files if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        app.logger.error('Prediction image output missing: %s', missing_outputs)
        return render_home_error('The prediction was completed, but one or more result images could not be saved. Please check the statics/uploads folder permissions and try again.', 500)
    app.logger.info('Mask and overlay generation completed in %.2f seconds (display size: %s)', time.perf_counter() - output_started, display_size)
    if customer_id is None:
        used = min(int(session.get('free_predictions_used', 0)), FREE_PREDICTION_LIMIT)
        if used >= FREE_PREDICTION_LIMIT:
            flash('You have used all 5 free predictions. Please login or register to continue.', 'warning')
            return redirect(url_for('login'))
        used += 1
        session['free_predictions_used'] = used
        session.modified = True
        if used == FREE_PREDICTION_LIMIT:
            flash('You have used all 5 free predictions. Please login or register before your next prediction.', 'warning')
    else:
        used = 0
    analysis_id = None
    if customer_id is not None:
        conn = get_db()
        field_name = save_or_get_field(conn, customer_id, field_name)
        cursor = conn.execute('\n            INSERT INTO analyses (\n                customer_id,\n                field_name,\n                crop_percentage,\n                soil_percentage,\n                original_filename,\n                weed_percentage,\n                weed_severity,\n                weed_location,\n                weed_pixel_count,\n                confidence,\n                weed_bbox,\n                source_filename,\n                overlay_filename,\n                weed_mask_filename,\n                suppression_status,\n                suppression_recommendation,\n                post_treatment_status,\n                workflow_status,\n                created_at\n            )\n            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n        ', (customer_id, field_name, crop_percentage, soil_percentage, original_filename, weed_percentage, severity, location, weed_pixel_count, confidence, bbox, source_filename, overlay_filename, weed_mask_filename, suppression['status'], suppression['recommendation'], post_treatment_status, 'Segmentation completed', get_current_time()))
        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        app.logger.info('PREDICTION: Analysis record saved; analysis_id=%s', analysis_id)
    classes = {'Background/Soil': soil_percentage, 'Crop': crop_percentage, 'Weed': weed_percentage}
    result = {'success': True, 'analysis_id': analysis_id, 'field_name': field_name, 'crop_percentage': crop_percentage, 'soil_percentage': soil_percentage, 'device': PROCESSING_DEVICE, 'processing_device': PROCESSING_DEVICE, 'source_url': url_for('static', filename=f'uploads/{source_filename}'), 'original_image_url': url_for('static', filename=f'uploads/{source_filename}'), 'original_image': url_for('static', filename=f'uploads/{source_filename}'), 'overlay_url': url_for('static', filename=f'uploads/{overlay_filename}'), 'overlay_image_url': url_for('static', filename=f'uploads/{overlay_filename}'), 'segmentation_overlay_url': url_for('static', filename=f'uploads/{overlay_filename}'), 'weed_mask_url': url_for('static', filename=f'uploads/{weed_mask_filename}'), 'weed_mask_image_url': url_for('static', filename=f'uploads/{weed_mask_filename}'), 'filename': original_filename, 'weed_percentage': weed_percentage, 'weed_pixel_count': weed_pixel_count, 'weed_severity': severity, 'weed_location': location, 'weed_bbox': bbox, 'confidence': confidence, 'classes': classes, 'suppression_status': suppression['status'], 'suppression_status_key': get_suppression_status_key(suppression['status']), 'suppression_recommendation': suppression['recommendation'], 'post_treatment_status': post_treatment_status, 'workflow_status': workflow_status}
    # Keep only the fields needed by index.html. This lets a language change
    # re-render the same prediction without rerunning the model.
    session['last_prediction'] = {
        key: result.get(key) for key in (
            'field_name', 'crop_percentage', 'soil_percentage', 'source_url',
            'overlay_url', 'weed_mask_url', 'weed_percentage', 'weed_severity',
            'weed_location', 'weed_pixel_count', 'confidence',
            'suppression_status_key'
        )
    }
    session.modified = True

    if customer_id is not None:
        conn = get_db()
        customer = conn.execute('SELECT * FROM customers WHERE id = ?', (customer_id,)).fetchone()
        conn.close()
        if customer is None:
            session.pop('customer_id', None)
            session.pop('customer_name', None)
            session.pop('customer_email', None)
            customer_id = None
            customer = None
    used = min(int(session.get('free_predictions_used', 0)), FREE_PREDICTION_LIMIT)
    app.logger.info(
        'PREDICTION: Completed successfully in %.2f seconds; weed=%.2f%% crop=%.2f%% soil=%.2f%% device=%s',
        time.perf_counter() - request_started,
        weed_percentage,
        crop_percentage,
        soil_percentage,
        PROCESSING_DEVICE
    )
    return render_template('index.html', customer=customer, result=result, is_guest=customer_id is None, free_predictions_used=used, free_predictions_remaining=max(0, FREE_PREDICTION_LIMIT - used), saved_fields=get_saved_fields(customer_id) if customer_id else [])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip().lower()
    phone = request.form.get('phone', '').strip()
    farm_location = request.form.get('farm_location', '').strip()
    password = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')
    if not name:
        flash('Please enter your name.', 'danger')
        return redirect(url_for('register'))
    if not email:
        flash('Please enter your email.', 'danger')
        return redirect(url_for('register'))
    if not password:
        flash('Please enter a password.', 'danger')
        return redirect(url_for('register'))
    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('register'))
    if len(password) < 6:
        flash('Password must contain at least 6 characters.', 'danger')
        return redirect(url_for('register'))
    conn = get_db()
    existing = conn.execute('\n        SELECT id\n        FROM customers\n        WHERE email = ?\n        ', (email,)).fetchone()
    if existing:
        conn.close()
        flash('An account with this email already exists.', 'danger')
        return redirect(url_for('login'))
    password_hash = generate_password_hash(password)
    conn.execute('\n        INSERT INTO customers (\n            name,\n            email,\n            phone,\n            farm_location,\n            password_hash,\n            created_at\n        )\n        VALUES (?, ?, ?, ?, ?, ?)\n        ', (name, email, phone, farm_location, password_hash, get_current_time()))
    conn.commit()
    conn.close()
    flash('Registration successful. Please login.', 'success')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    conn = get_db()
    customer = conn.execute('\n        SELECT *\n        FROM customers\n        WHERE email = ?\n        ', (email,)).fetchone()
    if not customer:
        conn.close()
        flash('Invalid email or password.', 'danger')
        return redirect(url_for('login'))
    password_valid = False
    try:
        password_valid = check_password_hash(customer['password_hash'], password)
    except Exception:
        password_valid = False
    if not password_valid:
        conn.close()
        flash('Invalid email or password.', 'danger')
        return redirect(url_for('login'))
    selected_language = session.get('language', 'en')
    if selected_language not in SUPPORTED_LANGUAGES:
        selected_language = 'en'
    session.clear()
    session['language'] = selected_language
    session['customer_id'] = customer['id']
    session['customer_name'] = customer['name']
    session['customer_email'] = customer['email']
    now = get_current_time()
    conn.execute('\n        UPDATE customers\n        SET last_login = ?\n        WHERE id = ?\n        ', (now, customer['id']))
    conn.execute('\n        INSERT INTO login_history (\n            customer_id,\n            login_at,\n            ip_address\n        )\n        VALUES (?, ?, ?)\n        ', (customer['id'], now, get_client_ip()))
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    customer_id = session.get('customer_id')
    if customer_id:
        conn = get_db()
        row = conn.execute('\n            SELECT id\n            FROM login_history\n            WHERE customer_id = ?\n            AND logout_at IS NULL\n            ORDER BY id DESC\n            LIMIT 1\n            ', (customer_id,)).fetchone()
        if row:
            conn.execute('\n                UPDATE login_history\n                SET logout_at = ?\n                WHERE id = ?\n                ', (get_current_time(), row['id']))
            conn.commit()
        conn.close()
    selected_language = session.get('language', 'en')
    if selected_language not in SUPPORTED_LANGUAGES:
        selected_language = 'en'
    session.clear()
    session['language'] = selected_language
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/api/fields', methods=['GET', 'POST'])
@login_required
def api_fields():
    """List or save the logged-in customer's field names independently."""
    customer_id = session['customer_id']
    if request.method == 'GET':
        return jsonify({'success': True, 'fields': get_saved_fields(customer_id)})
    data = request.get_json(silent=True) or request.form
    field_name = normalize_field_name(data.get('field_name', ''))
    if not field_name or len(field_name) > 80:
        return (jsonify({'success': False, 'message': 'Enter a field name up to 80 characters.'}), 400)
    conn = get_db()
    try:
        saved_name = save_or_get_field(conn, customer_id, field_name)
        conn.commit()
        return jsonify({'success': True, 'field_name': saved_name, 'fields': get_saved_fields(customer_id)})
    finally:
        conn.close()

@app.route('/dashboard')
@login_required
def dashboard():
    customer_id = session['customer_id']
    conn = get_db()
    customer = conn.execute('\n        SELECT *\n        FROM customers\n        WHERE id = ?\n        ', (customer_id,)).fetchone()
    total_analyses = conn.execute('\n        SELECT COUNT(*)\n        AS count\n        FROM analyses\n        WHERE customer_id = ?\n        ', (customer_id,)).fetchone()['count']
    average_weed = conn.execute('\n        SELECT AVG(weed_percentage)\n        AS average\n        FROM analyses\n        WHERE customer_id = ?\n        ', (customer_id,)).fetchone()['average']
    if average_weed is None:
        average_weed = 0
    average_weed = round(float(average_weed), 2)
    high_count = conn.execute("\n        SELECT COUNT(*)\n        AS count\n        FROM analyses\n        WHERE customer_id = ?\n        AND weed_severity = 'High'\n        ", (customer_id,)).fetchone()['count']
    medium_count = conn.execute("\n        SELECT COUNT(*)\n        AS count\n        FROM analyses\n        WHERE customer_id = ?\n        AND weed_severity = 'Medium'\n        ", (customer_id,)).fetchone()['count']
    low_count = conn.execute("\n        SELECT COUNT(*)\n        AS count\n        FROM analyses\n        WHERE customer_id = ?\n        AND weed_severity = 'Low'\n        ", (customer_id,)).fetchone()['count']
    recent_analyses = conn.execute('\n        SELECT *\n        FROM analyses\n        WHERE customer_id = ?\n        ORDER BY id DESC\n        LIMIT 10\n        ', (customer_id,)).fetchall()
    field_analytics = conn.execute("\n        SELECT\n            COALESCE(NULLIF(TRIM(field_name), ''), 'Unassigned Field') AS field_name,\n            COUNT(*) AS total_analyses,\n            ROUND(AVG(COALESCE(weed_percentage, 0)), 2) AS weed_percentage,\n            ROUND(AVG(COALESCE(crop_percentage, 0)), 2) AS crop_percentage\n        FROM analyses\n        WHERE customer_id = ?\n        GROUP BY LOWER(COALESCE(NULLIF(TRIM(field_name), ''), 'Unassigned Field'))\n        ORDER BY field_name COLLATE NOCASE\n        ", (customer_id,)).fetchall()
    saved_fields = conn.execute('SELECT field_name FROM fields WHERE customer_id = ? ORDER BY field_name COLLATE NOCASE', (customer_id,)).fetchall()
    chart_data = []
    today = datetime.now().date()
    for i in range(13, -1, -1):
        current_date = today - timedelta(days=i)
        date_string = current_date.strftime('%Y-%m-%d')
        row = conn.execute('\n            SELECT\n                COUNT(*) AS total,\n                AVG(weed_percentage) AS average\n            FROM analyses\n            WHERE customer_id = ?\n            AND DATE(created_at) = ?\n            ', (customer_id, date_string)).fetchone()
        average = row['average']
        if average is None:
            average = 0
        chart_data.append({'date': current_date.strftime('%d %b'), 'count': row['total'], 'average': round(float(average), 2)})
    conn.close()
    return render_template('dashboard.html', customer=customer, total_analyses=total_analyses, average_weed=average_weed, high_count=high_count, medium_count=medium_count, low_count=low_count, recent_analyses=recent_analyses, chart_data=chart_data, field_analytics=field_analytics, saved_fields=saved_fields)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    customer_id = session['customer_id']
    conn = get_db()
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        phone = request.form.get('phone', '').strip()
        farm_location = request.form.get('farm_location', '').strip()
        conn.execute('\n            UPDATE customers\n            SET name = ?,\n                phone = ?,\n                farm_location = ?\n            WHERE id = ?\n            ', (name, phone, farm_location, customer_id))
        conn.commit()
        session['customer_name'] = name
        flash('Profile updated successfully.', 'success')
    customer = conn.execute('\n        SELECT *\n        FROM customers\n        WHERE id = ?\n        ', (customer_id,)).fetchone()
    conn.close()
    return render_template('profile.html', customer=customer)

@app.route('/history')
@login_required
def history():
    customer_id = session['customer_id']
    conn = get_db()
    analyses = conn.execute('\n        SELECT *\n        FROM analyses\n        WHERE customer_id = ?\n        ORDER BY id DESC\n        ', (customer_id,)).fetchall()
    conn.close()
    return render_template('history.html', analyses=analyses)

@app.route('/api/analysis/<int:analysis_id>')
@login_required
def analysis_details(analysis_id):
    customer_id = session['customer_id']
    conn = get_db()
    analysis = conn.execute('\n        SELECT *\n        FROM analyses\n        WHERE id = ?\n        AND customer_id = ?\n        ', (analysis_id, customer_id)).fetchone()
    conn.close()
    if not analysis:
        return (jsonify({'success': False, 'error': 'Analysis not found.'}), 404)
    return jsonify({'success': True, 'analysis': dict(analysis)})
WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'tiny')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')
WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'int8')
_whisper_model = None

def get_whisper_model():
    """Load Faster-Whisper once and reuse it for later recordings."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE, cpu_threads=4, num_workers=1)
    return _whisper_model

@app.route('/api/local-transcribe', methods=['POST'])
@login_required
def local_transcribe():
    """Transcribe a microphone recording and automatically detect language."""
    audio = request.files.get('audio')
    if audio is None or not audio.filename:
        return (jsonify({'success': False, 'message': 'No audio recording was received.'}), 400)
    suffix = Path(audio.filename).suffix.lower()
    if suffix not in {'.webm', '.wav', '.ogg', '.mp4', '.m4a', '.mp3', '.mpeg'}:
        suffix = '.webm'
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            audio.save(temp_path)
        model = get_whisper_model()
        requested_language = str(request.form.get('language', 'auto')).strip().lower()
        whisper_language = requested_language if requested_language in SUPPORTED_LANGUAGES else None
        segments, info = model.transcribe(temp_path, task='transcribe', language=whisper_language, beam_size=1, best_of=1, vad_filter=True, condition_on_previous_text=False)
        transcript = ' '.join((segment.text.strip() for segment in segments if segment.text.strip())).strip()
        detected_language = (getattr(info, 'language', None) or '').lower()
        if not transcript:
            return (jsonify({'success': False, 'message': 'I could not understand the recording. Please speak clearly and try again.'}), 422)
        supported = {'en', 'te', 'ta', 'hi', 'ml', 'kn'}
        if detected_language not in supported:
            return jsonify({'success': True, 'text': transcript, 'language': detected_language, 'supported': False, 'message': 'Please speak English, Telugu, Tamil, Hindi, Malayalam, or Kannada.'})
        return jsonify({'success': True, 'text': transcript, 'language': detected_language, 'supported': True})
    except ImportError:
        app.logger.exception('Faster-Whisper is not installed')
        return (jsonify({'success': False, 'message': 'Speech recognition is not installed. Run: pip install faster-whisper'}), 500)
    except Exception:
        app.logger.exception('Local speech transcription failed')
        return (jsonify({'success': False, 'message': 'Speech recognition failed. Check the model download and try again.'}), 500)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

@app.route('/api/assistant/weed-status', methods=['GET'])
@login_required
def assistant_weed_status():
    """Return the logged-in user's latest saved U-Net analysis."""
    customer_id = session.get('customer_id')
    conn = get_db()
    try:
        row = conn.execute('\n            SELECT id, weed_percentage, weed_severity,\n                   suppression_recommendation, suppression_status,\n                   workflow_status, created_at\n            FROM analyses\n            WHERE customer_id = ?\n            ORDER BY id DESC\n            LIMIT 1\n            ', (customer_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return jsonify({'success': False, 'message': 'No weed analysis found. Upload a field image and run U-Net analysis first.'})
    return jsonify({'success': True, 'analysis': dict(row), 'id': row['id'], 'weed_percentage': row['weed_percentage'], 'weed_severity': row['weed_severity'], 'suppression_recommendation': row['suppression_recommendation'], 'suppression_status': row['suppression_status'], 'workflow_status': row['workflow_status'], 'created_at': row['created_at']})
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/chat')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')

def generate_local_assistant_answer(question, language):
    """Generate an answer with the locally running Ollama model."""
    language = language if language in SUPPORTED_LANGUAGES else 'en'
    language_name = SUPPORTED_LANGUAGES[language]
    conn = get_db()
    try:
        row = conn.execute('SELECT weed_percentage, weed_severity,\n                      suppression_recommendation, suppression_status,\n                      workflow_status, created_at\n               FROM analyses WHERE customer_id = ?\n               ORDER BY id DESC LIMIT 1', (session['customer_id'],)).fetchone()
    finally:
        conn.close()
    if row:
        analysis_context = f"Latest saved U-Net analysis: weed coverage={row['weed_percentage']}%, severity={row['weed_severity']}, status={row['suppression_status']}, recommendation={row['suppression_recommendation']}, workflow={row['workflow_status']}, date={row['created_at']}."
    else:
        analysis_context = 'No saved U-Net analysis exists for this user yet.'
    system_prompt = f"You are Smart Weed Assistant, a helpful agricultural assistant integrated into the Smart Weed Detection System.\n\nLANGUAGE:\n- Answer in {language_name}. Understand Telugu script and Telugu written using English letters.\n- Answer the user's actual question directly. Do not repeat an introduction.\n- If the question is about weed removal, give practical steps rather than only describing this project.\n\nPROJECT:\n- The project uses a U-Net deep-learning model to segment background/soil, crops, and weeds in field images.\n- The web application uses Flask, SQLite, and a local Ollama model.\n- Deployment is pending. Automatic weed removal and thermal suppression are proposed future features, not implemented.\n- Never invent image-analysis results. If the user asks about their own prediction and there is no saved analysis, ask them to upload an image; still answer general agricultural questions.\n\nWEED MANAGEMENT:\n- For questions about removing weeds, explain suitable methods such as pulling small weeds out with roots, shallow hoeing without damaging crop roots, suitable mulch, and removing weeds before they produce seeds.\n- Explain that the best method depends on crop, weed type, weed size, and field conditions.\n- For herbicides, recommend only crop-appropriate products and following the label and local agricultural guidance.\n- Never recommend salt, bleach, diesel, or other harmful substances on farmland.\n- Do not claim this website physically removes weeds or controls equipment.\n- Give simple, practical, crop-safe advice; ask a short follow-up only if necessary.\n\nLATEST SAVED ANALYSIS:\n{analysis_context}\n\nUSER QUESTION:\n{question}\n\nReply directly in {language_name}, using clear language and numbered steps when helpful."
    response = requests.post(OLLAMA_URL, json={'model': OLLAMA_MODEL, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': question}], 'stream': False, 'options': {'temperature': 0.4}}, timeout=180)
    response.raise_for_status()
    payload = response.json()
    answer = str(payload.get('message', {}).get('content', '')).strip()
    if not answer:
        raise ValueError('The local AI model returned an empty answer.')
    return answer

def _assistant_response_from_request():
    data = request.get_json(silent=True) or {}
    question = str(data.get('message', data.get('question', ''))).strip()
    language = str(data.get('language') or session.get('language') or 'en').lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        language = 'en'
    if not question:
        return (jsonify({'success': False, 'error': 'Please enter or speak a question.'}), 400)
    if len(question) > 2000:
        return (jsonify({'success': False, 'error': 'Please keep your question under 2000 characters.'}), 400)
    try:
        answer = generate_local_assistant_answer(question, language)
        return jsonify({'success': True, 'answer': answer, 'language': language})
    except requests.exceptions.ConnectionError:
        app.logger.exception('Could not connect to Ollama')
        return (jsonify({'success': False, 'error': 'Ollama is not running. Start Ollama and try again.'}), 503)
    except requests.exceptions.Timeout:
        return (jsonify({'success': False, 'error': 'The assistant took too long to respond. Please try again.'}), 504)
    except requests.exceptions.RequestException:
        app.logger.exception('Ollama request failed')
        return (jsonify({'success': False, 'error': 'Could not get a response from the local AI model.'}), 502)
    except ValueError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 502)

@app.route('/api/assistant/chat', methods=['POST'])
@login_required
def assistant_chat():
    """Chat endpoint for the multilingual website assistant."""
    return _assistant_response_from_request()

@app.route('/api/voice-answer', methods=['POST'])
@login_required
def voice_answer():
    """Compatibility endpoint for existing voice.js code."""
    return _assistant_response_from_request()
if __name__ == '__main__':
    print('=' * 60)
    print('SMART WEED DETECTION SYSTEM')
    print('=' * 60)
    print(f'Database: {DATABASE}')
    print(f'TFLite model: {MODEL_PATH}')
    print('Server: http://127.0.0.1:5000')
    print('=' * 60)
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
