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
MODEL_PATH = BASE_DIR / 'models' / 'recovered_unet.keras'
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
TRANSLATIONS = {'en': {'app_name': 'Smart Weed Detection System', 'welcome': 'Welcome', 'home': 'Home', 'dashboard': 'Dashboard', 'history': 'History', 'profile': 'Profile', 'logout': 'Logout', 'login': 'Login', 'register': 'Register', 'language': 'Language', 'select_language': 'Select Language', 'assistant': 'AI Assistant', 'assistant_welcome': 'Hello! How can I help you?', 'ready': 'Ready', 'ask_question': 'Ask a question...', 'speak': 'Speak', 'send': 'Send', 'upload_image': 'Upload Field Image', 'analyze': 'Analyze Image', 'prediction_result': 'Prediction Result', 'weed_coverage': 'Weed Coverage', 'weed_severity': 'Weed Severity', 'confidence': 'Confidence', 'source_image': 'Original Image', 'overlay_image': 'Segmentation Overlay', 'weed_mask': 'Weed Mask', 'recommendation': 'Recommendation', 'analyzing': 'Analyzing...', 'recent_analyses': 'Recent Analyses', 'date': 'Date', 'full_name': 'Full Name', 'email': 'Email', 'phone': 'Phone', 'password': 'Password', 'confirm_password': 'Confirm Password', 'create_account': 'Create Account', 'total_analyses': 'Total Analyses', 'average_weed': 'Average Weed Coverage', 'high': 'High', 'medium': 'Medium', 'low': 'Low', 'no_analyses': 'No analyses found', 'farm_location': 'Farm Location', 'save_changes': 'Save Changes', 'update_profile': 'Update Profile', 'guest_limit': 'Guest prediction limit', 'remaining': 'Remaining'}, 'te': {'app_name': 'స్మార్ట్ కలుపు మొక్కల గుర్తింపు వ్యవస్థ', 'welcome': 'స్వాగతం', 'home': 'హోమ్', 'dashboard': 'డాష్\u200cబోర్డ్', 'history': 'చరిత్ర', 'profile': 'ప్రొఫైల్', 'logout': 'లాగ్అవుట్', 'login': 'లాగిన్', 'register': 'నమోదు', 'language': 'భాష', 'select_language': 'భాషను ఎంచుకోండి', 'assistant': 'AI సహాయకుడు', 'upload_image': 'పొలం చిత్రాన్ని అప్\u200cలోడ్ చేయండి', 'analyze': 'చిత్రాన్ని విశ్లేషించండి', 'prediction_result': 'ఫలితం', 'weed_coverage': 'కలుపు మొక్కల విస్తీర్ణం', 'weed_severity': 'కలుపు తీవ్రత', 'confidence': 'నమ్మక స్థాయి', 'source_image': 'అసలు చిత్రం', 'overlay_image': 'విశ్లేషణ చిత్రం', 'weed_mask': 'కలుపు మాస్క్', 'recommendation': 'సిఫార్సు', 'recent_analyses': 'ఇటీవలి విశ్లేషణలు', 'date': 'తేదీ', 'full_name': 'పూర్తి పేరు', 'email': 'ఇమెయిల్', 'phone': 'ఫోన్', 'password': 'పాస్\u200cవర్డ్', 'confirm_password': 'పాస్\u200cవర్డ్ నిర్ధారించండి', 'create_account': 'ఖాతా సృష్టించండి', 'total_analyses': 'మొత్తం విశ్లేషణలు', 'average_weed': 'సగటు కలుపు విస్తీర్ణం', 'farm_location': 'వ్యవసాయ క్షేత్ర స్థానం', 'save_changes': 'మార్పులను సేవ్ చేయండి'}, 'ta': {'app_name': 'ஸ்மார்ட் களை கண்டறிதல் அமைப்பு', 'welcome': 'வரவேற்கிறோம்', 'home': 'முகப்பு', 'dashboard': 'டாஷ்போர்டு', 'history': 'வரலாறு', 'profile': 'சுயவிவரம்', 'logout': 'வெளியேறு', 'login': 'உள்நுழை', 'register': 'பதிவு', 'language': 'மொழி', 'select_language': 'மொழியைத் தேர்ந்தெடுக்கவும்', 'assistant': 'AI உதவியாளர்', 'upload_image': 'வயல் படத்தை பதிவேற்றவும்', 'analyze': 'படத்தை பகுப்பாய்வு செய்க', 'prediction_result': 'கணிப்பு முடிவு', 'weed_coverage': 'களை பரப்பளவு', 'weed_severity': 'களை தீவிரம்', 'confidence': 'நம்பகத்தன்மை', 'source_image': 'அசல் படம்', 'overlay_image': 'பகுப்பாய்வு படம்', 'weed_mask': 'களை மாஸ்க்', 'recommendation': 'பரிந்துரை', 'recent_analyses': 'சமீபத்திய பகுப்பாய்வுகள்', 'date': 'தேதி', 'full_name': 'முழு பெயர்', 'email': 'மின்னஞ்சல்', 'phone': 'தொலைபேசி', 'password': 'கடவுச்சொல்', 'confirm_password': 'கடவுச்சொல்லை உறுதிப்படுத்தவும்', 'create_account': 'கணக்கை உருவாக்கவும்', 'total_analyses': 'மொத்த பகுப்பாய்வுகள்', 'average_weed': 'சராசரி களை பரப்பளவு', 'farm_location': 'பண்ணை இடம்', 'save_changes': 'மாற்றங்களைச் சேமிக்கவும்'}, 'kn': {'app_name': 'ಸ್ಮಾರ್ಟ್ ಕಳೆ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ', 'welcome': 'ಸ್ವಾಗತ', 'home': 'ಮುಖಪುಟ', 'dashboard': 'ಡ್ಯಾಶ್\u200cಬೋರ್ಡ್', 'history': 'ಇತಿಹಾಸ', 'profile': 'ಪ್ರೊಫೈಲ್', 'logout': 'ಲಾಗ್ ಔಟ್', 'login': 'ಲಾಗಿನ್', 'register': 'ನೋಂದಣಿ', 'language': 'ಭಾಷೆ', 'select_language': 'ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ', 'assistant': 'AI ಸಹಾಯಕ', 'upload_image': 'ಹೊಲದ ಚಿತ್ರವನ್ನು ಅಪ್\u200cಲೋಡ್ ಮಾಡಿ', 'analyze': 'ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ', 'prediction_result': 'ಫಲಿತಾಂಶ', 'weed_coverage': 'ಕಳೆ ವ್ಯಾಪ್ತಿ', 'weed_severity': 'ಕಳೆಯ ತೀವ್ರತೆ', 'confidence': 'ನಿಖರತೆಯ ಮಟ್ಟ', 'source_image': 'ಮೂಲ ಚಿತ್ರ', 'overlay_image': 'ವಿಶ್ಲೇಷಣಾ ಚಿತ್ರ', 'weed_mask': 'ಕಳೆ ಮಾಸ್ಕ್', 'recommendation': 'ಶಿಫಾರಸು', 'recent_analyses': 'ಇತ್ತೀಚಿನ ವಿಶ್ಲೇಷಣೆಗಳು', 'date': 'ದಿನಾಂಕ', 'full_name': 'ಪೂರ್ಣ ಹೆಸರು', 'email': 'ಇಮೇಲ್', 'phone': 'ದೂರವಾಣಿ', 'password': 'ಪಾಸ್\u200cವರ್ಡ್', 'confirm_password': 'ಪಾಸ್\u200cವರ್ಡ್ ದೃಢೀಕರಿಸಿ', 'create_account': 'ಖಾತೆ ರಚಿಸಿ', 'total_analyses': 'ಒಟ್ಟು ವಿಶ್ಲೇಷಣೆಗಳು', 'average_weed': 'ಸರಾಸರಿ ಕಳೆ ವ್ಯಾಪ್ತಿ', 'farm_location': 'ಕೃಷಿ ಸ್ಥಳ', 'save_changes': 'ಬದಲಾವಣೆಗಳನ್ನು ಉಳಿಸಿ'}, 'ml': {'app_name': 'സ്മാർട്ട് കള തിരിച്ചറിയൽ സംവിധാനം', 'welcome': 'സ്വാഗതം', 'home': 'ഹോം', 'dashboard': 'ഡാഷ്ബോർഡ്', 'history': 'ചരിത്രം', 'profile': 'പ്രൊഫൈൽ', 'logout': 'ലോഗ് ഔട്ട്', 'login': 'ലോഗിൻ', 'register': 'രജിസ്റ്റർ', 'language': 'ഭാഷ', 'select_language': 'ഭാഷ തിരഞ്ഞെടുക്കുക', 'assistant': 'AI സഹായി', 'upload_image': 'കൃഷിയിട ചിത്രം അപ്\u200cലോഡ് ചെയ്യുക', 'analyze': 'ചിത്രം വിശകലനം ചെയ്യുക', 'prediction_result': 'ഫലം', 'weed_coverage': 'കളയുടെ വ്യാപ്തി', 'weed_severity': 'കളയുടെ തീവ്രത', 'confidence': 'വിശ്വാസ്യത', 'source_image': 'യഥാർത്ഥ ചിത്രം', 'overlay_image': 'വിശകലന ചിത്രം', 'weed_mask': 'കള മാസ്ക്', 'recommendation': 'ശുപാർശ', 'recent_analyses': 'സമീപകാല വിശകലനങ്ങൾ', 'date': 'തീയതി', 'full_name': 'പൂർണ്ണ പേര്', 'email': 'ഇമെയിൽ', 'phone': 'ഫോൺ', 'password': 'പാസ്\u200cവേഡ്', 'confirm_password': 'പാസ്\u200cവേഡ് സ്ഥിരീകരിക്കുക', 'create_account': 'അക്കൗണ്ട് സൃഷ്ടിക്കുക', 'total_analyses': 'ആകെ വിശകലനങ്ങൾ', 'average_weed': 'ശരാശരി കള വ്യാപ്തി', 'farm_location': 'കൃഷിയിട സ്ഥലം', 'save_changes': 'മാറ്റങ്ങൾ സംരക്ഷിക്കുക'}, 'hi': {'app_name': 'स्मार्ट खरपतवार पहचान प्रणाली', 'welcome': 'स्वागत है', 'home': 'होम', 'dashboard': 'डैशबोर्ड', 'history': 'इतिहास', 'profile': 'प्रोफ़ाइल', 'logout': 'लॉग आउट', 'login': 'लॉगिन', 'register': 'पंजीकरण', 'language': 'भाषा', 'select_language': 'भाषा चुनें', 'assistant': 'AI सहायक', 'upload_image': 'खेत की तस्वीर अपलोड करें', 'analyze': 'तस्वीर का विश्लेषण करें', 'prediction_result': 'परिणाम', 'weed_coverage': 'खरपतवार का क्षेत्र', 'weed_severity': 'खरपतवार की गंभीरता', 'confidence': 'विश्वास स्तर', 'source_image': 'मूल तस्वीर', 'overlay_image': 'विश्लेषण तस्वीर', 'weed_mask': 'खरपतवार मास्क', 'recommendation': 'सुझाव', 'recent_analyses': 'हाल के विश्लेषण', 'date': 'तारीख', 'full_name': 'पूरा नाम', 'email': 'ईमेल', 'phone': 'फोन', 'password': 'पासवर्ड', 'confirm_password': 'पासवर्ड की पुष्टि करें', 'create_account': 'खाता बनाएं', 'total_analyses': 'कुल विश्लेषण', 'average_weed': 'औसत खरपतवार क्षेत्र', 'farm_location': 'खेत का स्थान', 'save_changes': 'बदलाव सहेजें'}}

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
    if next_page.startswith('/') and (not next_page.startswith('//')):
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
_model = None
PROCESSING_DEVICE = "CPU"

def load_model():
    global _model, PROCESSING_DEVICE
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f'U-Net model not found:\n{MODEL_PATH}')

        app.logger.info('PREDICTION: TensorFlow import starting')
        tensorflow_started = time.perf_counter()
        import tensorflow as tf
        app.logger.info(
            'PREDICTION: TensorFlow import completed in %.2f seconds',
            time.perf_counter() - tensorflow_started
        )

        # Prevent TensorFlow from creating too many CPU worker threads on Render.
        try:
            tf.config.threading.set_intra_op_parallelism_threads(
                int(os.getenv('TF_NUM_INTRAOP_THREADS', '1'))
            )
            tf.config.threading.set_inter_op_parallelism_threads(
                int(os.getenv('TF_NUM_INTEROP_THREADS', '1'))
            )
        except RuntimeError:
            app.logger.warning(
                'TensorFlow threading was already initialized; using existing settings.'
            )

        gpu_devices = tf.config.list_physical_devices('GPU')
        for gpu in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                app.logger.warning('Could not set GPU memory growth for %s', gpu)

        PROCESSING_DEVICE = 'GPU' if gpu_devices else 'CPU'
        app.logger.info(
            'PREDICTION: Processing device: %s',
            'GPU (' + str(len(gpu_devices)) + ')' if gpu_devices else 'CPU'
        )

        model_load_started = time.perf_counter()
        app.logger.info('PREDICTION: U-Net model loading started')
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        app.logger.info(
            'PREDICTION: U-Net model loading completed in %.2f seconds',
            time.perf_counter() - model_load_started
        )
    return _model

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
        return render_template('index.html', customer=customer, result=None, is_guest=customer_id is None, free_predictions_used=used, free_predictions_remaining=max(0, FREE_PREDICTION_LIMIT - used), saved_fields=get_saved_fields(customer_id) if customer_id else [])
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
        app.logger.info('PREDICTION: Requesting U-Net model')
        model = load_model()
        app.logger.info('PREDICTION: Model is ready; starting U-Net inference')
        inference_started = time.perf_counter()
        prediction = model(input_array, training=False)
        if hasattr(prediction, 'numpy'):
            prediction = prediction.numpy()
        app.logger.info('PREDICTION: Raw U-Net output received; shape=%s', np.shape(prediction))
        probabilities = normalize_prediction(prediction)
        app.logger.info(
            'PREDICTION: U-Net inference and normalization completed in %.2f seconds',
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
    result = {'success': True, 'analysis_id': analysis_id, 'field_name': field_name, 'crop_percentage': crop_percentage, 'soil_percentage': soil_percentage, 'device': PROCESSING_DEVICE, 'processing_device': PROCESSING_DEVICE, 'source_url': url_for('static', filename=f'uploads/{source_filename}'), 'original_image_url': url_for('static', filename=f'uploads/{source_filename}'), 'original_image': url_for('static', filename=f'uploads/{source_filename}'), 'overlay_url': url_for('static', filename=f'uploads/{overlay_filename}'), 'overlay_image_url': url_for('static', filename=f'uploads/{overlay_filename}'), 'segmentation_overlay_url': url_for('static', filename=f'uploads/{overlay_filename}'), 'weed_mask_url': url_for('static', filename=f'uploads/{weed_mask_filename}'), 'weed_mask_image_url': url_for('static', filename=f'uploads/{weed_mask_filename}'), 'filename': original_filename, 'weed_percentage': weed_percentage, 'weed_pixel_count': weed_pixel_count, 'weed_severity': severity, 'weed_location': location, 'weed_bbox': bbox, 'confidence': confidence, 'classes': classes, 'suppression_status': suppression['status'], 'suppression_recommendation': suppression['recommendation'], 'post_treatment_status': post_treatment_status, 'workflow_status': workflow_status}
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
    print(f'Model: {MODEL_PATH}')
    print('Server: http://127.0.0.1:5000')
    print('=' * 60)
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
