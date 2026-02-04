from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import base64
import cv2
import numpy as np

# Forensic Libraries
from skimage import restoration, img_as_float, img_as_ubyte

# --- APP INITIALIZATION ---
app = Flask(__name__)
app.secret_key = 'glitter-cv-secret'

# --- DATABASE & UPLOAD SETUP ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- MODELS ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# --- AUTH & NAVIGATION ROUTES ---

@app.route('/')
def splash():
    """First entry point for the application."""
    return render_template('splash.html')

@app.route('/home', endpoint='home')
@app.route('/index', endpoint='index')
def home(): 
    """Fixes BuildError for 'home' and 'index' endpoints."""
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/price')
def price(): 
    return render_template('price.html')

# FIXED: Changed function name from 'price' to 'contact' to prevent AssertionError
@app.route('/contact')
def contact(): 
    return render_template('contact.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "info")
            return redirect(url_for('login'))
        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('workbench'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            login_user(user)
            return redirect(url_for('workbench'))
        flash("User not found.", "danger")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user() 
    session.clear() 
    flash("Securely logged out.", "info")
    return redirect(url_for('home'))

# --- WORKBENCH & UPLOAD ---

@app.route('/workbench', methods=['GET', 'POST'], endpoint='workbench')
@app.route('/upload', methods=['GET', 'POST'], endpoint='upload')
@login_required 
def workbench():
    filename = request.args.get('filename')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return render_template('workbench.html', filename=filename)

    return render_template('workbench.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- IMAGE PROCESSING TOOL ROUTES ---

@app.route('/deblur', methods=['GET', 'POST'])
def deblur():
    processed_img = None
    original_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            img_m = cv2.resize(img, (int(w * (500/h)), 500), interpolation=cv2.INTER_AREA)
            _, og_buf = cv2.imencode('.png', img_m)
            original_img = base64.b64encode(og_buf).decode('utf-8')
            img_f = img_as_float(img_m)
            psf = np.ones((5, 5)) / 25.0
            restored = [restoration.richardson_lucy(chan, psf, num_iter=30) for chan in cv2.split(img_f)]
            result = img_as_ubyte(np.clip(cv2.merge(restored), 0, 1))
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0).apply(l)
            final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            _, buffer = cv2.imencode('.png', final)
            processed_img = base64.b64encode(buffer).decode('utf-8')
    return render_template('deblur.html', processed_img=processed_img, original_img=original_img)

@app.route('/canny', methods=['GET', 'POST'])
def canny():
    processed_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            edges = cv2.Canny(img, 100, 200)
            _, buffer = cv2.imencode('.png', edges)
            processed_img = base64.b64encode(buffer).decode('utf-8')
    return render_template('canny.html', processed_img=processed_img)

@app.route('/otsu', methods=['GET', 'POST'])
def otsu():
    processed_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, buffer = cv2.imencode('.png', thresh)
            processed_img = base64.b64encode(buffer).decode('utf-8')
    return render_template('otsu.html', processed_img=processed_img)

@app.route('/harris', methods=['GET', 'POST'])
def harris():
    processed_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            dst = cv2.dilate(cv2.cornerHarris(gray, 2, 3, 0.04), None)
            img[dst > 0.01 * dst.max()] = [0, 240, 255]
            _, buffer = cv2.imencode('.png', img)
            processed_img = base64.b64encode(buffer).decode('utf-8')
    return render_template('harris.html', processed_img=processed_img)

@app.route('/watershed', methods=['GET', 'POST'])
def watershed():
    processed_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(opening, sure_fg)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(img, markers)
            label_hue = np.uint8(179 * markers / np.max(markers))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img[markers == -1] = [255, 255, 255]
            _, buffer = cv2.imencode('.png', labeled_img)
            processed_img = base64.b64encode(buffer).decode('utf-8')
    return render_template('watershed.html', processed_img=processed_img)

@app.route('/hough', methods=['GET', 'POST'])
def hough():
    processed_img = None
    original_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            img_m = cv2.resize(img, (int(w * (500/h)), 500), interpolation=cv2.INTER_AREA)
            _, og_buf = cv2.imencode('.png', img_m)
            original_img = base64.b64encode(og_buf).decode('utf-8')
            gray = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            line_img = img_m.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_img, (x1, y1), (x2, y2), (14, 14, 251), 3)
            _, buffer = cv2.imencode('.png', line_img)
            processed_img = base64.b64encode(buffer).decode('utf-8')
    return render_template('hough.html', processed_img=processed_img, original_img=original_img)

@app.route('/wiener', methods=['GET', 'POST'])
def wiener_filter():
    processed_img = None
    original_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            img_m = cv2.resize(img, (int(w * (500/h)), 500), interpolation=cv2.INTER_AREA)
            _, og_buf = cv2.imencode('.png', img_m)
            original_img = base64.b64encode(og_buf).decode('utf-8')
            img_f = img_as_float(img_m)
            psf = np.ones((5, 5)) / 25.0
            restored_channels = []
            for i in range(3):
                deconvolved, _ = restoration.unsupervised_wiener(img_f[:,:,i], psf)
                restored_channels.append(deconvolved)
            result = cv2.merge(restored_channels)
            result = img_as_ubyte(np.clip(result, 0, 1))
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            final = cv2.filter2D(result, -1, kernel)
            _, buffer = cv2.imencode('.png', final)
            processed_img = base64.b64encode(buffer).decode('utf-8')
    return render_template('wiener.html', processed_img=processed_img, original_img=original_img)

if __name__ == '__main__':
    app.run(debug=True, port=5001)