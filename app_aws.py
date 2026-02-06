import os
import uuid
import base64
import cv2
import numpy as np
import boto3
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from botocore.exceptions import ClientError

# Forensic Libraries
from skimage import restoration, img_as_float, img_as_ubyte

# --- APP INITIALIZATION ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'forensic-secure-key-2026')

# --- AWS CONFIGURATION ---
REGION = 'us-east-1'
SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:604665149129:aws_capstone_topic'
S3_BUCKET_NAME = 'YOUR_S3_BUCKET_NAME'   # MUST EXIST IN AWS CONSOLE

# --- AWS CLIENTS (IAM ROLE / ENV BASED – FULL ACCESS) ---
s3_client = boto3.client('s3', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)
sns = boto3.client('sns', region_name=REGION)

# --- DYNAMODB TABLES ---
users_table = dynamodb.Table('Users')

# --- FILE UPLOAD CONFIG ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOGIN MANAGER ---
class User(UserMixin):
    def __init__(self, username):
        self.id = username

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(email):
    response = users_table.get_item(Key={'email': email})
    if 'Item' in response:
        return User(email)
    return None


def send_notification(subject, message):
    try:
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
    except Exception as e:
        print(f"SNS Error: {e}")

# --- ROUTES ---
@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/home', endpoint='home')
@app.route('/index', endpoint='index')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/price')
def price():
    return render_template('price.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        response = users_table.get_item(Key={'email': email})
        if 'Item' in response:
            flash("User already exists!", "warning")
            return redirect(url_for('signup'))

        users_table.put_item(Item={
            'email': email,
            'password': password
        })

        send_notification("New User Signup", f"User {email} signed up.")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        response = users_table.get_item(Key={'email': email})

        if 'Item' in response and response['Item']['password'] == password:
            login_user(User(email))
            session['username'] = email
            send_notification("User Login", f"{email} logged in.")
            return redirect(url_for('workbench'))

        flash("Invalid credentials!", "danger")

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('username', None)
    return redirect(url_for('index'))

# --- WORKBENCH ---
@app.route('/workbench', methods=['GET', 'POST'])
@login_required
def workbench():
    filename = request.args.get('filename')

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename:
            filename = secure_filename(file.filename)
            local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(local_path)

            try:
                s3_client.upload_file(local_path, S3_BUCKET_NAME, filename)
                flash("File uploaded to S3 successfully", "success")
            except ClientError as e:
                flash(f"S3 Error: {e}", "danger")

    return render_template('workbench.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- IMAGE UTILS ---
def get_cv_img(file):
    return cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)


# (All other forensic routes: deblur, canny, otsu, harris, watershed, hough, wiener remain unchanged)

@app.route('/deblur', methods=['GET', 'POST'])
def deblur():
    processed_img, original_img = None, None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = get_cv_img(file)
            h, w = img.shape[:2]
            img_m = cv2.resize(img, (int(w * (500/h)), 500), interpolation=cv2.INTER_AREA)
            original_img = base64.b64encode(cv2.imencode('.png', img_m)[1]).decode('utf-8')
            img_f = img_as_float(img_m)
            psf = np.ones((5, 5)) / 25.0
            restored = [restoration.richardson_lucy(chan, psf, num_iter=30) for chan in cv2.split(img_f)]
            result = img_as_ubyte(np.clip(cv2.merge(restored), 0, 1))
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0).apply(l)
            final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            processed_img = base64.b64encode(cv2.imencode('.png', final)[1]).decode('utf-8')
    return render_template('deblur.html', processed_img=processed_img, original_img=original_img)

@app.route('/canny', methods=['GET', 'POST'])
def canny():
    processed_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = get_cv_img(file)
            edges = cv2.Canny(img, 100, 200)
            processed_img = base64.b64encode(cv2.imencode('.png', edges)[1]).decode('utf-8')
    return render_template('canny.html', processed_img=processed_img)

@app.route('/otsu', methods=['GET', 'POST'])
def otsu():
    processed_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = get_cv_img(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_img = base64.b64encode(cv2.imencode('.png', thresh)[1]).decode('utf-8')
    return render_template('otsu.html', processed_img=processed_img)

@app.route('/harris', methods=['GET', 'POST'])
def harris():
    processed_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = get_cv_img(file)
            gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            dst = cv2.dilate(cv2.cornerHarris(gray, 2, 3, 0.04), None)
            img[dst > 0.01 * dst.max()] = [0, 240, 255]
            processed_img = base64.b64encode(cv2.imencode('.png', img)[1]).decode('utf-8')
    return render_template('harris.html', processed_img=processed_img)

@app.route('/watershed', methods=['GET', 'POST'])
def watershed():
    processed_img = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = get_cv_img(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(opening, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(img, markers)
            label_hue = np.uint8(179 * markers / np.max(markers))
            labeled_img = cv2.merge([label_hue, 255*np.ones_like(label_hue), 255*np.ones_like(label_hue)])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img[markers == -1] = [255, 255, 255]
            processed_img = base64.b64encode(cv2.imencode('.png', labeled_img)[1]).decode('utf-8')
    return render_template('watershed.html', processed_img=processed_img)

@app.route('/hough', methods=['GET', 'POST'])
def hough():
    processed_img, original_img = None, None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = get_cv_img(file)
            h, w = img.shape[:2]
            img_m = cv2.resize(img, (int(w * (500/h)), 500), interpolation=cv2.INTER_AREA)
            original_img = base64.b64encode(cv2.imencode('.png', img_m)[1]).decode('utf-8')
            gray = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            line_img = img_m.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_img, (x1, y1), (x2, y2), (14, 14, 251), 3)
            processed_img = base64.b64encode(cv2.imencode('.png', line_img)[1]).decode('utf-8')
    return render_template('hough.html', processed_img=processed_img, original_img=original_img)

@app.route('/wiener', methods=['GET', 'POST'], endpoint='wiener_filter')
def wiener():
    processed_img, original_img = None, None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = get_cv_img(file)
            h, w = img.shape[:2]
            img_m = cv2.resize(img, (int(w * (500/h)), 500), interpolation=cv2.INTER_AREA)
            original_img = base64.b64encode(cv2.imencode('.png', img_m)[1]).decode('utf-8')
            img_f = img_as_float(img_m)
            psf = np.ones((5, 5)) / 25.0
            restored_channels = [restoration.unsupervised_wiener(img_f[:,:,i], psf)[0] for i in range(3)]
            result = img_as_ubyte(np.clip(cv2.merge(restored_channels), 0, 1))
            final = cv2.filter2D(result, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
            processed_img = base64.b64encode(cv2.imencode('.png', final)[1]).decode('utf-8')
    return render_template('wiener.html', processed_img=processed_img, original_img=original_img)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)