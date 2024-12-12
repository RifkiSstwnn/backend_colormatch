import os
from sqlite3 import IntegrityError
from flask import Flask, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import uuid
from flask_cors import CORS
from werkzeug.utils import secure_filename
from main import main
from PIL import Image
import random
import cv2
from sqlalchemy import func
import numpy as np
import json

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)

# Konfigurasi Database MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/colormatch'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inisialisasi SQLAlchemy
db = SQLAlchemy(app)

# Model untuk tabel User
class User(db.Model):
    __tablename__ = 'user'
    # UUID sebagai primary key
    uuid = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relasi ke tabel History
    histories = db.relationship('History', backref='user', lazy=True)

# Model untuk tabel History


class History(db.Model):
    __tablename__ = 'history'
    
    # ID untuk setiap history record
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # UUID sebagai foreign key dari tabel User
    user_uuid = db.Column(db.String(36), db.ForeignKey('user.uuid'), nullable=False)
    
    # Kolom untuk menyimpan foto input dan output
    foto_input = db.Column(db.String(255), nullable=False)
    foto_output = db.Column(db.String(255), nullable=False)
    
    # Kolom untuk menyimpan data skin tone dan color palette
    skin_tone = db.Column(db.String(50), nullable=False)
    color_palette = db.Column(db.String(500), nullable=False)

    # Kolom untuk menyimpan confidence level
    confidence = db.Column(db.String(50), nullable=False)  # Kolom baru untuk confidence level

    # Kolom untuk menyimpan timestamp
    timestamp = db.Column(db.DateTime, default=func.now(), nullable=False)

    # Kolom untuk menyimpan nama dengan default 'user'
    name = db.Column(db.String(50), default='user', nullable=False)



############################################
#End Point
############################################



# Route utama untuk menguji koneksi
@app.route('/')
def index():
    return jsonify(message="Koneksi ke MySQL berhasil!")

# Route untuk menambah user baru
@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.get_json()
    uuid = data.get('uuid')

    # Cek apakah UUID sudah ada di database
    user_exists = User.query.filter_by(uuid=uuid).first()
    if user_exists:
        return jsonify({"message": "User sudah ada"}), 200

    # Tambah user baru
    new_user = User(uuid=uuid)
    db.session.add(new_user)
    db.session.commit()
    return jsonify(message="User created successfully", uuid=new_user.uuid)

# Route untuk menambah history
# API untuk mengunggah gambar
INPUT_FOLDER = r'D:\Backend_pbl_colormatch\image\input'
OUTPUT_FOLDER = r'D:\Backend_pbl_colormatch\image\output'

if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    print("Received request for /upload_image")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    user_uuid = request.form.get('uuid')
    print(f"UUID: {user_uuid}")

    # Validasi pengguna
    user = User.query.filter_by(uuid=user_uuid).first()
    if not user:
        print("User not found")
        return jsonify({'error': 'User not found'}), 404

    # Simpan gambar input di folder yang ditentukan
    filename = secure_filename(file.filename)
    input_path = os.path.join(INPUT_FOLDER, filename)
    file.save(input_path)

    # Ekstraksi fitur gambar

    skin_tone, color_palette, outImg_rgb, confidence = main(input_path)    
    # cv2.imshow("Output Image", outImg_rgb)  
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    color_palette_str = json.dumps(color_palette)

    # Jika tidak ada wajah terdeteksi, kembalikan error
    if outImg_rgb is None:
        return jsonify({'error': 'No face detected'}), 400

    # if outImg_rgb is None:
    #     return jsonify({'error': 'No face detected'}), 400
    # else :
    #     skin_tone, color_palette = main(input_path)

    # Menghasilkan output filename menggunakan angka acak
    random_number = random.randint(100000, 999999)  # Angka acak antara 100000 hingga 999999
    output_filename = f'output_{random_number}.jpg'  # Menggunakan ekstensi yang sesuai dengan format gambar
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    # Pastikan outImg_rgb adalah dalam format float32 dan berada dalam rentang 0.0 hingga 1.0
    if outImg_rgb.dtype == np.float32:
        # Normalisasi dan konversi ke uint8
        outImg_rgb = (outImg_rgb * 255).astype(np.uint8)

    # Konversi gambar dari BGR (OpenCV) ke RGB (PIL)
    outImg_rgb = cv2.cvtColor(outImg_rgb, cv2.COLOR_BGR2RGB)

    # Menggunakan PIL untuk menyimpan gambar
    image = Image.fromarray(outImg_rgb)  # Mengonversi array NumPy ke objek Image PIL
    image.save(output_path)  # Menyimpan gambar ke path yang ditentukan

    # Simpan hasil ke database
    history = History(
        user_uuid=user_uuid,
        foto_input=input_path,
        foto_output=output_path,
        skin_tone=skin_tone,
        color_palette= color_palette_str,
        confidence = confidence
    )
    db.session.add(history)
    db.session.commit()

    return jsonify({
        'message': 'Image uploaded successfully',
        'skin_tone': skin_tone,
        'color_palette': color_palette,
        'output_image': output_path
    }), 200

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/history/latest/<user_uuid>', methods=['GET'])
def get_latest_history(user_uuid):

    # Validasi pengguna
    user = User.query.filter_by(uuid=user_uuid).first()
    if not user:
        print("User not found")
        return jsonify({'error': 'User not found'}), 404
    print("User found")

    # Ambil riwayat terbaru untuk pengguna
    records = History.query.filter_by(user_uuid=user_uuid).order_by(History.timestamp.desc()).all()
    if not records:
        print("No history record found for this user")
        return jsonify({'error': 'No history found for this user'}), 404

    latest_record = records[0]  # Ambil record paling pertama dari hasil
    print(f"Latest Record: {latest_record}")

    foto_input_filename = os.path.basename(latest_record.foto_input)
    foto_output_filename = os.path.basename(latest_record.foto_output)

    # Format hasil
    latest_history = {
        'foto_input': f"http://192.168.126.94:5000/image/input/{foto_input_filename}",
        'foto_output': f"http://192.168.126.94:5000/image/output/{foto_output_filename}",
        'skin_tone': latest_record.skin_tone,
        'color_palette': latest_record.color_palette,
        'timestamp': latest_record.timestamp,
        'name' : latest_record.name,
        'id' : latest_record.id,
        'confidence' : latest_record.confidence
    }
    return jsonify({'latest_history': latest_history}), 200

@app.route('/history/<user_uuid>', methods=['GET'])
def get_all_history(user_uuid):

    # Validasi pengguna
    user = User.query.filter_by(uuid=user_uuid).first()
    if not user:
        print("User not found")
        return jsonify({'error': 'User not found'}), 404
    print("User found")

    # Ambil semua riwayat untuk pengguna
    records = History.query.filter_by(user_uuid=user_uuid).order_by(History.timestamp.desc()).all()
    if not records:
        print("No history record found for this user")
        return jsonify({'error': 'No history found for this user'}), 404

    # Format hasil dengan looping melalui semua records 
    all_history = []
    
    for record in records:
        history_entry = {
            'foto_input': f"http://192.168.126.94:5000/image/input/{os.path.basename(record.foto_input)}",
            'foto_output': f"http://192.168.126.94:5000/image/output/{os.path.basename(record.foto_output)}",
            'skin_tone': record.skin_tone,
            'color_palette': record.color_palette,
            'timestamp': record.timestamp,
            'name' : record.name,
            'id' : record.id,
            'confidence' : record.confidence
        }
        all_history.append(history_entry)
    return jsonify({'all_history': all_history}), 200


# Endpoint untuk mengambil gambar output
@app.route('/image/output/<filename>', methods=['GET'])
def get_output_image(filename):
    print(f"Fetching output image: {filename}")
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/cek_user/<user_uuid>', methods=['GET'])
def cek_user(user_uuid):
    user = User.query.filter_by(uuid=user_uuid).first()
    if not user:
        print("User not found")
        return jsonify({'error': 'User not found'}), 404
    print("User found")

    return jsonify({
    'uuid': user_uuid  # Ganti dengan atribut yang sesuai
    }), 200


@app.route('/history/edit_name/<int:id>', methods=['PUT'])
def edit_name(id):
    data = request.json
    new_name = data.get('name')

    if not new_name:
        return jsonify({"error": "Name is required"}), 400

    history_record = History.query.get(id)
    if not history_record:
        return jsonify({"error": "History record not found"}), 404

    history_record.name = new_name

    try:
        db.session.commit()
        return jsonify({"message": "Name updated successfully"}), 200
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "Failed to update name"}), 500


@app.route('/history/delete/<int:id>', methods=['DELETE'])
def delete_history(id):
    history_item = History.query.get(id)
    if history_item is None:
        return jsonify({'message': 'History not found'}), 404

    db.session.delete(history_item)
    db.session.commit()
    return jsonify({'message': 'History deleted successfully'}), 200

@app.route('/delete_user/<user_uuid>', methods=['DELETE'])
def delete_user(user_uuid):
    # Validasi pengguna
    user = User.query.filter_by(uuid=user_uuid).first()
    if not user:
        return jsonify({'message': 'User  not found'}), 404

    # Hapus semua history yang terkait dengan user
    History.query.filter_by(user_uuid=user_uuid).delete()

    # Hapus user
    db.session.delete(user)
    db.session.commit()

    return jsonify({'message': 'User  and associated history deleted successfully'}), 200














# Jalankan server dan buat tabel jika belum ada
# Digunakan untuk debug di chrome
# if __name__ == '__main__':
#     # Membuat tabel jika belum ada
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)

if __name__ == '__main__':
    ## membuat tabel jika belum ada
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)

