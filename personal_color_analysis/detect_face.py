from imutils import face_utils
import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
from classes import WBsRGB as wb_srgb


class DetectFace:
    def __init__(self, image):
        # Inisialisasi pendeteksi wajah dlib (HOG-based) dan prediktor landmark wajah
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.faceCascade = cv2.CascadeClassifier("prep_model/haarcascade_frontalface_default.xml")

        # Membaca gambar input dan mengubah ukurannya jika terlalu besar
        self.img = cv2.imread(image)
        if self.img.shape[0] > 500:
            self.img = cv2.resize(self.img, dsize=(0, 0), fx=0.8, fy=0.8)
        
        # Variabel untuk menyimpan bagian wajah
        self.right_eyebrow = []
        self.left_eyebrow = []
        self.right_eye = []
        self.left_eye = []
        self.left_cheek = []
        self.right_cheek = []

        self.detect_face_part()

    def detect_face_part(self):
        # Mengonversi gambar ke grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Mendeteksi wajah menggunakan Haar Cascade
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.3, 
                                                  minNeighbors=4, minSize=(25, 25))
        print("Ditemukan {0} wajah!".format(len(faces)))

        # Jika ada wajah yang terdeteksi, pilih wajah terbesar
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            (x, y, w, h) = largest_face
            outImg = self.img[y:y + h + 14, x:x + w + 14]
            self.outImg = outImg.copy()

            # Pastikan gambar memiliki 3 saluran warna (RGB)
            if outImg.ndim == 3 and outImg.shape[2] == 3:
                if outImg.dtype != np.uint8:  # Konversi jika tipe data bukan uint8
                    outImg = (outImg * 255).clip(0, 255).astype(np.uint8)

                # Konversi gambar dari BGR ke RGB
                image_rgb = cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Gambar output memiliki jumlah saluran yang tidak valid.")
        
        # Variabel untuk menyimpan koordinat bagian wajah
        face_parts = [[], [], [], [], [], [], [], []]

        # Mendeteksi wajah pada gambar RGB menggunakan dlib
        rects = self.detector(image_rgb, 1)
        print(f"Jumlah wajah terdeteksi: {len(rects)}")
        
        if len(rects) == 0:
            print("Tidak ada wajah yang terdeteksi pada gambar RGB.") 

        # Menandai titik landmark wajah pada gambar
        for (i, rect) in enumerate(rects):
            shape = self.predictor(image_rgb, rect)
            shape = face_utils.shape_to_np(shape)
            for idx, (x, y) in enumerate(shape):
                cv2.circle(image_rgb, (x, y), 2, (0, 255, 0), -1)  # Menandai titik landmark

        # cv2.imshow("landmark_face", cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Mengambil landmark wajah pertama
        shape = self.predictor(image_rgb, rect)
        shape = face_utils.shape_to_np(shape)

        idx = 0
        # Membagi titik-titik landmark menjadi bagian-bagian wajah
        for ((i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            face_parts[idx] = shape[i:j]
            idx += 1
        face_parts = face_parts[2:6]  # Mengambil alis dan mata
        print(face_utils.FACIAL_LANDMARKS_IDXS)
        print(face_parts[0])

        # Menyimpan bagian-bagian wajah
        self.right_eyebrow = self.extract_face_part(outImg, face_parts[0])
        self.left_eyebrow = self.extract_face_part(outImg, face_parts[1])
        self.right_eye = self.extract_face_part(outImg, face_parts[2])
        self.left_eye = self.extract_face_part(outImg, face_parts[3])

        # cv2.imshow("right_eyebrow", self.right_eyebrow)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Menentukan area pipi berdasarkan posisi landmark wajah
        self.left_cheek = outImg[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]
        self.right_cheek = outImg[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]

    def extract_face_part(self, img, face_part_points):
        # Menghitung kotak pembatas dari titik-titik bagian wajah
        (x, y, w, h) = cv2.boundingRect(face_part_points)
        crop = img[y:y+h, x:x+w]

        # Menyesuaikan titik koordinat bagian wajah
        adj_points = np.array([np.array([p[0]-x, p[1]-y]) for p in face_part_points])

        # Membuat masking untuk memisahkan bagian wajah
        mask = np.zeros((crop.shape[0], crop.shape[1]))
        cv2.fillConvexPoly(mask, adj_points, 1)
        mask = mask.astype(np.bool_)
        crop[np.logical_not(mask)] = [255, 0, 0]

        return crop
