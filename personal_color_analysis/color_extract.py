import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
from itertools import compress

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters):
        self.CLUSTERS = clusters
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ubah format BGR (OpenCV) ke RGB
        self.IMAGE = img.reshape((img.shape[0] * img.shape[1], 3))  # Flatten gambar menjadi array 1D
        
        # Terapkan K-Means Clustering
        kmeans = KMeans(n_clusters=self.CLUSTERS, random_state=42)
        kmeans.fit(self.IMAGE)

        # Simpan warna dominan dan label pixel
        self.COLORS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_

    def domColor(self):
        # Membuat array label dari 0 hingga jumlah cluster
        numLabels = np.arange(0, self.CLUSTERS + 1)

        # Hitung jumlah pixel untuk setiap cluster
        (hist, _) = np.histogram(self.LABELS, bins=numLabels)

        # Normalisasi histogram agar nilai berada antara 0-1
        hist = hist.astype("float")
        hist /= hist.sum()

        # Urutkan warna berdasarkan frekuensi (dari terbesar ke terkecil)
        colors = self.COLORS
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]

        # Ubah warna menjadi integer
        for i in range(self.CLUSTERS):
            colors[i] = colors[i].astype(int)
        
        # Filter warna untuk mengabaikan warna putih terang dan hitam pekat
        fil = [colors[i][2] < 250 and colors[i][0] > 10 for i in range(self.CLUSTERS)]
        colors = list(compress(colors, fil))

        return colors, hist


    def plotHistogram(self):
        colors, hist = self.domColor()

        # Buat grafik kosong (50x500 pixel)
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # Gambar persegi panjang untuk setiap warna sesuai frekuensi
        for i in range(len(colors)):
            end = start + hist[i] * 500  # Lebar persegi panjang sesuai frekuensi warna
            r, g, b = colors[i]          # Ambil nilai RGB dari warna
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
            start = end

        # Tampilkan grafik menggunakan Matplotlib
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)  
        plt.show()

        return colors

