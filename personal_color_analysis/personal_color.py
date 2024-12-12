import cv2
import numpy as np
import joblib  
from personal_color_analysis.detect_face import DetectFace
from personal_color_analysis.color_extract import DominantColors
from colormath.color_objects import LabColor, sRGBColor, HSVColor
from colormath.color_conversions import convert_color

# Muat model machine learning
model_filename = 'random_forest_model.pkl'
model = joblib.load(model_filename)

def analysis(imgpath):
    #######################################
    #           Face detection            #
    #######################################
    df = DetectFace(imgpath)
    face = [df.left_cheek, df.right_cheek,
            df.left_eyebrow, df.right_eyebrow,
            df.left_eye, df.right_eye]
    
    
    #######################################
    #         Get Dominant Colors         #
    #######################################
    temp = []
    clusters = 4
    for f in face:
        dc = DominantColors(f, clusters)
        face_part_color, _ = dc.domColor()
        # dc.plotHistogram()

        if face_part_color:  
            temp.append(np.array(face_part_color[0]))
        else:
            print(f"Warning: No color data found for {f}")
            temp.append(np.zeros(3))
        print(f"Current temp: {temp}")


    # for i, color in enumerate(temp):
    #     color_image = np.zeros((100, 300, 3), dtype=np.uint8)  # Kotak 100x100
    #     color_image[:] = color  # Isi kotak dengan warna [R, G, B]
        
    #     # Tampilkan warna dominan
    #     cv2.imshow(f'Color Dominant {i+1}', color_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


    cheek = np.mean([temp[0], temp[1]], axis=0)
    eyebrow = np.mean([temp[2], temp[3]], axis=0)
    eye = np.mean([temp[4], temp[5]], axis=0)

    Lab_b, hsv_s = [], []
    color = [cheek, eyebrow, eye]
    for i in range(3):

        rgb = sRGBColor(color[i][0], color[i][1], color[i][2], is_upscaled=True)
        lab = convert_color(rgb, LabColor, through_rgb_type=sRGBColor)
        hsv = convert_color(rgb, HSVColor, through_rgb_type=sRGBColor)
        Lab_b.append(float(format(lab.lab_b, ".2f")))
        hsv_s.append(float(format(hsv.hsv_s, ".2f")) * 100)

    print('Lab_b[skin, eyebrow, eye]', Lab_b)
    print('hsv_s[skin, eyebrow, eye]', hsv_s)

    #######################################
    #        Predict using ML Model       #
    #######################################
    
    # Gabungkan Lab_b dan hsv_s menjadi array fitur
    features = np.array(Lab_b + hsv_s).reshape(1, -1)  # Reshape untuk model prediksi
    probabilities = model.predict_proba(features)[0]  # Probabilitas prediksi untuk setiap kelas
    tone = model.predict(features)[0]  
    confidence_numeric = np.max(probabilities)  
    confidence = f"{confidence_numeric * 100:.2f}%"  # Ubah ke string format persentase
    print(f"Confidence Level: {confidence}")
    print(f"Predicted Tone: {tone}")

    if tone == 'fall' :
        tone = 'autumn'

    #######################################
    #      Personal color Analysis        #
    #######################################
    def get_color_palette(tone):
        if tone == 'spring':
            return [
                '#faf0b9', '#72c161', '#f8af57', '#f5a7a6', '#ad549e',
                '#f1d0a8', '#cde07b', '#f59575', '#e7617a', '#f6ad56',
                '#fae46f', '#5ea144', '#b5dcb7', '#f597aa', '#3ec5e6',
                '#8e6d44', '#a2aa37', '#91c86f', '#f7d2d2', '#3f8cad'
            ]
        elif tone == 'autumn':
            return [
                '#f6ecd3', '#279e5d', '#eda852', '#bc2d26', '#73396b',
                '#ddbf9a', '#9e5825', '#e08d2d', '#b1485d', '#dc6f26',
                '#ebd464', '#4a7c3a', '#b5702e', '#f07b90', '#238ab1',
                '#66371a', '#808334', '#6dab83', '#dd9698', '#00577c'
            ]
        elif tone == 'summer':
            return [
                '#ffffff', '#b873b0', '#6481be', '#83cdbb', '#daa9ce',
                '#b7b7bb', '#cfa6ce', '#66cae9', '#bb3f6d', '#6c68a8',
                '#9e8585', '#9e85bd', '#6188c5', '#e56aa5', '#4dc2ce',
                '#000000', '#b7b2d8', '#a896c8', '#c760a1', '#80b6af'
            ]
        elif tone == 'winter':
            return [
                '#ffffff', '#a5479b', '#29368c', '#90d2c2', '#f0d2e5',
                '#a1a0a2', '#ba87bc', '#3ac1f0', '#a41f4d', '#5757a5',
                '#655554', '#f6f29d', '#416fb7', '#d42a54', '#a41f4d',
                '#000000', '#b7b2d8', '#675aa7', '#c3258a', '#49ba79'
            ]
        else:
            return []

    # Get color palette based on the tone
    print('{} Color tone adalah {}.'.format(imgpath, tone))
    color_palette = get_color_palette(tone)

    return tone, color_palette, df.outImg, confidence
