from personal_color_analysis import personal_color
import cv2

def main(img_path):
    skin_tone, color_palette, imgOut_rgb, confidence = personal_color.analysis(img_path)
    # cv2.imshow("Output Image", outImg_rgb)  # "Output Image" adalah nama jendela
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return skin_tone, color_palette, imgOut_rgb, confidence
    # return skin_tone, color_palette

