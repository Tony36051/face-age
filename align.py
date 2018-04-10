# -*- coding: utf-8 -*-
import cv2

classifier_xml = ur"D:\python\Lib\site-packages\opencv-data\haarcascade_frontalface_alt2.xml"
face_patterns = cv2.CascadeClassifier(classifier_xml)


def demo_one(img_path):
    sample_image = cv2.imread(img_path)
    faces = face_patterns.detectMultiScale(sample_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(sample_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imwrite('/Users/abel/201612_detected.png', sample_image)
    cv2.imshow("image", sample_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def

if __name__ == '__main__':
    img_path = ur"D:\wiki_crop\00\23300_1962-06-19_2011.jpg"
    demo_one(img_path)