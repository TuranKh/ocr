from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import time
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def box_extractor(scores, geometry, min_confidence):

    num_rows, num_cols = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            box_h = x_data0[x] + x_data2[x]
            box_w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y + (cos * x_data2[x]) - (sin * x_data1[x]))
            start_x = int(end_x - box_w)
            start_y = int(end_y - box_h)

            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rectangles, confidences



if __name__ == '__main__':

    w, h = None, None
    confidence = 0.5
    frame_width, frame_height = 320, 320
    ratio_w, ratio_h = None, None

    layer_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    selected_net = "frozen_east_text_detection.pb"
    net = cv2.dnn.readNet(selected_net)

    vs = VideoStream(src=0).start()
    time.sleep(1)

    while True:

        frame = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=1000)
        orig = frame.copy()
        orig_h, orig_w = orig.shape[:2]

        if w is None or h is None:
            h, w = frame.shape[:2]
            ratio_w = w / float(frame_width)
            ratio_h = h / float(frame_height)

        frame = cv2.resize(frame, (frame_width, frame_height))

        blob = cv2.dnn.blobFromImage(frame, 1.0, (frame_width, frame_height), (123.68, 116.78, 103.94),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        scores, geometry = net.forward(layer_names)

        rectangles, confidences = box_extractor(scores, geometry, min_confidence=confidence)
        boxes = non_max_suppression(np.array(rectangles), probs=confidences)

        for (start_x, start_y, end_x, end_y) in boxes:

            start_x = int(start_x * ratio_w)
            start_y = int(start_y * ratio_h)
            end_x = int(end_x * ratio_w)
            end_y = int(end_y * ratio_h)    
            dx = int((end_x - start_x) * 0.2)
            dy = int((end_y - start_y) * 0.75)

            start_x = max(0, start_x - dx)
            start_y = max(0, start_y - dy)
            end_x = min(orig_w, end_x + (dx * 2))
            end_y = min(orig_h, end_y + (dy * 2))

            # ROI to be recognized
            roi = orig[start_y:end_y, start_x:end_x]

            blurred_roi = cv2.GaussianBlur(roi, (201, 201), 0)
            orig[start_y:end_y, start_x:end_x] = blurred_roi


        cv2.imshow("Detection", orig)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break


    if not args.get('video', False):
        vs.stop()

    cv2.destroyAllWindows()
