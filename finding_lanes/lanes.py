import cv2                      # pip install opencv-contrib-python
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    # print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)    # x = (y - b) / m
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    # print(left_fit_average, 'left')
    # print(right_fit_average, 'right')
    # # print(left_fit)
    # print(right_fit)

    return np.array([left_line, right_line])


def canny(image):
    # Convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Take img and make it's low/high (3:1 here) threshold gradient
    return cv2.Canny(blur, 50, 150)


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # print(line)
            # x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2),    # Draw lines on black canvas
                     (255, 0, 0),                       # Blue color
                     10)                                # Thickness of the line
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)         # Black image
    cv2.fillPoly(mask, polygons, 255)   # Fill mask with white triangle
    masked_image = cv2.bitwise_and(image, mask)     # Compare pixels
    return masked_image


# image = cv2.imread('test_image0.jpg')
# lane_image = np.copy(image)
#
# canny_image = canny(lane_image)                   # Sketch picture
# cropped_image = region_of_interest(canny_image)
#
# lines = cv2.HoughLinesP(cropped_image,
#                         2,                      # Distance resolution in pxs
#                         np.pi/180,              # Angle resolution in radians (1 degree = Pi/180)
#                         100,                    # Min number of intersections
#                         np.array([]),           # placeholder array
#                         minLineLength=40,       # Length in pxs that we'll accept to the output
#                         maxLineGap=5)           # Distance between lines that we connect in one
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
#
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow("result", combo_image)               # window name, picture
# cv2.waitKey(0)                                  # 0 = Do not close the window
# plt.imshow(canny)
# plt.show()

cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny(frame)									# Sketch picture
    cropped_image = region_of_interest(canny_image)

    lines = cv2.HoughLinesP(cropped_image,
                            2,									# Distance resolution in pxs
                            np.pi / 180,  						# Angle resolution in radians (1 degree = Pi/180)
                            100,								# Min number of intersections
                            np.array([]),						# placeholder array
                            minLineLength=40,					# Length in pxs that we'll accept to the output
                            maxLineGap=5)						# Distance between lines that we connect in one
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)

    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image) 							# window name, picture
    if cv2.waitKey(1) == ord('q'):    							# if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
