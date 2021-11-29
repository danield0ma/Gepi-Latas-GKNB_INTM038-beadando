import cv2
import numpy as np

newWidth = 800
pics = ['bc.jpg', 'bc1.jpg', 'bc2.jpg', 'bc3.jpg', 'bc4.jpg', 'bc5.jpg', 'bc6.jpg', 'bc7.jpg', 'bc8.jpg']

for p in pics:
    I = cv2.imread(p)
    newHeight = int((newWidth / I.shape[1]) * I.shape[0])
    I = cv2.resize(I, (newWidth, newHeight))

    I_szurke = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    _, I_binaris = cv2.threshold(I_szurke, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    X = cv2.Sobel(I_szurke, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    Y = cv2.Sobel(I_szurke, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

    cv2.imshow('X', X)
    cv2.imshow('Y', Y)

    L = cv2.subtract(X, Y)
    L = cv2.convertScaleAbs(L)
    cv2.imshow('subtract', L)

    L = cv2.blur(L, (9, 9))
    _, L = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', L)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, kernel, iterations = 2)
    cv2.imshow('close', L)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    L = cv2.morphologyEx(L, cv2.MORPH_OPEN, kernel, iterations = 1)
    cv2.imshow('open', L)

    cnts, _ = cv2.findContours(L, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(approx)
        x, y, w, h, r = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]

        ratio = w / float(h)
        if ratio > 0.5:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(I, [box], -1, (0, 255, 0), 1)

    cv2.imshow("Image", I)
    cv2.imshow("Bináris", I_binaris)

    cv2.waitKey()
    cv2.destroyAllWindows()
