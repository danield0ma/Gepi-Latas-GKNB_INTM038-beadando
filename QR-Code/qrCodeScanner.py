import cv2
import numpy as np
newWidth = 800

pics = ['qr.jpg', 'qr0.jpg', 'qr1.jpg', 'qr2.jpg', 'qr3.jpg', 'qr4.jpg', 'qr5.jpg', 'qr6.jpg', 'qr7.jpg', 'qr8.jpg', 'qr9.jpg']

for p in pics:
    I = cv2.imread(p)
    newHeight = int((newWidth / I.shape[1]) * I.shape[0])
    I = cv2.resize(I, (newWidth, newHeight))

    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(I_gray, (3,3), 0)

    _, I_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, I_binaris = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    I_binaris = cv2.medianBlur(I_binaris, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    T = cv2.morphologyEx(I_binaris, cv2.MORPH_OPEN, kernel, iterations = 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    J = cv2.morphologyEx(T, cv2.MORPH_CLOSE, kernel, iterations = 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    K = cv2.morphologyEx(J, cv2.MORPH_OPEN, kernel, iterations = 2)

    cnts, h = cv2.findContours(K, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min = 10.0
    cnt = 0
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        perimeter = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(perimeter)
        x, y, w, h, r = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
        ar = w / float(h)

        if abs(1-ar) < min and len(perimeter) >= 4 and ar > 0.9 and ar < 1.1 and area / (newWidth * newHeight) > 0.015:
            min = abs(1-ar)
            cnt += 1

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(I, [box], 0, (0,0,255), 1)

            hh = int(h//2)
            ww = int(w//2)
            y = int(y)
            x = int(x)

            if r % 90 > 0.5:
                M = cv2.getRotationMatrix2D((x, y), r, 1.0)
                rotated = cv2.warpAffine(I_bin, M, (newWidth, newHeight))
                cv2.imshow('rotated ', rotated)
                ROI = rotated[y-hh:y+hh, x-ww:x+ww]
                cv2.imshow('ROI ', ROI)
            else:
                ROI = I_bin[x-ww:x+ww, y-hh:y+hh]
                cv2.imshow('ROI ', ROI)

    if(cnt == 0):
        pass
        #...

    cv2.imshow('T', T)
    cv2.imshow('J', J)
    cv2.imshow('K', K)
    cv2.imshow('szurke ', I_gray)
    cv2.imshow('binaris ', I_binaris)
    cv2.imshow('Kép ', I)

    cv2.waitKey()
    cv2.destroyAllWindows()
    print('########################')
