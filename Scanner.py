import numpy as np
import cv2
import sys
from collections import defaultdict
import copy

def transform(array, upper):
    temp = np.resize(array, (4, 2))
    if upper:
        temp[[0, 1, 2]] = temp[[2, 0, 1]]
    else:
        temp[[1, 2, 3]] = temp[[2, 3, 1]]
    return temp

def corner(black):
    w = black.shape[0]
    h = black.shape[1]
    R = cv2.cornerHarris(black.astype(np.uint8), 5, 5, 0.1)

    points = []
    hatar = R.max() * 0.5
    for x in range(black.shape[0]):
        for y in range(black.shape[1]):
            if R[x][y] > hatar:
                points.append([y, x])

    if len(points) != 0:
        dist = np.zeros([6, len(points)])
        for i in range(6):
            if i == 0: x = y = 0
            elif i == 1:
                x = 0
                y = h
            elif i == 2:
                x = w
                y = 0
            elif i == 3:
                x = w // 3
                y = 0
            elif i == 4:
                x = w // 3
                y = h
            else:
                x = w
                y = h // 3
            for n, j in enumerate(points):
                dist[i][n] = ((x - j[0]) ** 2 + (y - j[1]) ** 2) ** 0.5

        b_f = points[dist[0][:].argmin()]
        b_a = points[dist[1][:].argmin()]
        j_f = points[dist[2][:].argmin()]
        return b_f, b_a, j_f

    else: return [0, 0], [0, 0], [0, 0]

def hough(ROI):
    w = ROI.shape[1]
    h = ROI.shape[0]

    canny = cv2.Canny(ROI, 200, 250)
    blurred = cv2.blur(canny, (3, 3))

    L = cv2.HoughLines(blurred, 1, np.pi/180, int(w / 2 * 0.8))
    horizontal = []
    vertical = []
    intersections = []
    for i in range(0, len(L)):
        rho = L[i][0][0]
        theta = L[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        o = 3
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        if (theta < np.pi/(o*16) or theta > (np.pi - np.pi/(o*16))) and (np.abs(rho) < int(w*0.3) or np.abs(rho) > int(w*0.7)):
            vertical.append(np.array([rho, theta]))
        elif theta > np.pi/2 - np.pi/(o*16) and theta < np.pi/2 + np.pi/(o*16) and (np.abs(rho) < int(h*0.3) or np.abs(rho) > int(h*0.7)):
            horizontal.append(np.array([rho, theta]))

    for line1 in vertical:
        for line2 in horizontal:
            rho1, theta1 = line1
            rho2, theta2 = line2
            A = np.array([[np.cos(theta1), np.sin(theta1)],
                          [np.cos(theta2), np.sin(theta2)]
                         ])
            b = np.array([rho1, rho2])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            intersections.append([x0, y0])

    if len(intersections) != 0:
        dist = np.zeros([4, len(intersections)])
        for i in range(4):
            if i == 0: x = y = 0
            elif i == 1:
                x = 0
                y = h - 1
            elif i == 2:
                x = w - 1
                y = h - 1
            else:
                x = w - 1
                y = 0
            for n, j in enumerate(intersections):
                dist[i][n] = (((x - j[0]) ** 2) + ((y - j[1]) ** 2)) ** 0.5

        b_f = intersections[dist[0].argmin()]
        b_a = intersections[dist[1].argmin()]
        j_a = intersections[dist[2].argmin()]
        j_f = intersections[dist[3].argmin()]

        return b_f, b_a, j_a, j_f

    else: return [0, 0], [0, 0], [0, 0], [0, 0]

def clear(img, size):
    width, height = img.shape
    img[0:size, 0:height] = 255
    img[0:width, 0:size] = 255
    img[0:width, height-size:height] = 255
    img[width-size:width, 0:height] = 255
    return img

def calc(c, width, height):
    epsilon = 0.01 * cv2.arcLength(c, True)
    perimeter = cv2.approxPolyDP(c, epsilon, True)
    area = cv2.contourArea(c)
    rect = cv2.minAreaRect(perimeter)
    x, y, w, h, r = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    if h == 0: h = 1
    ratio = w / float(h)
    areaRatio = area / float(width * height)
    return perimeter, ratio, areaRatio, x, y, w, h, r, rect

def accuracyCounter(img):
    minta = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 255, 255, 255, 255, 255, 0],
                      [0, 255, 0, 0, 0, 255, 0],
                      [0, 255, 0, 0, 0, 255, 0],
                      [0, 255, 0, 0, 0, 255, 0],
                      [0, 255, 255, 255, 255, 255, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                     ])
    m = np.array([255, 255, 255, 255, 255, 255, 255])

    p = np.array([0, 255, 0, 255, 0])

    cnt = 0
    cnt += np.sum(np.abs(img[0:7, 0:7] - minta) / 255)
    cnt += np.sum(np.abs(img[14:21, 0:7] - minta) / 255)
    cnt += np.sum(np.abs(img[0:7, 14:21] - minta) / 255)

    cnt += np.sum(np.abs(img[0:7, 7] - m) / 255)
    cnt += np.sum(np.abs(img[14:21, 7] - m) / 255)
    cnt += np.sum(np.abs(img[0:7, 13] - m) / 255)

    cnt += np.sum(np.abs(img[7, 0:7] - m.T) / 255)
    cnt += np.sum(np.abs(img[13, 0:7] - m.T) / 255)
    cnt += np.sum(np.abs(img[7, 14:21] - m.T) / 255)

    cnt += np.sum(np.abs(img[7, 7] - 255) / 255)
    cnt += np.sum(np.abs(img[13, 7] - 255) / 255)
    cnt += np.sum(np.abs(img[7, 13] - 255) / 255)

    cnt += np.sum(np.abs(img[8:13, 6] - p) / 255)
    cnt += np.sum(np.abs(img[6, 8:13] - p.T) / 255)

    cnt += np.sum(np.abs(img[8, 13] - 0) / 255)

    return cnt

def unfold(code):
    numbers = np.zeros((12,7))
    numbers[0] = [int(i) for i in code[3:10]]
    numbers[1] = [int(i) for i in code[10:17]]
    numbers[2] = [int(i) for i in code[17:24]]
    numbers[3] = [int(i) for i in code[24:31]]
    numbers[4] = [int(i) for i in code[31:38]]
    numbers[5] = [int(i) for i in code[38:45]]
    numbers[6] = [int(i) for i in code[50:57]]
    numbers[7] = [int(i) for i in code[57:64]]
    numbers[8] = [int(i) for i in code[64:71]]
    numbers[9] = [int(i) for i in code[71:78]]
    numbers[10] = [int(i) for i in code[78:85]]
    numbers[11] = [int(i) for i in code[85:92]]
    return numbers

def check(code):
    num = unfold(code)
    ok = np.array((int(np.sum(num[0:6, 0])), int(np.sum(num[0:6, -1])), int(np.sum(num[6:12, 0])), int(np.sum(num[6:12, -1]))))
    left = np.sum(np.array((np.sum(num[0]) % 2 == 0, np.sum(num[1]) % 2 == 0, np.sum(num[2]) % 2 == 0, np.sum(num[3]) % 2 == 0, np.sum(num[4]) % 2 == 0, np.sum(num[5]) % 2 == 0)))
    right = np.sum(np.array((np.sum(num[6]) % 2 == 0, np.sum(num[7]) % 2 == 0, np.sum(num[8]) % 2 == 0, np.sum(num[9]) % 2 == 0, np.sum(num[10]) % 2 == 0, np.sum(num[11]) % 2 == 0)))

    if (ok == np.array((0, 6, 6, 0))).all() and right == 6:
        return 0
    elif (ok == np.array((0, 6, 6, 0))).all() and left == 6:
        return 1
    else: return -1

def find(img, I, I_binaris, newWidth, newHeight):
    k = 15
    ok = False
    d = 10
    ROI = np.ones((100, 100)) * 255
    C = np.ones((100, 100, 3)) * 255
    while k >= 3 and not ok:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        L = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        L = cv2.morphologyEx(L, cv2.MORPH_OPEN, kernel, iterations = 1)

        cnts, _ = cv2.findContours(L, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=lambda x:cv2.contourArea(x))

        for c in cnts:
            epsilon = 0.01 * cv2.arcLength(c, True)
            perimeter = cv2.approxPolyDP(c, epsilon, True)
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(perimeter)
            x, y, w, h, r = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
            if r > 70:
                w, h = h, w
            areaRatio = area / (newHeight * newWidth)
            if h == 0: h = 1
            ratio = w / float(h)
            if ratio > 1.2 and ratio < 2.4 and areaRatio > 0.05 and areaRatio < 0.35 and len(perimeter) >= 4 and len(perimeter) < 20:
                ok = True
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(I, [box], -1, (0, 255, 0), 1)

                min = abs(1-ratio)
                hh = int(h//2)
                ww = int(w//2)
                y = int(y)
                x = int(x)

                if r > 1 and r < 88:
                    if r > 45:
                        r = (90 - r) * -1
                    M = cv2.getRotationMatrix2D((x, y), r, 1.0)
                    rotated = cv2.warpAffine(I_binaris, M, (newWidth, newHeight))
                    I_rotated = cv2.warpAffine(I, M, (newWidth, newHeight))
                    ROI = rotated[y-hh-d:y+hh+d, x-ww-d:x+ww+d]
                    C = I_rotated[y-hh-d:y+hh+d, x-ww-d:x+ww+d]
                else:
                    ROI = copy.deepcopy(I_binaris[y-hh-d:y+hh+d, x-ww-d:x+ww+d])
                    C = copy.deepcopy(I[y-hh-d:y+hh+d, x-ww-d:x+ww+d])
                ROI = clear(ROI, 5)
        k -= 2
    return ROI, C, ok

def decode(img, I):
    n = 0
    cnt = 0
    result = ''
    numbers = defaultdict(int)
    while n < img.shape[0]:
        row = img[n][:]
        row = 255 - row
        row = np.trim_zeros(row)
        if len(row) == 0:
            n += 1
            continue
        row = cv2.resize(row, (1, 95))
        _, row = cv2.threshold(row, 150, 255, cv2.THRESH_BINARY)
        row = row / 255
        row = np.reshape(row, (95))
        r = ''
        for e in row:
            r += str(int(e))
        if r[0:3] == '101' and r[92:95] == '101' and r[45:50] == '01010':
            ok = check(r)
            if ok == 0:
                numbers[r] += 1
                cnt += 1
            elif ok == 1:
                numbers[r[::-1]] += 1
                cnt += 1
        n += 1

    if len(numbers) != 0:
        numbers = sorted(numbers.items(), key=lambda x:x[1], reverse=True)
        code = numbers[0][0]
        num = unfold(code)

        L = ['0001101', '0011001', '0010011', '0111101', '0100011', '0110001', '0101111', '0111011', '0110111', '0001011']
        G = ['0100111', '0110011', '0011011', '0100001', '0011101', '0111001', '0000101', '0010001', '0001001', '0010111']
        R = ['1110010', '1100110', '1101100', '1000010', '1011100', '1001110', '1010000', '1000100', '1001000', '1110100']
        S = {'LLLLLL': 0, 'LLGLGG': 1, 'LLGGLG': 2, 'LLGGGL': 3, 'LGLLGG': 4, 'LGGLLG': 5, 'LGGGLL': 6, 'LGLGLG': 7, 'LGLGGL': 8, 'LGGLGL': 9}

        scheme = ''
        for i in range(6):
            for j in range(10):
                st = ''
                for e in num[i]:
                    st += str(int(e))
                if st == L[j]:
                    result += str(j)
                    scheme += 'L'
                    continue
                elif st == G[j]:
                    result += str(j)
                    scheme += 'G'
                    continue

        result = str(S[scheme]) + result

        for i in range(6, 12):
            for j in range(10):
                st = ''
                for e in num[i]:
                    st += str(int(e))
                if st == R[j]:
                    result += str(j)
                    continue

        cv2.imshow('Kep', I)
        print('RESULT: ', result)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return result

def QR(pics):
    for p in pics:
        newWidth = 950
        I = cv2.imread(p)
        if newWidth < I.shape[1]:
            newHeight = int((newWidth / I.shape[1]) * I.shape[0])
            I = cv2.resize(I, (newWidth, newHeight))
        else:
            newHeight = I.shape[1]
            newWidth = I.shape[0]

        print('\n################', p, ' ', newWidth, 'x', newHeight, '\n')

        I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(I_gray, (3,3), 0)

        _, I_bin = cv2.threshold(I_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        I_binaris = 255 - I_bin
        I_binaris = cv2.medianBlur(I_binaris, 3)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        T = cv2.morphologyEx(I_binaris, cv2.MORPH_OPEN, kernel, iterations = 1)

        ok = False
        k = 5
        while not ok and k <= 25:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            J = cv2.morphologyEx(T, cv2.MORPH_CLOSE, kernel, iterations = 2)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            K = cv2.morphologyEx(J, cv2.MORPH_OPEN, kernel, iterations = 2)

            contours, _ = cv2.findContours(K, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            min = 10.0
            d = 15
            for c in contours:
                perimeter, ratio, areaRatio, x, y, w, h, r, rect = calc(c, newWidth, newHeight)
                if abs(1-ratio) < min and ratio > 0.85 and ratio < 1.25 and areaRatio > 0.0125 and areaRatio < 0.3 and perimeter.shape[0] >= 4 and perimeter.shape[0] < 20:
                    ok = True

                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(I, [box], 0, (0,0,255), 2)

                    min = abs(1-ratio)
                    hh = int(h//2)
                    ww = int(w//2)
                    y = int(y)
                    x = int(x)

                    ROI = np.ones([100, 100]) * 255
                    C = np.ones([100, 100, 3]) * 255
                    if r % 90 > 1:
                        M = cv2.getRotationMatrix2D((x, y), r, 1.0)
                        rotated = cv2.warpAffine(I_bin, M, (newWidth, newHeight))
                        I_rotated = cv2.warpAffine(I, M, (newWidth, newHeight))
                        ROI = rotated[y-hh-d:y+hh+d, x-ww-d:x+ww+d]
                        C = I_rotated[y-hh-d:y+hh+d, x-ww-d:x+ww+d]
                    else:
                        ROI = I_bin[y-hh-d:y+hh+d, x-ww-d:x+ww+d]
                        C = I[y-hh-d:y+hh+d, x-ww-d:x+ww+d]
                    ROI = clear(ROI, 10)
            k += 2

        if not ok:
            print('HIBA: Detektálás nem sikerült')
            continue

        szel = ROI.shape[0]
        mag = ROI.shape[1]

        contours, _ = cv2.findContours(ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        l_u = l_l = r_l = r_u = False
        rectangles = []
        for c in contours:
            perimeter, ratio, areaRatio, x, y, w, h, r, rect = calc(c, szel, mag)
            if (np.abs(r) < 10 or np.abs(r) > 80) and perimeter.shape[0] >= 4 and perimeter.shape[0] < 20:
                if ratio > 0.75 and ratio < 1.25 and areaRatio > 0.01 and areaRatio < 0.2:
                    rectangles.append(rect)

        rectangles = sorted(rectangles, key=lambda x: x[1][1], reverse=True)
        rectangles = rectangles[:3]
        black = np.zeros(ROI.shape)
        for c in rectangles:
            if c[0][0] < mag / 2 and c[0][1] < szel / 2: l_u = True
            elif c[0][0] < mag / 2 and c[0][1] > szel - szel / 2: l_l = True
            elif c[0][0] > mag - mag / 2 and c[0][1] < szel / 2: r_u = True
            elif c[0][0] > mag - mag / 2 and c[0][1] > szel - szel / 2: r_l = True
            box = cv2.boxPoints(c)
            box = np.int0(box)
            cv2.drawContours(black, [box], 0, (255, 0, 0), 1)

        if l_u + l_l + r_l + r_u == 3 and r_l == True:
            if l_u == False: r = 180
            elif l_l == False: r = 90
            elif r_u == False: r = 270
            x = mag // 2
            y = szel // 2
            M = cv2.getRotationMatrix2D((x, y), r, 1.0)
            ROI = cv2.warpAffine(ROI, M, (mag, szel), borderValue=255)
            C = cv2.warpAffine(C, M, (mag, szel), borderValue=(255, 255, 255))
            black = cv2.warpAffine(black, M, (mag, szel), borderValue=0)
            r_l = False
            l_u = l_l = r_u = True

        b_f, b_a, j_f = corner(black)
        d = b_a[0]-b_f[0]

        src = np.array([b_f, j_f, b_a]).astype(np.float32)
        dst = np.array([b_f, j_f, [b_a[0]-d, b_a[1]]]).astype(np.float32)
        warp = cv2.getAffineTransform(src, dst)
        ROI = cv2.warpAffine(ROI, warp, (mag, szel), borderValue=255)
        C = cv2.warpAffine(C, warp, (mag, szel), borderValue=(255, 255, 255))
        black = cv2.warpAffine(black, warp, (mag, szel), borderValue=0)

        b_f, b_a, j_f = corner(black)

        d = j_f[1]-b_f[1]

        src = np.array([b_f, b_a, j_f]).astype(np.float32)
        dst = np.array([b_f, b_a, [j_f[0], j_f[1]-d]]).astype(np.float32)
        warp = cv2.getAffineTransform(src, dst)
        ROI = cv2.warpAffine(ROI, warp, (mag, szel), borderValue=255)
        C = cv2.warpAffine(C, warp, (mag, szel), borderValue=(255, 255, 255))
        black = cv2.warpAffine(black, warp, (mag, szel), borderValue=0)

        b_f, b_a, j_f = corner(black)

        sz = j_f[0] - b_f[0] + 50
        m = b_a[1] - b_f[1] + 50

        src = np.array([b_f, b_a, j_f]).astype(np.float32)
        dst = np.array([[25, 25], [25, m-25], [sz-25, 25]]).astype(np.float32)
        warp = cv2.getAffineTransform(src, dst)
        ROI = cv2.warpAffine(ROI, warp, (sz, m), borderValue=255)
        C = cv2.warpAffine(C, warp, (sz, m), borderValue=(255, 255, 255))
        black = cv2.warpAffine(black, warp, (sz, m), borderValue=0)
        ROI = clear(ROI, 10)
        b_f, b_a, j_f = corner(black)

        if b_a == [0, 0] or j_f == [0, 0] or (j_f[0] - b_f[0]) == 0:
            print('HIBA: A sarkak detektálása nem sikerült')
            continue

        arctan = np.arctan((j_f[1] - b_f[1]) / (j_f[0] - b_f[0]))
        angle = arctan * (180 / np.pi)

        M = cv2.getRotationMatrix2D((szel // 2, mag // 2), angle, 1.0)
        ROI = cv2.warpAffine(ROI, M, (sz, m), borderValue=255)
        b_f, b_a, j_f = corner(black)

        bf, ba, ja, jf = hough(ROI)
        if b_a != [0, 0] and ba != [0, 0]:
            src = np.array([b_f, b_a, ja, j_f]).astype(np.float32)
            dst = np.array([[0, 0], [0, j_f[0]-b_f[0]], [j_f[0]-b_f[0], j_f[0]-b_f[0]], [j_f[0]-b_f[0], 0]]).astype(np.float32)
            warp = cv2.getPerspectiveTransform(src, dst)
            squareQR = cv2.warpPerspective(ROI, warp, (j_f[0]-b_f[0], j_f[0]-b_f[0]))

            squareQR = cv2.resize(squareQR, (21, 21))
            _, squareQR = cv2.threshold(squareQR, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            src = np.array([bf, ba, ja, jf]).astype(np.float32)
            dst = np.array([[0, 0], [0, jf[0]-bf[0]], [jf[0]-bf[0], jf[0]-bf[0]], [jf[0]-bf[0], 0]]).astype(np.float32)
            warp = cv2.getPerspectiveTransform(src, dst)
            houghQR = cv2.warpPerspective(ROI, warp, (jf[0]-bf[0], jf[0]-bf[0]))

            houghQR = cv2.resize(houghQR, (21, 21))
            _, houghQR = cv2.threshold(houghQR, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        else:
            print('HIBA: Sarkak detektálása nem sikerült!')
            continue

        sq = accuracyCounter(squareQR)
        h = accuracyCounter(houghQR)

        if sq < h:
            QR = copy.deepcopy(squareQR)
        else : QR = copy.deepcopy(houghQR)

        QR = QR / 255

        mask1 = np.flip(QR[16:19, 8].T)
        mask2 = QR[8, 2:5]
        if not (mask1 == mask2).all():
            print('HIBA: nem egyeznek a maszkbitek')
            continue

        mask = np.zeros([21, 21])
        if (mask1 == np.array([0., 0., 0.])).all():
            for i in range(21):
                for j in range(21):
                    if j % 3 == 0:
                        mask[i][j] = 1

        elif (mask1 == np.array([0., 0., 1.])).all():
            for i in range(21):
                for j in range(21):
                    if (i + j) % 3 == 0:
                        mask[i][j] = 1

        elif (mask1 == np.array([0., 1., 0.])).all():
            for i in range(21):
                for j in range(21):
                    if (i + j) % 2 == 0:
                        mask[i][j] = 1

        elif (mask1 == np.array([0., 1., 1.])).all():
            for i in range(21):
                if i % 2 == 0:
                    mask[i][:] = 1

        elif (mask1 == np.array([1., 0., 0.])).all():
            for i in range(21):
                for j in range(21):
                    if ((i * j) % 3 + i * j) % 2 == 0:
                        mask[i][j] = 1

        elif (mask1 == np.array([1., 0., 1.])).all():
            for i in range(21):
                for j in range(21):
                    if ((i * j) % 3 + i + j) % 2 == 0:
                        mask[i][j] = 1

        elif (mask1 == np.array([1., 1., 0.])).all():
            for i in range(21):
                for j in range(21):
                    if (i // 2 + j // 3) % 2 == 0:
                        mask[i][j] = 1

        elif (mask1 == np.array([1., 1., 1.])).all():
            for i in range(21):
                for j in range(21):
                    if (i * j) % 2 + (i * j) % 3 == 0:
                        mask[i][j] = 1

        elif (mask1 == np.array([1., 1., 1.])).all():
            for i in range(21):
                for j in range(21):
                    if (i * j) % 2 + (i * j) % 3 == 0:
                        mask[i][j] = 1

        else:
            print('HIBA: Ismeretlen maszk!')
            continue

        data = np.ones([21, 21])
        data[9:13, 1:7] = QR[9:13, 0:6]
        data[1:7, 9:13] = QR[0:6, 9:13]
        data[7:9, 9:13] = QR[7:9, 9:13]
        data[9:13, 7:21] = QR[9:13, 7:21]
        data[13:21, 9:21] = QR[13:21, 9:21]

        data = np.where(mask==1, 1-data, data)
        data[0:9, 0:9] = data[0:9, 13:21] = data[13:21, 0:9] = data[0, :] = data[:, 0] = 255

        letters = np.zeros([26, 4, 2])
        letters[1] = data[11:15, 19:21]
        letters[2] = transform(data[9:11, 17:21], True)
        letters[3] = np.flip(data[11:15, 17:19], axis=0)
        letters[4] = np.flip(data[15:19, 17:19], axis=0)
        letters[5] = transform(data[19:21, 15:19], False)
        letters[6] = data[15:19, 15:17]
        letters[7] = data[11:15, 15:17]
        letters[8] = transform(data[9:11, 13:17], True)
        letters[9] = np.flip(data[11:15, 13:15], axis=0)
        letters[10] = np.flip(data[15:19, 13:15], axis=0)
        letters[11] = transform(data[19:21, 11:15], False)
        letters[12] = data[15:19, 11:13]
        letters[13] = data[11:15, 11:13]
        letters[14] = data[7:11, 11:13]
        letters[15] = data[3:7, 11:13]
        letters[16] = transform(data[1:3, 9:13], True)
        letters[17] = np.flip(data[3:7, 9:11], axis=0)
        letters[18] = np.flip(data[7:11, 9:11], axis=0)
        letters[19] = np.flip(data[11:15, 9:11], axis=0)
        letters[20] = np.flip(data[15:19, 9:11], axis=0)
        letters[21] = transform(np.reshape(np.append(data[19:21, 9:11], data[11:13, 7:9]), (4, 2)), False)
        letters[22] = transform(data[9:11, 5:9], True)
        letters[23] = transform(data[11:13, 3:7], False)
        letters[24] = transform(data[9:11, 1:5], True)

        enc = int(15 - (8 * data[20, 20] + 4 * data[20, 19] + 2 * data[19, 20] + 1 * data[19, 19]))
        print('Encoding: ', enc)

        len = data[15:19, 19:21]
        length = int(255 - (128 * len[3, 1] + 64 * len[3, 0] + 32 * len[2, 1] + 16 * len[2, 0] + 8 * len[1, 1] + 4 * len[1, 0] + 2 * len[0, 1] + 1 * len[0, 0]))
        print('Hossz: ', length)

        if length > 24:
            print('HIBA: Hossz rosszul lettdetektálva')
            continue

        if not np.sum(np.reshape(letters[length+1], (8, 1))[4:]) == 4:
            print('HIBA: EOM nincs meg')

        letters = letters[:length+1]
        result = ''
        for i in range(1, length+1):
            l = letters[i]
            n = int(255 - (128 * l[3, 1] + 64 * l[3, 0] + 32 * l[2, 1] + 16 * l[2, 0] + 8 * l[1, 1] + 4 * l[1, 0] + 2 * l[0, 1] + 1 * l[0, 0]))
            print('\tASCII kód: ', n)
            result += chr(n)

        print('RESULT: ', result)
        cv2.imshow('Kep', I)

        cv2.waitKey()
        cv2.destroyAllWindows()

def BC(pics):
    for p in pics:
        newWidth = 950
        I = cv2.imread(p)
        if newWidth < I.shape[1]:
            newHeight = int((newWidth / I.shape[1]) * I.shape[0])
            I = cv2.resize(I, (newWidth, newHeight))
        else:
            newHeight = I.shape[1]
            newWidth = I.shape[0]
        print('\n################', p, ' ', newWidth, 'x', newHeight, '\n')

        I_szurke = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        _, I_binaris = cv2.threshold(I_szurke, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        X = cv2.Sobel(I_szurke, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
        Y = cv2.Sobel(I_szurke, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

        L = cv2.subtract(X, Y)
        L = cv2.convertScaleAbs(L)

        L = cv2.blur(L, (9, 9))
        _, L = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ROI, C, ok = find(L, I, I_binaris, newWidth, newHeight)

        if not ok:
            M = cv2.getRotationMatrix2D((newHeight // 2, newWidth // 2), 90, 1.0)
            L = cv2.warpAffine(L, M, (newHeight, newWidth))
            I = cv2.warpAffine(I, M, (newHeight, newWidth))
            I_binaris = cv2.warpAffine(I_binaris, M, (newHeight, newWidth))
            ROI, C, ok = find(L, I, I_binaris, newWidth, newHeight)
            if not ok:
                print('HIBA: Detektálás nem sikerült')
                continue

        result = decode(ROI, I)
        if len(result) != 0:
            continue
        else:
            width = ROI.shape[1]
            height = ROI.shape[0]

            c = cv2.Canny(ROI, 250, 200)
            b = cv2.blur(c, (5, 5))

            horizontal = []
            vertical = []
            intersections = []
            lines = cv2.HoughLines(b, 1, np.pi/180, int(height // 2 * 0.8))
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                o = 4
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                if (theta < np.pi/(o*16) or theta > (np.pi - np.pi/(o*16))):
                    cv2.line(C, pt1, pt2, (255, 0, 0), 1)
                    vertical.append(np.array([rho, theta]))
                elif theta > np.pi/2 - np.pi/(o*16) and theta < np.pi/2 + np.pi/(o*16) and (np.abs(rho) < int(height*0.3) or np.abs(rho) > int(height*0.7)):
                    cv2.line(C, pt1, pt2, (0, 0, 255), 1)
                    horizontal.append(np.array([rho, theta]))

            for line1 in vertical:
                for line2 in horizontal:
                    rho1, theta1 = line1
                    rho2, theta2 = line2
                    A = np.array([[np.cos(theta1), np.sin(theta1)],
                                  [np.cos(theta2), np.sin(theta2)]
                                 ])
                    b = np.array([rho1, rho2])
                    x0, y0 = np.linalg.solve(A, b)
                    x0, y0 = int(np.round(x0)), int(np.round(y0))
                    intersections.append([x0, y0])

            if len(intersections) != 0:
                dist = np.zeros((4, len(intersections)))
                for i in range(4):
                    if i == 0: x = y = 0
                    elif i == 1:
                        x = 0
                        y = height - 1
                    elif i == 2:
                        x = width - 1
                        y = height - 1
                    else:
                        x = width - 1
                        y = 0
                    for n, j in enumerate(intersections):
                        dist[i][n] = (((x - j[0]) ** 2) + ((y - j[1]) ** 2)) ** 0.5

                b_f = intersections[dist[0].argmin()]
                b_a = intersections[dist[1].argmin()]
                j_a = intersections[dist[2].argmin()]
                j_f = intersections[dist[3].argmin()]

            else:
                print('HIBA: Hough nem talált egyeneseket')
                continue

            w = j_f[0] - b_f[0] + 20
            h = b_a[1] - b_f[1] + 5
            src = np.array([[b_f[0]-10, b_f[1]-5], [b_a[0]-10, b_a[1]], [j_a[0]+10, j_a[1]], [j_f[0]+10, j_f[1]-5]]).astype(np.float32)
            dst = np.array([[0, 0], [0, h], [w, h], [w, 0]]).astype(np.float32)
            warp = cv2.getPerspectiveTransform(src, dst)
            barcode = cv2.warpPerspective(ROI, warp, (w, h), borderValue=255)
            D = cv2.warpPerspective(C, warp, (w, h))

            width = barcode.shape[1]
            height = barcode.shape[0]

            result = decode(barcode, I)
            if len(result) != 0:
                continue
            else:
                H = cv2.cornerHarris(barcode, 5, 5, 0.1)

                points = []
                hatar = H.max() * 0.5
                for x in range(barcode.shape[0]):
                    for y in range(barcode.shape[1]):
                        if H[x][y] > hatar:
                            points.append([y, x])

                if len(points) != 0:
                    dist = np.zeros((4, len(points)))
                    for i in range(4):
                        if i == 0:
                            x = 0
                            y = 0
                        elif i == 1:
                            x = 0
                            y = height // 10 * 9
                        elif i == 2:
                            x = width
                            y = 0
                        else:
                            x = width
                            y = height // 10 * 9
                        for n, j in enumerate(points):
                            dist[i][n] = ((x - j[0]) ** 2 + (y - j[1]) ** 2) ** 0.5

                    b_f = points[dist[0][:].argmin()]
                    b_a = points[dist[1][:].argmin()]
                    j_f = points[dist[2][:].argmin()]
                    j_a = points[dist[3][:].argmin()]

                    szel = j_f[0] - b_f[0] + 10
                    mag = b_a[1] - b_f[1]
                    src = np.array([[b_f[0]-5, b_f[1]], [b_a[0]-5, b_a[1]], [j_a[0]+5, j_a[1]], [j_f[0]+5, j_f[1]]]).astype(np.float32)
                    dst = np.array([[0, 0], [0, mag], [szel, mag], [szel, 0]]).astype(np.float32)
                    warp = cv2.getPerspectiveTransform(src, dst)
                    bc = cv2.warpPerspective(barcode, warp, (szel, mag), borderValue=255)

                    result = decode(bc, I)
                    if len(result) != 0:
                        continue
                    else: print('HIBA: Nem sikerült megfejteni a kódot')
                else: print('HIBA: Harris nem talált sarkokat')


if len(sys.argv) > 2 and sys.argv[1] == '-qr' :
    QR(sys.argv[2:])

elif len(sys.argv) > 2 and sys.argv[1] == '-bc':
    BC(sys.argv[2:])

else: print('Rossz indítás!\nHelyes indítás: Scanner.py -qr/-bc kep1.jpg kep2.jpg ...')
