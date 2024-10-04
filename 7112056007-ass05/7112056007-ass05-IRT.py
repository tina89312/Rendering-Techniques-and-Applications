import cv2
import os
import time
import copy
import numpy as np
import math

# 刪除txt檔名
def delete_txt(image_files):
    image_files_new = []
    for i in range(len(image_files)):
        if image_files[i][-4:] != '.txt':
            image_files_new.append(image_files[i])
    return image_files_new

# 讀取txt檔
def read_txt(file_path):
    f = open(file_path, 'r')
    line = f.readline().split(' ')
    a = int(line[0])
    b = int(line[1])
    c = int(line[2])
    d = int(line[3])
    M = int(line[4])
    N = int(line[5])
    G = int(f.readline())
    f.close()
    return a, b, c, d, M, N, G
# 找S
def find_S(t, p):
    S = 1
    while (S * t - 1) % p != 0:
        S = S + 1
    return S

# RT逆轉換g1
def inverse_rectangular_transform_g1(x, y, a, b, c, d, M, N):
    t = a*d - b*c
    p = math.gcd(M, N)
    S = find_S(t, p)
    x_and_y_p= np.mod(np.dot((S * np.array([[d, (p - 1) * b], [(p - 1) * c, a]])), np.array([[x], [y]])), p)
    return x_and_y_p[0][0], x_and_y_p[1][0]

# RT逆轉換g2
def inverse_rectangular_transform_g2(x, y, x_p, y_p, a, b, c, d, M, N):
    p = math.gcd(M, N)
    h = M / p
    v = N / p
    H = ((x - (a * x_p) - (b * y_p)) / p) + (math.ceil((a * p) / h) * h) + (math.ceil((b * p) / h) * h)
    V = ((y - (c * x_p) - (d * y_p)) / p) + (math.ceil((c * p) / v) * v) + (math.ceil((d * p) / v) * v)
    return H, V

# RT逆轉換g3
def inverse_rectangular_transform_g3(H, V, a, b, c, d, M, N):
    p = math.gcd(M, N)
    h = M / p
    v = N / p
    if b % h == 0:
        x_h = (find_S(a, h) * H) %  h
        y_v = find_S(d, v) *(V + (math.ceil((c * h) / v) * v) - (c * x_h)) % v
    elif c % v == 0:
        y_v = (find_S(d, v) * V) % V
        x_h = find_S(a, h) *(H + (math.ceil((b * v) / h) * h) - (b * y_v)) % h
    return x_h, y_v

# RT逆轉換g4
def inverse_rectangular_transform_g4(x_p, y_p, x_h, y_v, M, N):
    p = math.gcd(M, N)
    x = x_p + p * x_h
    y = y_p + p * y_v
    return int(x), int(y)

# RT逆轉換
def inverse_rectangular_transform(image, a, b, c, d, M, N):
    image_change = copy.deepcopy(image)
    for i in range(N):
        for j in range(M):
            x_p, y_p = inverse_rectangular_transform_g1(j, i, a, b, c, d, M, N)
            H, V = inverse_rectangular_transform_g2(j, i, x_p, y_p, a, b, c, d, M, N)
            x_h, y_v = inverse_rectangular_transform_g3(H, V, a, b, c, d, M, N)
            change_x, change_y = inverse_rectangular_transform_g4(x_p, y_p, x_h, y_v, M, N)
            image_change[change_y][change_x] = image[i][j]
    return image_change

#  輸出轉換時間txt
def write_txt(time, G, file_path):
    f = open(file_path, 'w')
    f.write('Decryption round:' + str(G) + '\n')
    f.write('Decryption time:' + str(time) + '\n')
    f.close()

# 已加密圖片資料夾的路徑
folder_path = "encry directory"

# 資料夾中所有圖片的名字
image_files = os.listdir(folder_path)
image_files = delete_txt(image_files)


for image_file in image_files:

    # 開始逆轉換時間
    start_time = time.perf_counter()

    # 構建完整的文件路徑
    image_path = os.path.join(folder_path, image_file) 

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 讀取Secret-Key.txt
    a, b, c, d, M, N, G= read_txt('encry directory/%s-Secret-Key.txt'%image_file[:-8])

    # 進行RT逆轉換
    for i in range(G):
        decryp_image = inverse_rectangular_transform(image, a, b, c, d, M, N)

    # 結束轉換時間
    end_time = time.perf_counter()

    #  輸出轉換時間txt
    write_txt(round((end_time - start_time), 2), G, 'decry directory/%s_dec_time.txt'%image_file[:-8])

    # 轉換後圖片檔名
    image_new_name = image_file[:-8] + '_dec' + image_file[-4:]

    # 保存圖像
    cv2.imwrite('decry directory/%s'%image_new_name, decryp_image)

    #  顯示圖片
    cv2.imshow('%s'%image_new_name, decryp_image)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()