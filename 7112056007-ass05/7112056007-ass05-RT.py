import cv2
import os
import time
import copy
import numpy as np

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

# RT轉換
def rectangular_transform(image, a, b, c, d, M, N):
    image_change = copy.deepcopy(image)
    A = np.array([[a, b], [c, d]])
    for i in range(N):
        for j in range(M):
            pixcel_change = np.mod(np.dot(A, np.array([[j], [i]])), np.array([[M], [N]]))
            image_change[pixcel_change[1][0]][pixcel_change[0][0]] = image[i][j]
    return image_change

#  輸出轉換時間txt
def write_txt(time, G, file_path):
    f = open(file_path, 'w')
    f.write('Encryption round:' + str(G) + '\n')
    f.write('Encryption time:' + str(time) + '\n')
    f.close()

# 圖片資料夾的路徑
folder_path = "source directory"

# 資料夾中所有圖片的名字
image_files = os.listdir(folder_path)

for image_file in image_files:

    # 開始轉換時間
    start_time = time.perf_counter()

    # 構建完整的文件路徑
    image_path = os.path.join(folder_path, image_file) 

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 讀取Secret-Key.txt
    a, b, c, d, M, N, G= read_txt('encry directory/%s-Secret-Key.txt'%image_file[:-4])

    # 進行RT轉換
    for i in range(G):
        encrypted_image = rectangular_transform(image, a, b, c, d, M, N)

    # 結束轉換時間
    end_time = time.perf_counter()

    #  輸出轉換時間txt
    write_txt(round((end_time - start_time), 2), G, 'encry directory/%s_enc_time.txt'%image_file[:-4])

    # 轉換後圖片檔名
    image_new_name = image_file[:-4] + '_enc' + image_file[-4:]

    # 保存圖像
    cv2.imwrite('encry directory/%s'%image_new_name, encrypted_image)

    #  顯示圖片
    cv2.imshow('%s'%image_new_name, encrypted_image)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()