import cv2
import os
import copy
import numpy as np
import pandas as pd
import re
import random

# 讀取PA table、n、M、W、Z
def read_PA_table(PA_table_file_name):
    PA_table = pd.read_csv(f'patab directory/{PA_table_file_name}')
    PA_table = PA_table.drop(columns=[PA_table.columns[0], PA_table.columns[2], PA_table.columns[6]], axis=1)
    PA_table = PA_table.iloc[:-3, :]
    columns_name = PA_table.iloc[0].tolist()
    PA_table.columns = columns_name
    PA_table = PA_table.drop(PA_table.index[0])
    PA_table = PA_table.reset_index(drop=True).astype(int)

    # 使用正则表达式提取数字
    PA_table_name_numbers = re.findall(r'\d+', PA_table_file_name)
    n = int(PA_table_name_numbers[0])
    M = int(PA_table_name_numbers[1])
    W = np.array(PA_table_name_numbers[2:-1]).astype(int)
    Z = int(PA_table_name_numbers[-1])

    return PA_table, n, M, W, Z

# 產生藏匿的訊息
def produce_secret_message(M, size):
    random.seed(100)
    secret_message = [random.randint(0, 100) % M for _ in range(size)]

    return secret_message

# 從PA table找到對應的A_d
def find_A_d(PA_table, d):
    A_d = PA_table.loc[PA_table['d'] == d].to_numpy()
    A_d = A_d[0, 1:]

    return A_d

# 解決overflow
def solve_overflow(P_prime, P, PA_table, M, W, Z, S):
    P_prime_overflow_index = np.where(P_prime > 255)
    for index in P_prime_overflow_index:
        P[index] = P[index] - Z
    r = np.dot(P, W.T) % M
    d = (S - r) % M
    A_d = find_A_d(PA_table, d)
    P_prime_new = P + A_d

    return P_prime_new, P

# 解決underflow
def solve_underflow(P_prime, P, PA_table, M, W, Z, S):
    P_prime_underflow_index = np.where(P_prime < 0)
    for index in P_prime_underflow_index:
        P[index] = P[index] + Z
    r = np.dot(P, W.T) % M
    d = (S - r) % M
    A_d = find_A_d(PA_table, d)
    P_prime_new = P + A_d

    return P_prime_new, P

# 進行彩色影像的GWM
def GWM_color(image, PA_table, M, W, Z):
    image_change = copy.deepcopy(image)
    secret_messages = produce_secret_message(M, image_change.shape[0] * image_change.shape[1])
    for i in range(image_change.shape[0]):
        for j in range(image_change.shape[1]):
            P = np.flipud(image_change[i][j])
            r = np.dot(P, W.T) % M

            # 藏匿的訊息
            S = secret_messages[(i * image_change.shape[1]) + j]

            d = (S - r) % M
            A_d = find_A_d(PA_table, d)
            P_prime = P + A_d

            while np.where(P_prime > 255)[0].shape[0] > 0 or np.where(P_prime < 0)[0].shape[0] > 0:
                if np.where(P_prime > 255)[0].shape[0] > 0:
                    P_prime, P = solve_overflow(P_prime, P, PA_table, M, W, Z, S)
                else:
                    P_prime, P = solve_underflow(P_prime, P, PA_table, M, W, Z, S)

            image_change[i][j] = np.flipud(P_prime)
    
    return image_change

# 進行灰階影像的GWM
def GWM_gray(image, PA_table, n, M, W, Z):
    image_change = copy.deepcopy(image).reshape(1, -1)
    secret_messages = produce_secret_message(M, int(image_change.shape[1] / n)) 
    for i in range(0, image_change.shape[1], n):
        P = np.array(image_change[0,i:i+n])
        if P.shape[0] == 3:
            r = np.dot(P, W.T) % M

            # 藏匿的訊息
            S = secret_messages[int(i / n)]

            d = (S - r) % M
            A_d = find_A_d(PA_table, d)
            P_prime = P + A_d

            while np.where(P_prime > 255)[0].shape[0] > 0 or np.where(P_prime < 0)[0].shape[0] > 0:
                if np.where(P_prime > 255)[0].shape[0] > 0:
                    P_prime, P = solve_overflow(P_prime, P, PA_table, M, W, Z, S)
                else:
                    P_prime, P = solve_underflow(P_prime, P, PA_table, M, W, Z, S)

            image_change[0,i:i+n] = P_prime
    
    image_change = image_change.reshape(image.shape[0], image.shape[1])
        
    return image_change

# 圖片資料夾的路徑
folder_path = "cover directory"

# 資料夾中所有圖片的名字
image_files = os.listdir(folder_path)

# 資料夾中所有PA table檔案的名字
PA_table_files = os.listdir("patab directory")

for image_file in image_files:
    # 構建完整的文件路徑
    image_path = os.path.join(folder_path, image_file) 

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 讀取PA table、n、M、W、Z
    if image_files.index(image_file) % 4 == 0:
        PA_table, n, M, W, Z = read_PA_table(PA_table_files[int(image_files.index(image_file) / 4)])

    if len(image.shape) == 3:
        image_change = GWM_color(image, PA_table, M, W, Z)
    else:
        image_change = GWM_gray(image, PA_table, n, M, W, Z)

    # 轉換後圖片檔名
    image_new_name = image_file[:-4] + '_stego_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + image_file[-4:]

    # 保存圖像
    cv2.imwrite('stego directory/%s'%image_new_name, image_change)

    #  顯示圖片
    cv2.imshow('%s'%image_new_name, image_change)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()

    