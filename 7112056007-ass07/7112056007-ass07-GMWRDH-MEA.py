import cv2
import os
import copy
import numpy as np
import pandas as pd
import re
import random

# 讀取RPA table、n、M、W、Z
def read_RPA_table(RPA_table_file_name):
    RPA_table = pd.read_csv(f'rpatab/{RPA_table_file_name}')
    RPA_table = RPA_table.drop(columns=[RPA_table.columns[0], RPA_table.columns[2], RPA_table.columns[6]], axis=1)
    RPA_table = RPA_table.iloc[:-3, :]
    columns_name = RPA_table.iloc[0].tolist()
    RPA_table.columns = columns_name
    RPA_table = RPA_table.drop(RPA_table.index[0])
    RPA_table = RPA_table.reset_index(drop=True).astype(int)

    # 使用正则表达式提取数字
    RPA_table_name_numbers = re.findall(r'\d+', RPA_table_file_name)
    n = int(RPA_table_name_numbers[0])
    M = int(RPA_table_name_numbers[1])
    W = np.array(RPA_table_name_numbers[2:-1]).astype(int)
    Z = int(RPA_table_name_numbers[-1])

    return RPA_table, n, M, W, Z

# 產生藏匿的訊息
def produce_secret_message(M, size, seed):
    random.seed(seed)
    secret_messages = [random.randint(0, 100) % M for _ in range(size)]

    with open(f"mesmea/mes_mea_{int(seed/100)}.txt", "w") as mesmea_file:
        for secret_message in secret_messages:
            mesmea_file.write("%s " % secret_message)

    return secret_messages

# 從RPA table找到對應的A_d
def find_A_d(RPA_table, d):
    A_d = RPA_table.loc[RPA_table['d'] == d].to_numpy()
    A_d = A_d[0, 1:]

    return A_d

# 進行GMWRDH
def GMWRDH(image, RPA_table, n, M, W, Z, seed):
    image_change1 = copy.deepcopy(image)
    image_change2 = copy.deepcopy(image)
    image_change3 = copy.deepcopy(image)
    secret_messages = produce_secret_message(M, image.shape[0] * image.shape[1], seed) 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] >= 0 and image[i][j] < Z:
                P = np.array([Z] * n)
            elif image[i][j] > (255 - Z) and image[i][j] <= 255:
                P = np.array([255 - Z] * n)
            else:
                P = np.array([image[i][j]] * n)

            r = np.dot(P, W.T) % M

            # 藏匿的訊息
            S = secret_messages[(i * image.shape[1]) + j]

            d = (S - r) % M
            A_d = find_A_d(RPA_table, d)
            P_prime = P + A_d

            image_change1[i][j] = P_prime[0]
            image_change2[i][j] = P_prime[1]
            image_change3[i][j] = P_prime[2]
        
    return image_change1, image_change2, image_change3

# 圖片資料夾的路徑
folder_path = "origin"

# 資料夾中所有圖片的名字
image_files = os.listdir(folder_path)

# 資料夾中所有RPA table檔案的名字
RPA_table_files = os.listdir("rpatab")

for image_file in image_files:
    # 構建完整的文件路徑
    image_path = os.path.join(folder_path, image_file) 

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 讀取RPA table、n、M、W、Z
    RPA_table, n, M, W, Z = read_RPA_table(RPA_table_files[image_files.index(image_file)])

    image_change1, image_change2, image_change3 = GMWRDH(image, RPA_table, n, M, W, Z, (image_files.index(image_file) + 1) * 100)

    # 轉換後圖片檔名
    image_new_name1 = image_file[:-4] + '_mark_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I1' + image_file[-4:]
    image_new_name2 = image_file[:-4] + '_mark_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I2' + image_file[-4:]
    image_new_name3 = image_file[:-4] + '_mark_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I3' + image_file[-4:]

    # 保存圖像
    cv2.imwrite('marked/%s'%image_new_name1, image_change1)
    cv2.imwrite('marked/%s'%image_new_name2, image_change2)
    cv2.imwrite('marked/%s'%image_new_name3, image_change3)

    #  顯示圖片
    cv2.imshow('%s'%image_new_name1, image_change1)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()
    cv2.imshow('%s'%image_new_name2, image_change2)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()
    cv2.imshow('%s'%image_new_name3, image_change3)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()