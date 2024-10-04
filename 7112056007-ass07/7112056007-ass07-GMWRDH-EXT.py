import cv2
import os
import copy
import numpy as np
import pandas as pd
import re
import csv

# 找對應的RPA table
def find_RPA_table(image_name):
    # 使用正则表达式提取数字
    image_name = image_name.split('_')
    image_name = '_'.join(image_name[2:])
    image_name_numbers = re.findall(r'\d+', image_name)
    RPA_table_file_name = 'RPA_' + image_name_numbers[0] + '_' + image_name_numbers[1] + '_(' + image_name_numbers[2] + '_' + image_name_numbers[3] + '_' + image_name_numbers[4] + ')_' + image_name_numbers[5] + '.csv'

    return RPA_table_file_name

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

# 解密訊息
def GMWRDH_Message_Extraction(image1, image2, image3, W, M, seed):
    secret_messages = []
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            P_prime = np.array([image1[i][j], image2[i][j], image3[i][j]])
            S = np.dot(P_prime, W.T) % M
            secret_messages.append(S)
    
    with open(f"mesext/mes_ext_{int(seed/100)}.txt", "w") as mesext_file:
        for secret_message in secret_messages:
            mesext_file.write("%s " % secret_message)
            
    return secret_messages

# 解密圖片
def GMWRDH_Restoration(image1, image2, image3, n):
    image_origin = copy.deepcopy(image1)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            P_prime = np.array([image1[i][j], image2[i][j], image3[i][j]])
            P = round(np.sum(P_prime) / n)
            image_origin[i][j] = P
    
    return image_origin

# 轉換後圖片檔名
def create_image_new_name(image_name):
    image_name = image_name.split('_')
    image_name[1] = 'rest'
    image_name.pop()
    image_name[-1] = image_name[-1] + '.png'
    image_new_name = '_'.join(image_name)

    return image_new_name

# csv檔名
def create_csv_name(image_name):
    image_name = image_name.split('_')
    image_name[1] = 'qualit'
    image_name.pop()
    image_name[-1] = image_name[-1] + '.csv'
    csv_name = '_'.join(image_name)

    return csv_name

# 計算MSE
def calculate_MSE(image_new_name):
    image_origin_name = image_new_name.split('_')
    image_origin_name = image_origin_name[0] + '.png'
    image_origin = cv2.imread(f'origin/{image_origin_name}', cv2.IMREAD_UNCHANGED)
    image_new = cv2.imread(f'restor/{image_new_name}', cv2.IMREAD_UNCHANGED)
    MSE = np.mean((image_origin - image_new)**2)

    return MSE

# 計算PSNR
def calculate_PSNR(MSE):
    PSRN = round(10 * np.log10((255**2) / MSE), 2)

    return PSRN

# 計算EC
def calculate_EC(H, V, M):
    EC = round(H * V * np.log2(M), 0)

    return EC

# 計算ER
def calculate_ER(H, V, M):
    ER = round(H * V * np.log2(M) / 3, 5)

    return ER

# 紀錄嵌密與取密結果csv檔案
def write_csv_file(csv_name, n, M, W, MSE, PSNR, EC, ER):
    # 開啟輸出的 CSV 檔案
    with open(f'imgres/{csv_name}', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)

        writer.writerow(['RPA', str(n), str(M), 'w1', 'w2', 'w3'])
        writer.writerow(['Index', 'd', 'SE', str(W[0]), str(W[1]), str(W[2])])
        writer.writerow(['MSE', str(MSE)])
        writer.writerow(['PSNR', str(PSNR)])
        writer.writerow(['EC', str(EC)])
        writer.writerow(['ER', str(ER)])
    
    return
    
# 圖片資料夾的路徑
folder_path = "marked"

# 資料夾中所有圖片的名字
image_files = np.reshape(os.listdir(folder_path),(-1,3))

for index, image_file in enumerate(image_files):
    # 構建完整的文件路徑
    image_path1 = os.path.join(folder_path, image_file[0])
    image_path2 = os.path.join(folder_path, image_file[1]) 
    image_path3 = os.path.join(folder_path, image_file[2])  

    # 使用OpenCV讀取圖像
    image1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)
    image3 = cv2.imread(image_path3, cv2.IMREAD_UNCHANGED)

    # 找對應的RPA table
    RPA_table_file_name = find_RPA_table(image_file[0])

    # 讀取RPA table、n、M、W、Z
    RPA_table, n, M, W, Z = read_RPA_table(RPA_table_file_name)

    # 解密訊息
    secret_messages = GMWRDH_Message_Extraction(image1, image2, image3, W, M, (index + 1) * 100)

    # 解密圖片
    image_origin = GMWRDH_Restoration(image1, image2, image3, n)

    # 轉換後圖片檔名
    image_new_name = create_image_new_name(image_file[0])

    # 保存圖像
    cv2.imwrite('restor/%s'%image_new_name, image_origin)

    # 計算各種誤差值
    MSE = calculate_MSE(image_new_name)
    PSNR = calculate_PSNR(MSE)
    EC = calculate_EC(image_origin.shape[1], image_origin.shape[0], M)
    ER = calculate_ER(image_origin.shape[1], image_origin.shape[0], M)

    # 紀錄嵌密與取密結果csv檔案
    csv_name = create_csv_name(image_file[0])
    write_csv_file(csv_name, n, M, W, MSE, PSNR, EC, ER)

    #  顯示圖片
    cv2.imshow('%s'%image_new_name, image_origin)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()
