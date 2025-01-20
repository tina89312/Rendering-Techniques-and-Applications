import cv2
import os
import copy
import numpy as np
import pandas as pd
import re
import random
import hashlib
import math

# 讀取RPA table、n、M、W、Z
def read_RPA_table(RPA_table_file_name):
    RPA_table = pd.read_csv(f'10-rpatab/{RPA_table_file_name}')
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

    with open(f"11-mesmea/mes_mea_{int(seed/100)}.txt", "w") as mesmea_file:
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

# 將三張灰階影像疊成一張彩色影像
def channel_composition(image1, image2, image3):
    image_change = np.stack([image3.flatten(), image2.flatten(), image1.flatten()], axis=1).reshape((image1.shape[0], image1.shape[1], 3))

    return image_change

# 打亂channel
def channel_permutation(image, PK):
    image_change = copy.deepcopy(image)
    match PK:
        case 1:
            image_change[:, :, 0] = image[:, :, 0]
            image_change[:, :, 1] = image[:, :, 2]
            image_change[:, :, 2] = image[:, :, 1]
        case 2:
            image_change[:, :, 0] = image[:, :, 2]
            image_change[:, :, 1] = image[:, :, 0]
            image_change[:, :, 2] = image[:, :, 1]
        case 3:
            image_change[:, :, 0] = image[:, :, 1]
            image_change[:, :, 1] = image[:, :, 0]
            image_change[:, :, 2] = image[:, :, 2]
        case 4:
            image_change[:, :, 0] = image[:, :, 1]
            image_change[:, :, 1] = image[:, :, 2]
            image_change[:, :, 2] = image[:, :, 0]
        case 5:
            image_change[:, :, 0] = image[:, :, 2]
            image_change[:, :, 1] = image[:, :, 1]
            image_change[:, :, 2] = image[:, :, 0]

    return image_change

# 讀取txt檔
def RT_secret_key(file_path):
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

# 用mod函數取七個隨機數
def rand_num(seed, quantity):
    rand_array = []
    for i in range(quantity, 1, -1):
        rand_array.append(seed % i)
    rand_array = np.array(rand_array)
    return rand_array

# 轉換成二進制
def Convert_to_binary(num):
    num_binary = bin(num)
    num_binary_array = [0, 0, 0, 0, 0, 0, 0, 0] 
    for i in range(len(num_binary)-1, 1, -1):
        num_binary_array[i + (8 - len(num_binary))] = int(num_binary[i])
    num_binary_array = np.array(num_binary_array)
    return num_binary_array

# 二進制轉換成十進制
def binary_to_decimal(num_array):
    num = 0
    for i in range(len(num_array)):
        if(num_array[i] == 1):
            num += math.pow(2, 7-i)
    return num

# 定義彩色圖的Durstenfeld_Random_Permutation函數
def Durstenfeld_Random_Permutation_True_Color(num, rand_array):
    num_array_blue = Convert_to_binary(num[0])
    num_array_green = Convert_to_binary(num[1])
    num_array_red = Convert_to_binary(num[2])
    num_array = np.concatenate((num_array_red, num_array_green, num_array_blue))
    for i in range(len(rand_array)):
        change_site = rand_array[i]
        change_num = num_array[change_site]
        num_array[change_site] = num_array[len(rand_array)-i]
        num_array[len(rand_array)-i] = change_num
    num_array_red_transform, num_array_green_transform, num_array_blue_transform = np.split(num_array, 3)
    num_array_red_transform = binary_to_decimal(num_array_red_transform)
    num_array_green_transform = binary_to_decimal(num_array_green_transform)
    num_array_blue_transform = binary_to_decimal(num_array_blue_transform)
    return np.array([num_array_blue_transform, num_array_green_transform, num_array_red_transform])

# 定義灰階圖的Durstenfeld_Random_Permutation函數
def Durstenfeld_Random_Permutation_Gray(num, rand_array):
    num_array = Convert_to_binary(num)
    for i in range(len(rand_array)):
        change_site = rand_array[i]
        change_num = num_array[change_site]
        num_array[change_site] = num_array[len(rand_array)-i]
        num_array[len(rand_array)-i] = change_num
    num_transform =  binary_to_decimal(num_array)
    return num_transform

# 判斷圖片為彩色或灰階並做Durstenfeld_Random_Permutation
def differentiate_image_color(image, seed):
    if len(image.shape) == 3:
        rand_array = rand_num(seed, 24)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = Durstenfeld_Random_Permutation_True_Color(image[i][j], rand_array)
        image_transform = image
    else:
        rand_array = rand_num(seed, 8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = Durstenfeld_Random_Permutation_Gray(image[i][j], rand_array)
        image_transform = image
    return image_transform

# 產生圖片的SHA-512值
def image_SHA_512_value(image_path):
    with open(image_path,"rb") as f:
        bytes = f.read()
        readable_hash = hashlib.sha256(bytes).hexdigest();
        return bin(int(readable_hash,base=16))

# 執行Operating pixel diffusion
def Operating_pixel_diffusion(image, SHA_512_value, g):

    # control parameters(a, b)
    control_parameters = np.array([500.0, 500.0])

    # initial values(x0, y0)
    initial_values = np.array([0.1, 0.1])

    k1 = int('0b'+SHA_512_value[2:18], 2)
    k2 = int('0b'+SHA_512_value[18:34], 2)
    k3 = int('0b'+SHA_512_value[34:42], 2)
    k4 = int('0b'+SHA_512_value[42:50], 2)

    # 修改control parameters與initial values
    control_parameters[0] = round(control_parameters[0] + (k1 / math.pow(2, 16)), 7)
    control_parameters[1] = round(control_parameters[1] + (k2 / math.pow(2, 16)), 7)
    initial_values[0] = round(initial_values[0] + ((k3 / math.pow(2, 8)) * 0.1), 7)
    initial_values[1] = round(initial_values[1] + ((k4 / math.pow(2, 8)) * 0.1), 7)

    x = []
    y = []

    for i in range(g + (image.shape[0]*image.shape[1]*2)):
        if i == 0:
            x.append(round(math.sin(math.pi * (1 - control_parameters[0] * math.pow(initial_values[0], 2) + initial_values[1])), 7))
            y.append(round(math.sin(math.pi * (control_parameters[1] * initial_values[0])), 7))
        else:
            x.append(round(math.sin(math.pi * (1 - control_parameters[0] * math.pow(x[i-1], 2) + y[i-1])), 7))
            y.append(round(math.sin(math.pi * (control_parameters[1] * x[i-1])), 7))
    
    x = x[g+1: ]
    y = y[g+1: ]

    r = []

    for i in range(len(x)):
        r.append(int((x[i] * math.pow(10, 7)) % 256))
        r.append(int((y[i] * math.pow(10, 7)) % 256))

    return np.array(r), control_parameters, initial_values

# 執行Pixel scrambling using exclusive OR
def Pixel_scrambling_using_XOR(image, SHA_512_value, r):

    P0 = [int('0b'+SHA_512_value[50:58], 2), int('0b'+SHA_512_value[58:66], 2), int('0b'+SHA_512_value[66:74], 2)]

    C0_B = int('0b'+SHA_512_value[74:82], 2)

    image_change = copy.deepcopy(image)

    if len(image.shape) == 3:
        for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if i==0 and j==0:
                        image_change[i][j][2] = r[3 * (image.shape[1] * i + j)] ^ image[i][j][2] ^ P0[0] ^ C0_B
                        image_change[i][j][1] = r[(3 * (image.shape[1] * i + j)) + 1] ^ image[i][j][1] ^ P0[1] ^ image_change[i][j][2]
                        image_change[i][j][0] = r[(3 * (image.shape[1] * i + j)) + 2] ^ image[i][j][0] ^ P0[2] ^ image_change[i][j][1]
                    elif j==0:
                        image_change[i][j][2] = r[3 * (image.shape[1] * i + j)] ^ image[i][j][2] ^ image[i-1][image.shape[1] - 1][2] ^ image_change[i-1][image.shape[1] - 1][0]
                        image_change[i][j][1] = r[(3 * (image.shape[1] * i + j)) + 1] ^ image[i][j][1] ^ image[i-1][image.shape[1] - 1][1] ^ image_change[i][j][2]
                        image_change[i][j][0] = r[(3 * (image.shape[1] * i + j)) + 2] ^ image[i][j][0] ^ image[i-1][image.shape[1] - 1][0] ^ image_change[i][j][1]
                    else:
                        image_change[i][j][2] = r[3 * (image.shape[1] * i + j)] ^ image[i][j][2] ^ image[i][j-1][2] ^ image_change[i][j-1][0]
                        image_change[i][j][1] = r[(3 * (image.shape[1] * i + j)) + 1] ^ image[i][j][1] ^ image[i][j-1][1] ^ image_change[i][j][2]
                        image_change[i][j][0] = r[(3 * (image.shape[1] * i + j)) + 2] ^ image[i][j][0] ^ image[i][j-1][0] ^ image_change[i][j][1]
    else:
        for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if i==0 and j==0:
                        image_change[i][j] = r[image.shape[1] * i + j] ^ image[i][j] ^ P0[0] ^ C0_B
                    elif j==0:
                        image_change[i][j] = r[image.shape[1] * i + j] ^ image[i][j] ^ image[i-1][image.shape[1] - 1] ^ image_change[i-1][image.shape[1] - 1]
                    else:
                        image_change[i][j] = r[image.shape[1] * i + j] ^ image[i][j] ^ image[i][j-1] ^ image_change[i][j-1]

    return image_change, P0, C0_B

#  輸出secret keys記錄在Secret-Key.txt
def write_txt(a, b, c, d, M, N, G, is_gray, seed, control_parameters, initial_values, g, P0, C0_B, file_path):
    f = open(file_path, 'w')
    f.write(str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + ' ' + str(M) + ' ' + str(N) + '\n')
    f.write(str(G) + '\n')
    f.write(str(seed) + '\n')
    f.write(str(control_parameters[0]) + ' ' + str(control_parameters[1]) + '\n')
    f.write(str(initial_values[0]) + ' ' + str(initial_values[1]) + '\n')
    f.write(str(g) + '\n')
    if is_gray:
        f.write(str(P0[0]) + '\n')
    else:
        f.write(str(P0[0]) + ' ' + str(P0[1]) + ' ' + str(P0[2]) + '\n')
    f.write(str(C0_B) + '\n')
    f.close()

# 圖片資料夾的路徑
folder_path = "1-origin"

# 資料夾中所有圖片的名字
image_files = os.listdir(folder_path)

# 資料夾中所有RPA table檔案的名字
RPA_table_files = os.listdir("10-rpatab")

# 生成PK
np.random.seed(24)
PK = np.random.randint(0, 6, size=len(image_files))

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
    cv2.imwrite('2-marked/%s'%image_new_name1, image_change1)
    cv2.imwrite('2-marked/%s'%image_new_name2, image_change2)
    cv2.imwrite('2-marked/%s'%image_new_name3, image_change3)

    # 將三張灰階影像疊成一張彩色影像
    gray_to_color_image = channel_composition(image_change1, image_change2, image_change3)

    # 轉換後圖片檔名
    image_new_name4 = image_file[:-4] + '_channe_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z)  + image_file[-4:]

    # 保存圖像
    cv2.imwrite('3-channe/%s'%image_new_name4, gray_to_color_image)

    # 打亂channel
    channel_permutation_image = channel_permutation(gray_to_color_image, PK[image_files.index(image_file)])

    # 轉換後圖片檔名
    image_new_name5 = image_file[:-4] + '_permut_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z)  + image_file[-4:]

    # 保存圖像
    cv2.imwrite('4-permut/%s'%image_new_name5, channel_permutation_image)

    # 讀取RT-Secret-Key.txt
    a, b, c, d, M, N, G= RT_secret_key('12-encpar/%s-Secret-Key.txt'%image_file[:-4])

    # 進行RT轉換
    for i in range(G):
        encrypted_image = rectangular_transform(channel_permutation_image, a, b, c, d, M, N)
        channel_permutation_image = encrypted_image

    # 取隨機亂數用
    seed = 100

    # 執行Durstenfeld_Random_Permutation
    random_permutation_image = differentiate_image_color(encrypted_image, seed)

    # 產生圖片的SHA-512值
    SHA_512_value = image_SHA_512_value('3-channe/%s'%image_new_name4)

    # 執行Operating pixel diffusion
    g = 7
    r, control_parameters, initial_values = Operating_pixel_diffusion(random_permutation_image, SHA_512_value, g)

    # 執行Pixel scrambling using exclusive OR
    pixel_scrambling_image, P0, C0_B = Pixel_scrambling_using_XOR(random_permutation_image, SHA_512_value, r)

    # 保存圖像
    cv2.imwrite('5-encry/%s'%image_file[:-4] + '_enc' + image_file[-4:], pixel_scrambling_image)

    #  輸出secret keys記錄在Secret-Key.txt
    write_txt(a, b, c, d, M, N, G, False, seed, control_parameters, initial_values, g, P0, C0_B, '13-decpar/%s-Secret-Key.txt'%image_file[:-4])
