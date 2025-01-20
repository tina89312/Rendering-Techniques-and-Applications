import cv2
import os
import copy
import numpy as np
import pandas as pd
import re
import math
import csv

# 讀取txt檔
def secret_key(is_gray, file_path):
    f = open(file_path, 'r')
    line = f.readline().split(' ')
    a = int(line[0])
    b = int(line[1])
    c = int(line[2])
    d = int(line[3])
    M = int(line[4])
    N = int(line[5])
    G = int(f.readline())
    seed = int(f.readline())
    line = f.readline().split(' ')
    control_parameters = [float(line[0]), float(line[1])]
    line = f.readline().split(' ')
    initial_values = [float(line[0]), float(line[1])]
    g = int(f.readline())
    if is_gray:
        P0 = int(f.readline())
    else:
        line = f.readline().split(' ')
        P0 = [int(line[0]), int(line[1]), int(line[2])]
    C0_B = int(f.readline())
    f.close()

    return a, b, c, d, M, N, G, seed, control_parameters, initial_values, g, P0, C0_B

# 執行Operating pixel diffusion
def Operating_pixel_diffusion(image, control_parameters, initial_values, g):
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

    return np.array(r)

# 執行Pixel De scrambling using exclusive OR
def Pixel_De_scrambling_using_XOR(image, r, P0, C0_B):
    
    image_change = copy.deepcopy(image)

    if len(image.shape) == 3:
        for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if i==0 and j==0:
                        image_change[i][j][2] = r[3 * (image.shape[1] * i + j)] ^ image[i][j][2] ^ P0[0] ^ C0_B
                        image_change[i][j][1] = r[(3 * (image.shape[1] * i + j)) + 1] ^ image[i][j][1] ^ P0[1] ^ image[i][j][2]
                        image_change[i][j][0] = r[(3 * (image.shape[1] * i + j)) + 2] ^ image[i][j][0] ^ P0[2] ^ image[i][j][1]
                    elif j==0:
                        image_change[i][j][2] = r[3 * (image.shape[1] * i + j)] ^ image[i][j][2] ^ image_change[i-1][image.shape[1] - 1][2] ^ image[i-1][image.shape[1] - 1][0]
                        image_change[i][j][1] = r[(3 * (image.shape[1] * i + j)) + 1] ^ image[i][j][1] ^ image_change[i-1][image.shape[1] - 1][1] ^ image[i][j][2]
                        image_change[i][j][0] = r[(3 * (image.shape[1] * i + j)) + 2] ^ image[i][j][0] ^ image_change[i-1][image.shape[1] - 1][0] ^ image[i][j][1]
                    else:
                        image_change[i][j][2] = r[3 * (image.shape[1] * i + j)] ^ image[i][j][2] ^ image_change[i][j-1][2] ^ image[i][j-1][0]
                        image_change[i][j][1] = r[(3 * (image.shape[1] * i + j)) + 1] ^ image[i][j][1] ^ image_change[i][j-1][1] ^ image[i][j][2]
                        image_change[i][j][0] = r[(3 * (image.shape[1] * i + j)) + 2] ^ image[i][j][0] ^ image_change[i][j-1][0] ^ image[i][j][1]
    else:
        for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if i==0 and j==0:
                        image_change[i][j] = r[image.shape[1] * i + j] ^ image[i][j] ^ P0 ^ C0_B
                    elif j==0:
                        image_change[i][j] = r[image.shape[1] * i + j] ^ image[i][j] ^ image_change[i-1][image.shape[1] - 1] ^ image[i-1][image.shape[1] - 1]
                    else:
                        image_change[i][j] = r[image.shape[1] * i + j] ^ image[i][j] ^ image_change[i][j-1] ^ image[i][j-1]
    return image_change

# 轉換成二進制
def Convert_to_binary(num):
    num_binary = bin(num)
    num_binary_array = [0, 0, 0, 0, 0, 0, 0, 0] 
    for i in range(len(num_binary)-1, 1, -1):
        num_binary_array[i + (8 - len(num_binary))] = int(num_binary[i])
    num_binary_array = np.array(num_binary_array)
    return num_binary_array

# 用mod函數取七個隨機數(已反轉)
def rand_num(seed, quantity):
    rand_array = []
    for i in range(quantity, 1, -1):
        rand_array.append(seed % i)
    rand_array = np.flipud(np.array(rand_array))
    return rand_array

# 二進制轉換成十進制
def binary_to_decimal(num_array):
    num = 0
    for i in range(len(num_array)):
        if(num_array[i] == 1):
            num += math.pow(2, 7-i)
    return num

# 定義彩色圖的Durstenfeld_Random_Permutation函數
def Durstenfeld_Reverse_Random_Permutation_True_Color(num, rand_array):
    num_array_blue = Convert_to_binary(num[0])
    num_array_green = Convert_to_binary(num[1])
    num_array_red = Convert_to_binary(num[2])
    num_array = np.concatenate((num_array_red, num_array_green, num_array_blue))
    for i in range(len(rand_array)):
        change_site = rand_array[i]
        change_num = num_array[change_site]
        num_array[change_site] = num_array[i+1]
        num_array[i+1] = change_num
    num_array_red_transform, num_array_green_transform, num_array_blue_transform = np.split(num_array, 3)
    num_array_red_transform = binary_to_decimal(num_array_red_transform)
    num_array_green_transform = binary_to_decimal(num_array_green_transform)
    num_array_blue_transform = binary_to_decimal(num_array_blue_transform)
    return np.array([num_array_blue_transform, num_array_green_transform, num_array_red_transform])

# 定義灰階圖的Durstenfeld_Random_Permutation函數
def Durstenfeld_Reverse_Random_Permutation_Gray(num, rand_array):
    num_array = Convert_to_binary(num)
    for i in range(len(rand_array)):
        change_site = rand_array[i]
        change_num = num_array[change_site]
        num_array[change_site] = num_array[i+1]
        num_array[i+1] = change_num
    num_transform =  binary_to_decimal(num_array)
    return num_transform

# 判斷圖片為彩色或灰階並做Durstenfeld_Reverse_Random_Permutation
def differentiate_image_color(image, seed):
    if len(image.shape) == 3:
        rand_array = rand_num(seed, 24)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = Durstenfeld_Reverse_Random_Permutation_True_Color(image[i][j], rand_array)
        image_transform = image
    else:
        rand_array = rand_num(seed, 8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = Durstenfeld_Reverse_Random_Permutation_Gray(image[i][j], rand_array)
        image_transform = image
    return image_transform

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

# 將channel排回去
def channel_inverse_permutation(image, PK):
    image_change = copy.deepcopy(image)
    match PK:
        case 1:
            image_change[:, :, 0] = image[:, :, 0]
            image_change[:, :, 1] = image[:, :, 2]
            image_change[:, :, 2] = image[:, :, 1]
        case 2:
            image_change[:, :, 0] = image[:, :, 1]
            image_change[:, :, 1] = image[:, :, 2]
            image_change[:, :, 2] = image[:, :, 0]
        case 3:
            image_change[:, :, 0] = image[:, :, 1]
            image_change[:, :, 1] = image[:, :, 0]
            image_change[:, :, 2] = image[:, :, 2]
        case 4:
            image_change[:, :, 0] = image[:, :, 2]
            image_change[:, :, 1] = image[:, :, 0]
            image_change[:, :, 2] = image[:, :, 1]
        case 5:
            image_change[:, :, 0] = image[:, :, 2]
            image_change[:, :, 1] = image[:, :, 1]
            image_change[:, :, 2] = image[:, :, 0]

    return image_change

# 將圖片分成三張灰階圖片
def channel_decomposition(image):
    image_change1 = image[:, :, 2]
    image_change2 = image[:, :, 1]
    image_change3 = image[:, :, 0]

    return image_change1, image_change2, image_change3

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

# 解密訊息
def GMWRDH_Message_Extraction(image1, image2, image3, W, M, seed):
    secret_messages = []
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            P_prime = np.array([image1[i][j], image2[i][j], image3[i][j]])
            S = np.dot(P_prime, W.T) % M
            secret_messages.append(S)
    
    with open(f"14-mesext/mes_ext_{int(seed/100)}.txt", "w") as mesext_file:
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

# 計算MSE
def calculate_MSE(image_new_name):
    image_origin_name = image_new_name.split('_')
    image_origin_name = image_origin_name[0] + '.png'
    image_origin = cv2.imread(f'1-origin/{image_origin_name}', cv2.IMREAD_UNCHANGED)
    image_new = cv2.imread(f'9-restor/{image_new_name}', cv2.IMREAD_UNCHANGED)
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
    with open(f'15-imgres/{csv_name}', 'w', newline='') as csvfile:
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
folder_path = "5-encry"

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

    # 讀取Secret-Key.txt
    a, b, c, d, M, N, G, seed, control_parameters, initial_values, g, P0, C0_B = secret_key(False, '13-decpar/%s-Secret-Key.txt'%image_file.split('_')[0])

    # 執行Operating pixel diffusion
    r = Operating_pixel_diffusion(image, control_parameters, initial_values, g)

    # 執行Pixel De scrambling using exclusive OR
    image = Pixel_De_scrambling_using_XOR(image, r, P0, C0_B)

    # 執行Durstenfeld_Reverse_Random_Permutation
    image = differentiate_image_color(image, seed)

    # 進行RT逆轉換
    for i in range(G):
        decryp_image = inverse_rectangular_transform(image, a, b, c, d, M, N)
        image = decryp_image

    # 轉換後圖片檔名
    image_new_name1 = image_file[:-8] + '_dec' + image_file[-4:]

    # 保存圖像
    cv2.imwrite('6-decry/%s'%image_new_name1, decryp_image)

    # 讀取RPA table、n、M、W、Z
    RPA_table, n, M, W, Z = read_RPA_table(RPA_table_files[image_files.index(image_file)])

    # 將channel排回去
    channel_inverse_permutation_image = channel_inverse_permutation(decryp_image, PK[image_files.index(image_file)])

    # 轉換後圖片檔名
    image_new_name2 = image_file[:-8] + '_invmut_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z)  + image_file[-4:]

    # 保存圖像
    cv2.imwrite('7-invmut/%s'%image_new_name2, channel_inverse_permutation_image)

    # 將圖片分成三張灰階圖片
    image_change1, image_change2, image_change3 = channel_decomposition(channel_inverse_permutation_image)

    # 轉換後圖片檔名
    image_new_name3 = image_file[:-8] + '_decom_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I1' + image_file[-4:]
    image_new_name4 = image_file[:-8] + '_decom_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I2' + image_file[-4:]
    image_new_name5 = image_file[:-8] + '_decom_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '_I3' + image_file[-4:]

    # 保存圖像
    cv2.imwrite('8-decom/%s'%image_new_name3, image_change1)
    cv2.imwrite('8-decom/%s'%image_new_name4, image_change2)
    cv2.imwrite('8-decom/%s'%image_new_name5, image_change3)

    # 解密訊息
    secret_messages = GMWRDH_Message_Extraction(image_change1, image_change2, image_change3, W, M, (image_files.index(image_file) + 1) * 100)

    # 解密圖片
    image_origin = GMWRDH_Restoration(image_change1, image_change2, image_change3, n)

    # 轉換後圖片檔名
    image_new_name6 = image_file[:-8] + '_restor_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + image_file[-4:]

    # 保存圖像
    cv2.imwrite('9-restor/%s'%image_new_name6, image_origin)

    # 計算各種誤差值
    MSE = calculate_MSE(image_new_name6)
    PSNR = calculate_PSNR(MSE)
    EC = calculate_EC(image_origin.shape[1], image_origin.shape[0], M)
    ER = calculate_ER(image_origin.shape[1], image_origin.shape[0], M)

    # 紀錄嵌密與取密結果csv檔案
    csv_name = image_file[:-8] + '_qualit_N' + str(n) + '_M' + str(M) + '_' + str(W[0]) + '_' + str(W[1]) + '_' + str(W[2]) + '_Z' + str(Z) + '.csv'
    write_csv_file(csv_name, n, M, W, MSE, PSNR, EC, ER)
