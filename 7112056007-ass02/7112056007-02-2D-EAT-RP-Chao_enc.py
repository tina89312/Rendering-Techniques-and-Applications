import cv2
import numpy as np
import os
import copy
import math
import hashlib

# 定義2D-EAT函數
def two_dimensional_equilateral_arnold_transform(image, a, b):
    image_size = image.shape[0]
    image_change = copy.deepcopy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixcel_change = np.mod(np.dot(np.array([[1, a], [b, a*b+1]]), np.array([[i], [j]])), image_size)
            image_change[pixcel_change[0][0], pixcel_change[1][0]] = image[i, j]
    return image_change

# 轉換成二進制
def Convert_to_binary(num):
    num_binary = bin(num)
    num_binary_array = [0, 0, 0, 0, 0, 0, 0, 0] 
    for i in range(len(num_binary)-1, 1, -1):
        num_binary_array[i + (8 - len(num_binary))] = int(num_binary[i])
    num_binary_array = np.array(num_binary_array)
    return num_binary_array

# 用mod函數取七個隨機數
def rand_num(seed, quantity):
    rand_array = []
    for i in range(quantity, 1, -1):
        rand_array.append(seed % i)
    rand_array = np.array(rand_array)
    return rand_array

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
def write_txt(is_gray, a, b, G, seed, control_parameters, initial_values, g, P0, C0_B, file_path):
    f = open(file_path, 'w')
    f.write(str(a) + ' ' + str(b) + '\n')
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
folder_path = "source"

# 資料夾中所有圖片的名字
image_files = os.listdir(folder_path)

for image_file in image_files:
    # 構建完整的文件路徑
    image_path = os.path.join(folder_path, image_file) 

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  

    # 2D-EAT的次數
    G = 10

    for i in range(G):
        # 執行2D-EAT加密
        a = 1
        b = 1
        encrypted_image = two_dimensional_equilateral_arnold_transform(image, a, b)
        image = encrypted_image

    # 取隨機亂數用
    seed = 100

    # 執行Durstenfeld_Random_Permutation
    image = differentiate_image_color(image, seed)

    # 產生圖片的SHA-512值
    SHA_512_value = image_SHA_512_value(image_path)

    # 執行Operating pixel diffusion
    g = 7
    r, control_parameters, initial_values = Operating_pixel_diffusion(image, SHA_512_value, g)

    # 執行Pixel scrambling using exclusive OR
    image, P0, C0_B = Pixel_scrambling_using_XOR(image, SHA_512_value, r)

    # 轉換後圖片檔名
    image_new_name = image_file[:-4] + '_enc' + image_file[-4:]

    # 保存圖像
    cv2.imwrite('encryp/%s'%image_new_name, image)

    #  顯示圖片
    cv2.imshow('%s'%image_new_name, image)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()

    #  輸出secret keys記錄在Secret-Key.txt
    write_txt(len(image.shape) != 3, a, b, G, seed, control_parameters, initial_values, g, P0, C0_B, 'encryp/%s-Secret-Key.txt'%image_file[:-4])


