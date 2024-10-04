import cv2
import numpy as np
import os
import copy
import math

# 刪除txt檔名
def delete_txt(image_files):
    image_files_new = []
    for i in range(len(image_files)):
        if image_files[i][-4:] != '.txt':
            image_files_new.append(image_files[i])
    return image_files_new

# 讀取txt檔
def read_txt(is_gray,file_path):
    f = open(file_path, 'r')
    line = f.readline().split(' ')
    a = int(line[0])
    b = int(line[1])
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
    return a, b, G, seed, control_parameters, initial_values, g, P0, C0_B

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

# # 判斷圖片是否為灰階圖
# def is_gray(image):
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             if image[i][j][0] != image[i][j][1] or image[i][j][1] != image[i][j][2] or image[i][j][2] != image[i][j][0]:
#                 return False
#     return True

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

# 定義2D-EAT函數
def two_dimensional_equilateral_arnold_inverse_transform(image, a, b):
    image_size = image.shape[0]
    image_change = copy.deepcopy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixcel_change = np.mod(np.dot(np.array([[a*b+1, -a], [-b, 1]]), np.array([[i], [j]])), image_size)
            image_change[pixcel_change[0][0], pixcel_change[1][0]] = image[i, j]
    return image_change

# 測量MSE
def measure_MSE(image, image_original, image_name):
    MSE = 0
    if len(image.shape) == 3:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    MSE += math.pow((image[i][j][k] - image_original[i][j][k]), 2)
        MSE /= (image.shape[0] * image.shape[1] * 3)
    else:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                MSE += math.pow((image[i][j] - image_original[i][j]), 2)
        MSE /= (image.shape[0] * image.shape[1])
    print('%s的MSE:%d'%(image_name, MSE))
    return

#圖片資料夾的路徑
folder_path = "encryp"

#資料夾中所有圖片的名字
image_files = os.listdir(folder_path)

# 刪除txt檔名
image_files = delete_txt(image_files)

for image_file in image_files:
    # 構建完整的文件路徑
    image_path = os.path.join(folder_path, image_file) 

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 讀取Secret-Key.txt
    a, b, G, seed, control_parameters, initial_values, g, P0, C0_B = read_txt(len(image.shape) != 3,'encryp/%s-Secret-Key.txt'%image_file[:-8])

    # 執行Operating pixel diffusion
    r = Operating_pixel_diffusion(image, control_parameters, initial_values, g)

    # 執行Pixel De scrambling using exclusive OR
    image = Pixel_De_scrambling_using_XOR(image, r, P0, C0_B)

    # 執行Durstenfeld_Reverse_Random_Permutation
    image = differentiate_image_color(image, seed)

    for i in range(G):
        # 執行2D-EAT反轉換
        encrypted_image = two_dimensional_equilateral_arnold_inverse_transform(image, a, b)
        image = encrypted_image

    # 測量MSE
    image_original = cv2.imread(os.path.join("source", '%s%s'%(image_file[:-8], image_file[-4:])), cv2.IMREAD_UNCHANGED)
    measure_MSE(image, image_original, image_file[:-8])

    # 轉換後圖片檔名
    image_new_name = image_file[:-8] + '_dec' + image_file[-4:]

    # 保存圖像
    cv2.imwrite('decryp/%s'%image_new_name, image)

    #顯示圖片
    cv2.imshow('%s'%image_new_name, image)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()