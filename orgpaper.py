import math
from telnetlib import KERMIT
from cv2 import DrawMatchesFlags_DRAW_RICH_KEYPOINTS
import numpy as np
import random
import timeit
import cv2
from math import *
import numpy as np
from PIL import Image
# start_time = timeit.default_timer()
# ***************************************************

#it convert data in binary formate
def data2binary(data):
    if type(data) == str:
        p = ''.join([format(ord(i), '08b')for i in data])
    elif type(data) == bytes or type(data) == np.ndarray:
        p = [format(i, '08b')for i in data]
    return p
# hide data in given img
def hidedata(img, data):
    data += "$$"                                   #'$$'--> secrete key
    d_index = 0
    b_data = data2binary(data)
    len_data = len(b_data)
 #iterate pixels from image and update pixel values
    for value in img:
        for pix in value:
            r, g, b = data2binary(pix)
            if d_index < len_data:
                pix[0] = int(r[:-1] + b_data[d_index])
                d_index += 1
            if d_index < len_data:
                pix[1] = int(g[:-1] + b_data[d_index])
                d_index += 1
            if d_index < len_data:
                pix[2] = int(b[:-1] + b_data[d_index])
                d_index += 1
            if d_index >= len_data:
                break
    return img
def encode(keyMsg):

    image = cv2.imread("card.png")
    img = Image.open("card.png", 'r')
    w, h = img.size

    if keyMsg == 0:
        if len(keyMsg) == 0:
            raise ValueError("Empty data")

    enc_data = hidedata(image, str(keyMsg))
    cv2.imwrite("card1.png", enc_data)
    img1 = Image.open("card1.png", 'r')
    img1 = img1.resize((w, h),Image.ANTIALIAS)
    # optimize with 65% quality
    if w != h:
        img1.save("card1.png", optimize=True, quality=65)
    else:
        img1.save("card1.png")
    print("encoding successful")
    print("")
    print("length of msg ",len(keyMsg))
    print("heigth :",h," width ",w)
# decoding
def find_data(img):
    print("in find data")
    print("")
    bin_data = ""
    for value in img:
        for pix in value:
            r, g, b = data2binary(pix)
            bin_data += r[-1]
            bin_data += g[-1]
            bin_data += b[-1]
    all_bytes = [bin_data[i: i + 8] for i in range(0, len(bin_data), 8)]
    readable_data = ""
    for x in all_bytes:
        readable_data += chr(int(x, 2))
        if readable_data[-2:] == "$$":
            break
    return readable_data[:-2]
def decode():
    print("decoding started")
    print("")
    img_name = "card1.png"
    image = cv2.imread(img_name)
    # img=Image.open(img_name,'r')
    # print(image)
    msg = find_data(image)
    print("decoding successful")
    return msg


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr,mse

def MSE(original,compressed):
    err=np.sum((original.astype("float")-compressed.astype("float"))**2)
    print(" err msg",err)
    err/=float(original.shape[0]*original.shape[1])
    return err
# ***************************************************

def adjoint_matrix(matrix):
    try:
        determinant = np.linalg.det(matrix)
        if (determinant != 0):
            cofactor = None
            cofactor = np.linalg.inv(matrix).T * determinant
            # return cofactor matrix of the given matrix

            return np.transpose(cofactor)
        else:
            raise Exception("singular matrix")
    except Exception as e:
        print("could not find cofactor matrix due to", e)


x = input("OTP from the server :")
print("")
Key = int(input("input Public Key :"))

print(len(x))
final = []
ascii = []
messageKey = []
# converting message into ascii values
for character in x:
    ascii.append(ord(character))

# finding the length fof the message for matrix generation
size = math.floor(math.sqrt(len(ascii))) + 1

# Encryption

# ascii list is multplied with public key value
messageKey = [element * Key for element in ascii]

# converting list to matrix on size n*n
n, m = size, size

k = 0
print("")
print("size of the matrix : " + str(size))
print("")
# here msgMatrix is Message Matrix
msgMatrix = []

while n * m != len(ascii):
    # checking if Matrix Possible else append 32 in remaining position
    ascii.append(32)

    # Constructing enciphered Matrix
for idx in range(0, n):
    sub = []
    for jdx in range(0, m):
        sub.append(ascii[k])
        k += 1
    msgMatrix.append(sub)

# printing result
print("Message Matrix ")

print(msgMatrix)

while n * m != len(messageKey):
    # checking if Matrix Possible else append 32 in remaining position
    messageKey.append(32 * Key)

n, m = size, size
l = 0
# keyMsg is enciphered Matrix
keyMsg = []
for idx1 in range(0, n):
    sub1 = []
    for jdx1 in range(0, m):
        sub1.append(messageKey[l])
        l += 1
    keyMsg.append(sub1)

print("")
print("Enciphered Matrix")

print(keyMsg)
# print("")
# print(type(keyMsg))

ciph=""
for i in keyMsg:
    for j in i:
        ciph += str(j)
        ciph += ' '



encode(ciph)
# decryption
# using key value construction of a matrix
# generating random matrix according to size of key value

cypmsg = decode()

cypmsg = cypmsg.split(' ')

cypmsg.remove('')
for i in range(0,len(cypmsg)):
    cypmsg[i]=int(cypmsg[i])
print(cypmsg)
n =int(math.sqrt(len(cypmsg)))
cypmsg_matrix = []
while cypmsg != []:
    cypmsg_matrix.append(cypmsg[:n])
    cypmsg = cypmsg[n:]



random_matrix = [[random.random() for e in range(len(cypmsg_matrix))] for e in range(len(cypmsg_matrix))]

# converting random matrix into lower triangular matrix
for i in range(n):
    for j in range(n):
        if (i < j):
            random_matrix[i][j] = 0

for i in range(n):
    for j in range(n):
        if (i == j):
            random_matrix[i][j] = 1
random_matrix[0][0] = Key

det = np.linalg.det(random_matrix)
round(det)

adj_random = []
inverse_random_Matrix = []
inverse_adj_random = []
adj_random = adjoint_matrix(random_matrix)

inverse_random_Matrix = np.linalg.inv(random_matrix)

inverse_adj_random = np.linalg.inv(adj_random)

Product_rand_adjrand = []
Product_rand_adjrand = np.dot(inverse_random_Matrix, inverse_adj_random)

decoded_matrix = []
decoded_matrix = np.dot(cypmsg_matrix, Product_rand_adjrand)
print("")
print("*******Deciphering ******")
print("")

print("decipheed Matrix")

print(decoded_matrix)
print("")

dec_list = []
dec_list = decoded_matrix.flatten()

dec_list = [int(round(x)) for x in dec_list]
# print(dec_list)
decrepted_Message = "".join([chr(value) for value in dec_list])

# printing deciphered Message
final.append(decrepted_Message)
print("")
print("OTP message after deciphering:")
print(''.join(final))
print("")


original =cv2.imread("card.png")
compressed=cv2.imread("card1.png")

value,mse=PSNR(original,compressed)
value1=MSE(original,compressed)
print("PSNR value  :",value,"MSE :",mse)
print("MSE value   ",value1)

# stop_time = timeit.default_timer()

# print('Time: ', stop_time - start_time)
