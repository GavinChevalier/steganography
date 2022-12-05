import numpy as np
from etc import config

nz = config["size_of_z_latent"]


# 'secret' is a string of 0 or 1
# 'm' is a list of fragment secret; type(fragment secret)=string
def secret_transform(secret, lambda_):
    m = []
    len_s = len(secret)
    if len_s % lambda_ != 0:
        num_of_supplement_bits = lambda_ - len_s % lambda_
        for i in range(num_of_supplement_bits):
            secret += '0'
    len_of_m = len(secret) // lambda_
    for i in range(len_of_m):
        m.append(secret[lambda_ * i:lambda_ * i + lambda_])
    return m


transfer_table = [
    [0, 1],
    [0, 1, 3, 2],
    [0, 1, 3, 2, 6, 4, 5, 7],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13, 29, 21, 17, 16, 18, 19, 23, 22, 20, 28, 24, 25, 27, 26, 30,
     31],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13, 29, 21, 17, 16, 18, 19, 23, 22, 20, 28, 24, 25, 27, 26, 30,
     31, 63, 47, 39, 35, 33, 32, 34, 38, 36, 37, 45, 41, 40, 42, 43, 59, 51, 49, 48, 50, 54, 52, 53, 55]]


def secret_mapping_v2(secret, lambda_):
    m = secret_transform(secret, lambda_)
    noise = np.empty([1, nz])
    i = 0
    for item in m:
        m_ten = int(item, 2)

        m_ten = transfer_table[lambda_ - 1].index(m_ten)

        noise[0][i] = m_ten / (2 ** (lambda_ - 1)) - 1

        i += 1
    return noise


def secret_mapping_inorder(secret, lambda_):
    m = secret_transform(secret, lambda_)
    noise = np.empty([1, nz])
    i = 0
    for item in m:
        m_ten = int(item, 2)

        noise[0][i] = m_ten / (2 ** (lambda_ - 1)) - 1

        i += 1
    return noise
