from etc import config

nz = config["size_of_z_latent"]

transfer_table = [
    [0, 1],
    [0, 1, 3, 2],
    [0, 1, 3, 2, 6, 4, 5, 7],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13, 29, 21, 17, 16, 18, 19, 23, 22, 20, 28, 24, 25, 27, 26, 30,
     31],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13, 29, 21, 17, 16, 18, 19, 23, 22, 20, 28, 24, 25, 27, 26, 30,
     31, 63, 47, 39, 35, 33, 32, 34, 38, 36, 37, 45, 41, 40, 42, 43, 59, 51, 49, 48, 50, 54, 52, 53, 55]]


def de_map(noise_vector, lambda_):
    block = 2 / (2 ** lambda_)
    de_secret = ''
    for i in range(nz):
        noise_bit = noise_vector[0][i]
        for j in range(2 ** lambda_):
            if -1.0 + block * j <= noise_bit <= -1.0 + block * (j + 1):
                de_secret += bin(transfer_table[lambda_ - 1][j])[2:].rjust(lambda_, '0')
                continue
    return de_secret


def de_map_inorder(noise_vector, lambda_):
    block = 2 / (2 ** lambda_)
    de_secret = ''
    for i in range(nz):
        noise_bit = noise_vector[0][i]
        for j in range(2 ** lambda_):
            if -1.0 + block * j <= noise_bit <= -1.0 + block * (j + 1):
                de_secret += bin(j)[2:].rjust(lambda_, '0')
                continue
    return de_secret
