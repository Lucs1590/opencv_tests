import os
import glob
from kmeans_remove_background import KMeansClass
import csv
import datetime


def main():
    print(datetime.datetime.now())
    caminho = "resultados/imagens_oficiais"
    nome = "teste_x"
    path = os.getcwd()
    os.makedirs(nome)
    os.chdir(nome)
    with open('{}.csv'.format(nome), mode='w') as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        os.chdir(path)
        for arquivo in glob.glob(os.path.join(caminho, "*.png")):
            i = get_image_rect(arquivo)
            resposta = KMeansClass().runKmeans(arquivo, i, 5)
            csv_writer.writerow(resposta)
        print(datetime.datetime.now())


def get_image_rect(arquivo_name):
    rects = {
        "1": (430, 196, 800, 310),
        "2": (80, 99, 1172, 565),
        "3": (145, 125, 771, 384),
        "4": (51, 4, 1166, 689),
        "5": (125, 118, 1126, 431),
        "6": (125, 89, 1043, 442),
        "7": (117, 31, 1122, 632),
        "8": (91, 35, 1125, 627),
        "9": (93, 37, 1179, 545),
        "10": (419, 225, 588, 283),
        "11": (323, 209, 601, 232),
        "12": (201, 92, 844, 478),
        "13": (180, 72, 881, 409),
        "14": (129, 87, 1534, 565),
        "15": (413, 196, 632, 261),
        "16": (212, 176, 943, 414),
        "17": (94, 34, 1156, 534),
        "18": (121, 90, 1004, 440),
        "19": (443, 193, 734, 349),
        "20": (158, 32, 1023, 526)
    }
    number = arquivo_name.split("/")[-1].split("_")[0].replace("f", "")
    return rects[number]


if __name__ == '__main__':
    main()
