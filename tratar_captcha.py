import cv2
import os
import glob
from PIL import Image


def tratar_imagens(pasta_origem, pasta_destino='imgs_ajustadas'):
    arquivos = glob.glob(f'{pasta_origem}/*') #retorna local e o nome de todos os arquivos na pasta
    for arquivo in arquivos:
        imagem = cv2.imread(arquivo)
        # transformar a imagem em escala de cinza
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

        # aplicando tratamento 1
        _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_TRUNC or cv2.THRESH_OTSU)
        nome_arquivo = os.path.basename(arquivo)
        cv2.imwrite(f'{pasta_destino}/{nome_arquivo}', imagem_tratada)

    arquivos = glob.glob(f'{pasta_destino}/*')
    for arquivo in arquivos:
        imagem = Image.open(arquivo)
        imagem = imagem.convert('P')  # garantindo que a imagem estará em tons de cinza
        imagem2 = Image.new(mode='P', size=imagem.size, color=(255, 255, 255))

        # aplicando tratamento 2
        for x in range(imagem.size[1]):  # percorrendo a largura da imagem
            for y in range(imagem.size[0]):  # percorrendo a altura da imagem
                cor_pixel = imagem.getpixel((y, x))
                if cor_pixel < 115:
                    # pinta de preto, os demais permanecerão brancos
                    imagem2.putpixel((y, x), (0, 0, 0))

        nome_arquivo = os.path.basename(arquivo)
        imagem2.save(f'{pasta_destino}/{nome_arquivo}')


if __name__ == "__main__":
    tratar_imagens('bdcaptcha')