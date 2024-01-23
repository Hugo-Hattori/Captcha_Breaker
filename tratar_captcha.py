import cv2
import os
import glob


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


if __name__ == "__main__":
    tratar_imagens('bdcaptcha')