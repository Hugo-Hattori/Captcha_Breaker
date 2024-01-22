import cv2
from PIL import Image

metodos = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV,
]

imagem = cv2.imread('bdcaptcha/telanova0.png')

# transformar a imagem em escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

i = 0
for metodo in metodos:
    i += 1
    _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 255, metodo or cv2.THRESH_OTSU) #cv2.THRESH_OTSU é um método auxiliar
    cv2.imwrite(f'teste_metodo/imagemtratada_{i}.png' , imagem_tratada)


# após rodar o código acima descobri que o método THRESH_TRUNC é o melhor para o tratamento inicial
imagem = Image.open('teste_metodo/imagemtratada_3.png')
imagem = imagem.convert('P') #garantindo que a imagem estará em tons de cinza
imagem2 = Image.new(mode='P', size=imagem.size, color=(255, 255, 255))

# transformando a imagem em preto e branco puros
for x in range(imagem.size[1]): #percorrendo a largura da imagem
    for y in range(imagem.size[0]): #percorrendo a altura da imagem
        cor_pixel = imagem.getpixel((y, x))
        if cor_pixel < 115:
            #pinta de preto, os demais permanecerão brancos
            imagem2.putpixel((y, x), (0, 0, 0))

imagem2.save('teste_metodo/imagem_final.png')