import cv2
import os
import glob


arquivos = glob.glob('imgs_ajustadas/*')
for arquivo in arquivos:
    # obs: embora as imagens já estando em preto e branco, é necessário seguir esses passos
    # para que o opencv entenda que estamos lidando com imagens preto e branco
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    # converter preto e branco
    # aqui criamos "nova_imagem" somente para identificar os contornos, mas depois iremos desenhar o retângulo na imagem original
    _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)

    # achando contornos das letras
    contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regiao_letras = []

    # filtrar os contornos que são realmente de letras
    for contorno in contornos:
        (x, y, largura, altura) = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        if area > 115:
            regiao_letras.append((x, y, largura, altura))
    if len(regiao_letras) != 5:
        continue #pula o passo abaixo e vai para o próximo item do for

    # desenhar os contornos (retângulo) e separa as letras em arquivos individuais
    imagem_final = cv2.merge([imagem] * 3) #recriando listas R, G e B a partir da imagem preto a branco

    i = 0
    for retangulo in regiao_letras:
        x, y, largura, altura = retangulo
        imagem_letra = imagem[y-2:y+altura+2, x-2:x+largura+2] #adicionando +2 e -2 para aumentar a folga do retângulo
        i += 1
        height, width = imagem_letra.shape
        if (height > 0) and (width > 0):
            nome_arquivo = os.path.basename(arquivo).replace('.png', f'letra{i}.png')
            cv2.imwrite(f'letras/{nome_arquivo}', imagem_letra)
            cv2.rectangle(imagem_final, (x - 2, y - 2), (x + largura + 2, y + altura + 2), (0, 255, 0), 1)  # imagem, ponto inicial, ponto final, cor da linha
        else:
            print("Imagem vazia")

    nome_arquivo = os.path.basename(arquivo)
    cv2.imwrite(f'identificado/{nome_arquivo}', imagem_final)