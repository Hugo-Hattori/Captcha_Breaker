import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from helpers import resize_to_fit


dados = []
rotulos = []
pasta_base_imagens = 'letras_rotuladas'

imagens = paths.list_images(pasta_base_imagens)

for arquivo in imagens:
    # a string 'letras_rotuladas\\A\\telanova1002letra4.png' é convertido em uma lista ['letras_rotuladas', 'A', 'telanova1002letra4.png']
    rotulo = arquivo.split(os.path.sep)[-2]
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # padronizar imagens em 20 x 20
    imagem = resize_to_fit(imagem, 20, 20)

    # adicionar uma dimensão para o Keras poder ler a imagem
    imagem = np.expand_dims(imagem, axis=2)

    # adicionar às listas de dados e de rótulos
    rotulos.append(rotulo)
    dados.append(imagem)

# redefinindo as escala da imagem de 0 a 255 para 0 a 1
dados = np.array(dados, dtype='float') / 255
rotulos = np.array(rotulos)

# separação dados de treino (75%) e dados de teste (25%)
(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)

# converter letras para números com one-hot encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# salvar LabelBinarizer em um arquivo com o pickle
with open('rotulos_modelo.dat', 'wb') as arquivo_pickle:
    pickle.dump(lb, arquivo_pickle)

# criar a Inteligência Artificial
modelo = Sequential()

# criar a 1ª camada da rede neural
modelo.add(Conv2D(20, (5, 5), padding='same', input_shape=(20, 20, 1), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# criar a 2ª camada
modelo.add(Conv2D(50, (5, 5), padding='same', activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# criar a 3ª camada
modelo.add(Flatten())
modelo.add(Dense(500, activation='relu'))

# camada de saída (output)
modelo.add(Dense(26, activation='softmax'))

# compilar todas as camadas
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# treinar a Inteligência Artificial
modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=26, epochs=10, verbose=1)

# salvar o modelo treinado em um arquivo
modelo.save('modelo_treinado.hdf5')