# Captcha Breaker

## Project's Objective

<p>Developing a sophisticated AI model with the capability to effectively decipher and
overcome a targeted category of CAPTCHA challenges.</p>

### Packages used:

+ keras.models
+ keras.layers
+ sklearn.preprocessing
+ sklearn.model_selection
+ cv2
+ helpers
+ imutils
+ numpy
+ pickle
+ glob
+ os
+ PIL

## Steps to the Solution

## 1) Database

<p> This project contains 248 text-based CAPTCHAs, each of them carries 5 different letters
that need to be identified by the solver (in this case, the AI we're building) totaling a
database of 1240 images.</p>

<div align="center">

| Example                                   |
|-------------------------------------------|
| ![telanova0.png](bdcaptcha/telanova0.png) |

</div>

## 2) Image/Database Processing

<p>In "teste_modelo.py" 5 different Image Processing Methods are applied and tested: THRESH_BINARY, 
THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO and THRESH_TOZERO_INV.

| THRESH_BINARY                                             | THRESH_BINARY_INV                                       |
|-----------------------------------------------------------|---------------------------------------------------------|
| ![imagemtratada_1.png](teste_metodo/imagemtratada_1.png)  | ![imagemtratada_2.png](teste_metodo/imagemtratada_2.png) |

| THRESH_TOZERO_INV                                        | THRESH_TOZERO                                            |
|----------------------------------------------------------|----------------------------------------------------------|
| ![imagemtratada_5.png](teste_metodo/imagemtratada_5.png) | ![imagemtratada_4.png](teste_metodo/imagemtratada_4.png) |

<div align="center">

| THRESH_TRUNC                                             |
|----------------------------------------------------------|
| ![imagemtratada_3.png](teste_metodo/imagemtratada_3.png) |

</div>

<p> For this scenario the chosen method was THRESH_TRUNC. Therefore, "tratar_captcha.py" is responsible
for applying the method in all images contained in the database and save the processed result in 
"imgs_ajustadas" folder.</p>

## 3) Separate Letters

<p>In "separar_letras.py" we identify the contour of each letter by determining a minimal area of
clustered points, anything below that threshold will not be considered a "letter" and will be discarded.</p>

<p>After the contour is defined, a rectangle is created around that contour so that we can finally
crop each individual letter and save them in different image files.</p>

<div align="center">

![img.png](img.png)

</div>

## 4) Manual Categorization

<p>This step consists in manually separate every images in different folders for a total of
26 folders (1 for each letter). With this, the final database is ready to be used for the AI Training.</p>

<div align="center">

![img_1.png](img_1.png)

</div>

## 5) Training Artificial Intelligence Model and Solving CAPTCHAs

<p>In "treinar_modelo.py" we utilize scikit-learn and Keras to create and train an AI Model by using
the previously mentioned database and a Neural Network Algorithm.</p>

<div align="center">

![img_2.png](img_2.png)

</div>

<p>With the "modelo_treinado.hdf5" AI Model trained we can finally use it to solve CAPTCHAs. 
"resolver_captcha.py" is an example of how to utilize the trained model.</p>