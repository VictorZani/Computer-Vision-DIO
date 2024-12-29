import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Caminho da imagem
image_path = r'linhas-coloridas.png'

# Carregar a imagem em escala de cinza
img = cv.imread(image_path, cv.IMREAD_COLOR)
assert img is not None, f"Erro ao carregar a imagem: {image_path}"

# Aplicar Canny para detectar bordas
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

_ , binary = cv.threshold(gray,100,255,cv.THRESH_BINARY)


# Exibir os resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Imagem em Escala de Cinza")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary, cmap="gray")
plt.title("Imagem Binarizada")
plt.axis("off")

plt.tight_layout()
plt.show()

