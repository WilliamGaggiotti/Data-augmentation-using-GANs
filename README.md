# GANs como técnica de data augmentation para el reconocimiento de lengua de señas.

## Introducción

Con el fin de Estudiar las Redes Generativas Adversarias, se utilizaron las mismas con el fin de generar imágenes sintéticas de la lengua de señas Argentina (LSA) y Estadounidense (ASL), para posteriormente utilizar las mismas como técnica de data augmentation.

## Datasets
- LSA16<br />
<img src="https://github.com/WilliamGaggiotti/Data-augmentation-using-GANs/blob/main/Images/datasets_example/LSA_16.png" width="250">
- LSA64 with rotations
- PugeaultASL<br />
- <img src="https://github.com/WilliamGaggiotti/Data-augmentation-using-GANs/blob/main/Images/datasets_example/ASL.png" width="250">

Para mas informacion sobre los dataset  https://github.com/midusi/handshape_datasets

## Métricas de evaluación

Se usó la Distancia Inception de Fréchet (FID). La misma compara realmente las estadísticas de las muestras generadas con las muestras reales. Para esto se utiliza un modelo Inception previamente entrenado (o cualquier otra CNN) para obtener un vector de características o codificación de las muestras reales y falsas. En base a estos vectores se estima la media y covarianza de cada uno, lo que permite formar dos distribuciones Gaussianas, donde una representa la distribución de probabilidad de las imágenes reales, mientras que la otra representa a las falsas. Finalmente se calcula la distancia Fréchet entre estas dos distribuciones, donde un valor bajo significa que las distribuciones están próximas, es decir, el generador está generando imágenes con alta fidelidad y diversidad.

Para el cálculo se utilizó el siguiente libreria https://github.com/mseitzer/pytorch-fid

## Modelos
### GAN Básica
Es el primer modelo GAN que se planteó en 2014 por Ian Goodfellow.

#### Resultados: 
En este caso no se calculo el FID de las muestras generadas ebido a que la fidelidad y diversidad de las imágenes generadas eran evidentemente muy pobres a simple vista. El colapso de moda se puede apreciar fácilmente.

- LSA16
![alt text](https://github.com/WilliamGaggiotti/Data-augmentation-using-GANs/blob/main/Images/results/cap5_BasicGAN_lsa16.png)
- LSA64 with rotations
![alt text](https://github.com/WilliamGaggiotti/Data-augmentation-using-GANs/blob/main/Images/results/cap5_BasicGAN_lsa16.png)

### WCGAN

Esta GAN es una combinación de Wasserstein GAN, cumpliendo la restricción 1-Lipschitz mediante la normalización espectral y Conditional GAN, usando la concatenación como método de condicionamiento. 

#### Resultados: 
Para el cálculo se utilizaron lotes de 6400 imágenes.
- LSA64: El mejor FID obtenido en este caso fue de ~78
![alt text](https://github.com/WilliamGaggiotti/Data-augmentation-using-GANs/blob/main/Images/results/cap5_wgan_lsa16.png)
- ASL:  El mejor FID obtenido en este caso fue de ~84
![alt text](https://github.com/WilliamGaggiotti/Data-augmentation-using-GANs/blob/main/Images/results/cap5_wgan_asl.png)

### BigGAN
Este es uno de los modelos más robustos de GANs que existen en la actualidad. Combina lo mejor de varios modelos. Concretamente, utiliza la distancia Wasserstein y la normalización espectral de WGAN, el módulo de atención de SAGAN para modelar dependencias de largo alcance, y el acondicionamiento de CGAN para controlar la generación de las imágenes. Para G se utilizó normalización por lotes condicional, mientras que para D se utilizó la técnica de proyección.
Como es un modelo tan robusto se tuvieron que hacer recortes en su arquitectura en base al hardware disponible.

#### Resultados: 
Para el cálculo se utilizaron lotes de 6400 imágenes.
- LSA64: El mejor FID obtenido en este caso fue de ~19.5
![alt text](https://github.com/WilliamGaggiotti/Data-augmentation-using-GANs/blob/main/Images/results/cap5_BigGAN_lsa16.png)
- ASL:  El mejor FID obtenido en este caso fue de ~21,5
![alt text](https://github.com/WilliamGaggiotti/Data-augmentation-using-GANs/blob/main/Images/results/cap5_BigGAN_asl.png)
