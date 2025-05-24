# Восстановление пространственных расположений ключевых точек и камер по последовательности изображений

## Краткая постановка задачи

* **Входы** — последовательность кадров с изображениями ключевых точек:
<div align="center">
    <a href="./">
        <img src="./figures/input.svg" width="100%"/>
    </a>
</div>

* **Выход** — трехмерная реконструкция сцены:
<div align="center">
    <a href="./">
        <img src="./figures/reconstruction.svg" width="50%"/>
    </a>
</div>

---

**Используемые обозначения:**

* Искомые параметры сцены:
    * Ключевые точки:

      <!-- $\displaystyle \boldsymbol{x}^i = [\kern1pt x^i, y^i, z^i]^T$ — пространственные координаты $i$-й ключевой точки. -->
      ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Cboldsymbol%7Bx%7D%5Ei=%5B%5Ckern1pt%20x%5Ei,y%5Ei,z%5Ei%5D%5ET%7D) — пространственные координаты $i$-й ключевой точки

      <!-- $i = 1, \dots, N$. -->
      ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7Di=1,%5Cdots,N%7D)

    * Камеры:

      <!-- $\displaystyle \boldsymbol{c}^{(j)} = [\kern1pt c_x^{(j)}, c_y^{(j)}, c_z^{(j)}]^T$ — пространственные координаты $j$-й камеры. -->
      ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Cboldsymbol%7Bc%7D%5E%7B(j)%7D=%5B%5Ckern1pt%20c_x%5E%7B(j)%7D,c_y%5E%7B(j)%7D,c_z%5E%7B(j)%7D%5D%5ET%7D) — пространственные координаты $j$-й камеры

      <!-- $\displaystyle \boldsymbol{\theta}^{(j)} = [\kern1pt \theta_x^{(j)}, \theta_y^{(j)}, \theta_z^{(j)}]^T$ — пространственные углы поворотов $j$-й камеры. -->
      ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Cboldsymbol%7B%5Ctheta%7D%5E%7B(j)%7D=%5B%5Ckern1pt%5Ctheta_x%5E%7B(j)%7D,%5Ctheta_y%5E%7B(j)%7D,%5Ctheta_z%5E%7B(j)%7D%5D%5ET%7D) — пространственные углы поворотов $j$-й камеры

      <!-- $\displaystyle \varphi_\mathrm{x}$ — горизонтальный угол обзора всех камер. -->
      ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Cvarphi_%5Cmathrm%7Bx%7D%7D) — горизонтальный угол обзора всех камер.

      <!-- $j = 1, \dots, K$. -->
      ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7Dj=1,%5Cdots,K%7D)

    <!-- $\boldsymbol{X} = \{\boldsymbol{x}^i\}_{i=1}^N$, $\ \boldsymbol{C} = \{\boldsymbol{c}^{(j)}\}_{j=1}^K$, $\ \boldsymbol{\varTheta} = \{\boldsymbol{\theta}^{(j)}\}_{j=1}^K$. -->
    ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Cboldsymbol%7BX%7D=%5C%7B%5Cboldsymbol%7Bx%7D%5Ei%5C%7D_%7Bi=1%7D%5EN,%5C;%5Cboldsymbol%7BC%7D=%5C%7B%5Cboldsymbol%7Bc%7D%5E%7B(j)%7D%5C%7D_%7Bj=1%7D%5EK,%5C;%5Cboldsymbol%7B%5CvarTheta%7D=%5C%7B%5Cboldsymbol%7B%5Ctheta%7D%5E%7B(j)%7D%5C%7D_%7Bj=1%7D%5EK%7D)

* Изображения ключевых точек, полученные с камер:

    <!-- $\displaystyle \overset{\text{pc}}{\textbf{x}} \kern0pt ^{i,(j)} = [\kern1pt \overset{\text{pc}}{\text{x}} \kern0pt ^{i,(j)}, \overset{\text{pc}}{\text{y}} \kern0pt ^{i,(j)}]^T$ — координаты центра входного пикселя изображения $i$-й ключевой точки с $j$-й камеры. -->
    ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Coverset%7B%5Ctext%7Bpc%7D%7D%7B%5Ctextbf%7Bx%7D%7D%5Ckern0pt%5E%7Bi,(j)%7D=%5B%5Ckern1pt%5Coverset%7B%5Ctext%7Bpc%7D%7D%7B%5Ctext%7Bx%7D%7D%5Ckern0pt%5E%7Bi,(j)%7D,%5Coverset%7B%5Ctext%7Bpc%7D%7D%7B%5Ctext%7By%7D%7D%5Ckern0pt%5E%7Bi,(j)%7D%5D%5ET%7D) — координаты центра входного пикселя изображения $i$-й ключевой точки с $j$-й камеры

    <!-- $\displaystyle \textbf{x}^{i,(j)} = \mathbf{x} (\boldsymbol{x}^i, \boldsymbol{c}^{(j)}, \boldsymbol{\theta}^{(j)}, \varphi_\mathrm{x})$ — координаты изображения $i$-й ключевой точки с $j$-й камеры, полученные при заданных $\boldsymbol{x}^i, \boldsymbol{c}^{(j)}, \boldsymbol{\theta}^{(j)}, \varphi_\mathrm{x}$. -->
    ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Ctextbf%7Bx%7D%5E%7Bi,(j)%7D=%5Cmathbf%7Bx%7D(%5Cboldsymbol%7Bx%7D%5Ei,%5Cboldsymbol%7Bc%7D%5E%7B(j)%7D,%5Cboldsymbol%7B%5Ctheta%7D%5E%7B(j)%7D,%5Cvarphi_%5Cmathrm%7Bx%7D)%7D) — координаты изображения $i$-й ключевой точки с $j$-й камеры, полученные при заданных ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Cboldsymbol%7Bx%7D%5Ei,%5Cboldsymbol%7Bc%7D%5E%7B(j)%7D,%5Cboldsymbol%7B%5Ctheta%7D%5E%7B(j)%7D,%5Cvarphi_%5Cmathrm%7Bx%7D%7D)

  <!-- $(i, j) \in \mathcal{V}$, -->
  ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D(i,j)%5Cin%5Cmathcal%7BV%7D%7D)
  
  <!-- $\mathcal{V} \subseteq \{1, \dots, N\} \times \{1, \dots, K\}$ — пары индексов $(i, j)$, отвечающие парам точка-камера, при которых $i$-я точка видна для $j$-й камеры. -->
  ![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Cmathcal%7BV%7D%5Csubseteq%5C%7B1,%5Cdots,N%5C%7D%5Ctimes%5C%7B1,%5Cdots,K%5C%7D%7D) — пары индексов $(i, j)$, отвечающие парам точка-камера, при которых $i$-я точка видна для $j$-й камеры

---

**Ключевая задача оптимизации для оценки параметров сцены:**

<!-- $$\mathcal{E} = \frac{1}{2 \kern1pt |\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \big\| \mathbf{x}^{i,(j)} - \overset{\mathrm{pc}}{\mathbf{x}} \kern0pt ^{i,(j)} \big\|_2^2 \to \min_{\boldsymbol{X}, \boldsymbol{C}, \boldsymbol{\varTheta}, \varphi_\mathrm{x}}$$ -->

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Cmathcal%7BE%7D=%5Cfrac%7B1%7D%7B2%5Ckern1pt%7C%5Cmathcal%7BV%7D%7C%7D%5Csum_%7B(i,j)%5Cin%5Cmathcal%7BV%7D%7D%5Cbig%5C%7C%5Cmathbf%7Bx%7D%5E%7Bi,(j)%7D-%5Coverset%7B%5Cmathrm%7Bpc%7D%7D%7B%5Cmathbf%7Bx%7D%7D%5Ckern0pt%5E%7Bi,(j)%7D%5Cbig%5C%7C_2%5E2%5Cto%5Cmin_%7B%5Cboldsymbol%7BX%7D,%5Cboldsymbol%7BC%7D,%5Cboldsymbol%7B%5CvarTheta%7D,%5Cvarphi_%5Cmathrm%7Bx%7D%7D%7D" alt="equation" />
</p>


## Пример процедуры реконструкции сцены в Google Colab

По ссылке 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/drive/1ERbHEKV-CYzrwC1bCXnFl55ZQrCOkXb0?usp=sharing
) 
должен быть доступен Colab-блокнот, в котором можно полностью воспроизвести действия с входными данными из [demo_examples/moscow_ride/input.mp4](https://github.com/OlegRokin/3D_reconstruction/blob/main/demo_examples/moscow_ride/input.mp4), чтобы получить реконстуркцию из [demo_examples/moscow_ride/reconstruction.mp4](https://github.com/OlegRokin/3D_reconstruction/blob/main/demo_examples/moscow_ride/reconstruction.mp4):

<div align="center">
    <a href="./">
        <img src="./demo_examples/moscow_ride/input_frame_0.png" width="50%"/>
        <img src="./demo_examples/moscow_ride/reconstruction_frame_0.png" width="100%"/>
    </a>
</div>