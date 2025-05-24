# Восстановление пространственных расположений ключевых точек и камер по последовательности изображений

## Краткая постановка задачи

* **Входы** — последовательность кадров с изображениями ключевых точек:
<div align="center">
    <a href="./">
        <img src="./figures/input.svg" width="75%"/>
    </a>
</div>

* **Выход** — трехмерная реконструкция сцены:
<div align="center">
    <a href="./">
        <img src="./figures/reconstruction.svg" width="33%"/>
    </a>
</div>

---

**Используемые обозначения:**

* Искомые параметры сцены:
    * Ключевые точки:

      $\displaystyle \boldsymbol{x}^i = [\kern1pt x^i, y^i, z^i]^T$ — пространственные координаты $i$-й ключевой точки.
      
      $i = 1, \dots, N$.

    * Камеры:

      $\displaystyle \boldsymbol{c}^{(j)} = [\kern1pt c_x^{(j)}, c_y^{(j)}, c_z^{(j)}]^T$ — пространственные координаты $j$-й камеры.

      $\displaystyle \boldsymbol{\theta}^{(j)} = [\kern1pt \theta_x^{(j)}, \theta_y^{(j)}, \theta_z^{(j)}]^T$ — пространственные углы поворотов $j$-й камеры.
      
      $\displaystyle \varphi_\mathrm{x}$ — горизонтальный угол обзора всех камер.

      $j = 1, \dots, K$.

    $\boldsymbol{X} = \{\boldsymbol{x}^i\}_{i=1}^N$, $\ \boldsymbol{C} = \{\boldsymbol{c}^{(j)}\}_{j=1}^K$, $\ \boldsymbol{\varTheta} = \{\boldsymbol{\theta}^{(j)}\}_{j=1}^K$.

* Изображения ключевых точек, полученные с камер:

    $\displaystyle \overset{\text{pc}}{\textbf{x}} \kern0pt ^{i,(j)} = [\kern1pt \overset{\text{pc}}{\text{x}} \kern0pt ^{i,(j)}, \overset{\text{pc}}{\text{y}} \kern0pt ^{i,(j)}]^T$ — координаты центра входного пикселя изображения $i$-й ключевой точки с $j$-й камеры.

    $\displaystyle \textbf{x}^{i,(j)} = \mathbf{x} (\boldsymbol{x}^i, \boldsymbol{c}^{(j)}, \boldsymbol{\theta}^{(j)}, \varphi_\mathrm{x})$ — координаты изображения $i$-й ключевой точки с $j$-й камеры, полученные при заданных $\boldsymbol{x}^i, \boldsymbol{c}^{(j)}, \boldsymbol{\theta}^{(j)}, \varphi_\mathrm{x}$.

  $(i, j) \in \mathcal{V}$,
  
  $\mathcal{V} \subseteq \{1, \dots, N\} \times \{1, \dots, K\}$ — пары индексов $(i, j)$, отвечающие парам точка-камера, при которых $i$-я точка видна для $j$-й камеры.

---

**Ключевая задача оптимизации для оценки параметров сцены:**

$\displaystyle \mathcal{E} = \frac{1}{2 \kern1pt |\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \big\| \mathbf{x}^{i,(j)} - \overset{\mathrm{pc}}{\mathbf{x}} \kern0pt ^{i,(j)} \big\|_2^2 \to \min_{\boldsymbol{X}, \boldsymbol{C}, \boldsymbol{\varTheta}, \varphi_\mathrm{x}}$.


## Пример процедуры реконструкции сцены в Google Colab

По ссылке 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/drive/1ERbHEKV-CYzrwC1bCXnFl55ZQrCOkXb0?usp=sharing
) 
должен быть доступен Colab-блокнот, в котором можно полностью воспроизвести действия с входными данными из [demo_examples/moscow_ride/input.mp4](https://github.com/OlegRokin/3D_reconstruction/blob/main/demo_examples/moscow_ride/input.mp4) и получить реконстуркцию из [demo_examples/moscow_ride/reconstruction.mp4](https://github.com/OlegRokin/3D_reconstruction/blob/main/demo_examples/moscow_ride/reconstruction.mp4).

<div align="center">
    <a href="./">
        <img src="./demo_examples/moscow_ride/input_frame_0.png" width="33%"/>
        <img src="./demo_examples/moscow_ride/reconstruction_frame_0.png" width="75%"/>
    </a>
</div>