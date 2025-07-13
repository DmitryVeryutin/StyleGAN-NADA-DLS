# StyleGAN-NADA-DLS
Реимплементация StyleGAN-NADA в качестве дипломного проекта для Deep Learning School (1-й семестр, весна 2025).

Данный проект - моя реимплементация сети StyleGAN-NADA для изменения картинок в заданном пользователем направлении. Поддерживается только редактирование лиц, т.к. модель предобучена на датасете лиц FFHQ. Ссылка на оригинальную статью 2021 года: https://arxiv.org/abs/2108.00946.

**Общий принцип действия сети:**

Используются два предобученных генератора, один из которых сохраняют неизменным (Gfrozen), а другой обучают изменять картинку (Gtrain), чтобы сделать ее похожей на целевой промпт. Неизменный генератор нужен для того, чтобы иметь исходную картинку для любого латентного вектора, а не обучаться только на одной заданной картинке.

<p float="centered">
  <img src="img/2_generators.png" width=49% />
</p>

**Как использовать:**

Нужно просто открыть My_StyleGAN_NADA.ipynb и запустить код. Поддерживается изменение как сгенерированных, так и реальных картинок. Основной код модели находится в файле model_nada.py, лоссы реализованы в losses.py, модули для инверсии реальных картинок находятся в latent_optimizer.py и e4e_encoder.py.


**Основные новшества статьи:**

- использование в качестве лосса не просто косинусного расстояния между вектором изменяемой картинки и вектором целевого промпта, но косинусного расстояния между векторами от исходного промпта к целевому (ΔT) и от исходной картинки к желаемой (ΔI) в пространстве CLIP (т.н. направленный лосс):
<p float="centered">
  <img src="img/CLIP_directions.PNG" width=39% />
</p>

- обучение не всех слоев генератора, а только тех, которые наиболее сильно влияют на результирующую картинку, такие слои определяют по значению градиента для их параметров (loss feedback):

<p float="centered">
  <img src="img/Layers_choice.PNG" width=49% />
</p>

**Примеры работы моей реализации** (слева адаптация сгенерированной картинки, справа - инвертированного реального фото, взятого из фотостока, над картинками - описание стартовой картинки -> описание целевой картинки):

- photo -> sketch
<p float="centered">
  <img src="img/Gen_image_adapted_from_Photo_to_Sketch.png" width=49% />
  <img src="img/Real_image_adapted_from_Photo_to_Sketch.png" width=49% />
</p>

- photo -> anime
<p float="centered">
  <img src="img/Gen_image_adapted_from_Photo_to_Anime.png" width=49% />
  <img src="img/Real_image_adapted_from_Photo_to_Anime.png" width=49% />
</p>

- human -> Joker
<p float="centered">
  <img src="img/Gen_image_adapted_from_Person_to_Joker.png" width=49% />
  <img src="img/Real_image_adapted_from_Person_to_Joker.png" width=49% />
</p>

- human -> Nicolas Cage
<p float="centered">
  <img src="img/Gen_image_adapted_from_Person_to_Nicolas Cage.png" width=49% />
  <img src="img/Real_image_adapted_from_Person_to_Nicolas Cage.png" width=49% />
</p>

