# StyleGAN-NADA-DLS
Реимплементация StyleGAN-NADA в качестве дипломного проекта для Deep Learning School (1-й семестр, весна 2025).
https://colab.research.google.com/drive/1UUAuVIstGxMJr44uyVV8E8bj7F5TtZyC?usp=sharing

Данный проект - моя реимплементация сети StyleGAN-NADA для изменения картинок в заданном пользователем направлении. Поддерживается только редактирование лиц, т.к. модель предобучена на датасете лиц FFHQ. Ссылка на оригинальную статью 2021 года: https://arxiv.org/abs/2108.00946.
Общий принцип действия сети: используют два предобученных генератора, один из которых сохраняют неизменным, а другой обучают изменять картинку, чтобы сделать ее похожей на целевой промпт.

<p float="centered">
  <img src="img/2_generators.png" width=49% />
</p>

Основные новшества статьи:
- использование в качестве лосса не просто косинусного расстояния между вектором изменяемой картинки и вектором целевого промпта, но косинусного расстояния между векторами от исходного промпта к целевому (ΔT) и от исходной картинки к желаемой (ΔI) в пространстве CLIP (т.н. направленный лосс), картинка слева
- обучение не всех слоев генератора, а только тех, которые наиболее сильно влияют на результирующую картинку, такие слои определяют по значению градиента для их параметров (loss feedback), картинка справа

<p float="centered">
  <img src="img/CLIP_directions.PNG" width=39% />
  <img src="img/Layers_choice.PNG" width=49% />
</p>

Примеры работы (слева адаптация сгенерированной картинки, справа - инвертированного реального фото, взятого из фотостока, над картинками - использованные промпты):
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

