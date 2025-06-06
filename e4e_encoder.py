import torch
import os

from argparse import Namespace
from torchvision import transforms
from PIL import Image
import time  # Модуль для работы со временем
from utils.common import tensor2im
import numpy as np
import gc
from torchvision.utils import save_image
import torch.nn as nn

class e4eEncoder(nn.Module):
    # Импортируем необходимые библиотеки
    def __init__(self, net=0, fixed_generator=0, device='cuda'):
        super(e4eEncoder, self).__init__()
        self.net = net
        self.pic_size = (256, 256)
        self.fixed_generator = fixed_generator
        # Определяем преобразования для изображения
        self.transform = transforms.Compose([
                transforms.Resize(self.pic_size),  # Изменяем размер изображения до 256x256
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Нормализуем значения пикселей
        ])

    # Определяем функцию для отображения результата рядом с исходным
    def display_alongside_source_image(self, result_image, source_image):
        """Отображает результат рядом с исходным изображением.

        Args:
            result_image: Результирующее изображение (PIL.Image).
            source_image: Исходное изображение (PIL.Image).

        Returns:
            Объединенное изображение (PIL.Image).
        """
        res = np.concatenate([np.array(source_image.resize(self.pic_size)),
                              np.array(result_image.resize(self.pic_size))], axis=1)
        return Image.fromarray(res)

    def run_on_batch(self, inputs, net):
        """Запускает модель на пакете данных.

        Args:
            inputs: Входные данные (тензор PyTorch).
            net: Модель pSp.

        Returns:
            Сгенерированные изображения и латентные векторы.
        """
        images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)  # Запускаем модель
        return images, latents

    def forward(self, input_image):

        transformed_image = self.transform(input_image)  # Применяем преобразования, определенные ранее

        # Запускаем модель и измеряем время выполнения
        with torch.no_grad():
            tic = time.time()  # Время начала выполнения
            images, latents = self.run_on_batch(transformed_image.unsqueeze(0), self.net)
            result_image, inv_latent = images[0], latents[0].unsqueeze(0)  # Извлекаем результат и латентный вектор
            toc = time.time()  # Время окончания выполнения
            print('Inference took {:.4f} seconds.'.format(toc - tic))

            # Отображаем результат инверсии
            #print(inv_latent.size())
            real_result_image = self.fixed_generator([inv_latent], input_is_latent=True)[0].squeeze(0)
            #print(real_result_image.size())
            e4e_result = self.display_alongside_source_image(tensor2im(real_result_image), input_image) #tensor2im(result_image)
            '''
            print(e4e_result)
            e4e_result.show()
            '''
            display(e4e_result)

        gc.collect()
        torch.cuda.empty_cache()

        # Сохранение результатов
        save_dir = 'optimization_results_e4e'
        filename = "real_image_w+_optimization_e4e"

        # Проверка, что создана папка для сохранения результатов
        os.makedirs(save_dir, exist_ok=True)

        # Save the latents to a .pt file.
        latent_path = os.path.join(save_dir, filename + ".pt")

        torch.save(inv_latent, latent_path)

        # Save the image to a .png file.
        image_path = os.path.join(save_dir, filename + ".png")
        save_image(result_image, image_path)

        print(f"Результаты успешно сохранены в папку {save_dir} с именем {filename}")
