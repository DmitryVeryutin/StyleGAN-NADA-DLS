import torch
import math
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class LatentOptimizer(nn.Module):
    # оптимизирует латент для инверсии картинки
    def __init__(self, fixed_generator=0, pic_size=(1024, 1024), perceptual_loss=0, device='cuda'):

        super(LatentOptimizer, self).__init__()
        self.generator = fixed_generator
        self.perceptual_loss = perceptual_loss
        self.transform = transforms.Compose([
                transforms.Resize(pic_size),  # Изменяем размер изображения до 256x256
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Нормализуем значения пикселей
        ])
        self.device=device

    def get_lr(self, t, initial_lr, rampdown=0.25, rampup=0.05):
            """
            Кастомный скедулер, сначала быстро поднимает шаг обучения, потом долгое время сохраняет его и затем резко снижает
            """
            lr_ramp = min(1, (1 - t) / rampdown)
            lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
            lr_ramp = lr_ramp * min(1, t / rampup)

            return initial_lr * lr_ramp
    
    def noise_regularize(self, noises):
        """
        Функция для регуляризации шума.
        Вычисляет лосс на основе шума, чтобы уменьшить его влияние на генерацию.

        Параметры:
        noises (list): Список тензоров шума.

        Возвращает:
        loss (float): Общая потеря, связанная с шумом.
        """
        loss = 0  # Инициализация потерь

        for noise in noises:  # Проход по каждому шуму в списке
            size = noise.shape[2]

            while True:  # Цикл для уменьшения размера шума
                # Вычисление лосса на основе свертки шума с его сдвинутой версией
                loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)  # Сдвиг по ширине
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)  # Сдвиг по высоте
                )

                if size <= 8:
                    break

                # Уменьшение размера шума вдвое
                noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
                noise = noise.mean([3, 5])
                size //= 2

        return loss


    def noise_normalize_(self, noises):
        """
        Нормализация шума.
        Приводит шум к нулевому среднему и единичной дисперсии.

        Параметры:
        noises (list): Список тензоров шума.
        """
        for noise in noises:
            mean = noise.mean()  # Вычисление среднего значения
            std = noise.std()  # Вычисление стандартного отклонения

            # Нормализация: вычитание среднего и деление на стандартное отклонение
            noise.data.add_(-mean).div_(std)


    def latent_noise(self, latent, strength):
        """
        Добавление шума к латентному вектору.
        Генерирует случайный шум и добавляет его к латентному вектору.

        Параметры:
        latent (Tensor): Латентный вектор.
        strength (float): Сила добавляемого шума.

        Возвращает:
        Tensor: Латентный вектор с добавленным шумом.
        """
        noise = torch.randn_like(latent) * strength

        return latent + noise


    def forward(self, input_image, num_steps_w=2000,  # Количество шагов оптимизации в W-пространстве
                alpha=5,  # Коэффициент для перцептивной потери
                beta=1e5):

        target_img_tensor = self.transform(input_image).unsqueeze(0)  # Применяем преобразования, определенные ранее

        # Преобразуем оптимизированный Z-вектор в W-пространство
        latent_z = torch.randn(10000, 512, device=self.device).mean(0, keepdim=True) # Z noise
        latent_w = self.generator.style(latent_z) # Преобразование Z-вектора в W-вектор
        latent_mean = latent_w.mean(0)
        latent_std = ((latent_w - latent_mean).pow(2).sum() / 10000) ** 0.5
        # print(latent_z.mean(0).shape, latent_mean.shape)
        noises_single = self.generator.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(1, 1, 1, 1).normal_())

        # Функция потерь MSE
        mse = torch.nn.MSELoss()

        w_plus = True
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(1, 1)

        if w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, self.generator.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer_w = optim.Adam([latent_in] + noises, lr=0.25)  # Оптимизатор для инверсии в W-пространстве
        # scheduler_w = torch.optim.lr_scheduler.ExponentialLR(optimizer_w, gamma=0.99)  # Плавное снижение скорости обучения


        losses = {
                'mse': [],
                'lpips': [],
                'noise_reg': [],
                'all': []
            }
        init_lr = 0.25

        # Инверсия в W-пространстве с шумовой регуляризацией
        for step in tqdm(range(num_steps_w)):
            torch.cuda.empty_cache()
            t = step / num_steps_w
            lr = self.get_lr(t, init_lr)
            optimizer_w.param_groups[0]["lr"] = lr
            optimizer_w.zero_grad()  # Обнуление градиентов перед новой итерацией

            noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength.item())

            generated_img, _ = self.generator([latent_n], input_is_latent=True, noise=noises)  # Генерация изображения из W-вектора

            # Основные потери
            noise_loss = self.noise_regularize(noises)
            mse_loss = mse(generated_img.to(self.device), target_img_tensor.to(self.device))  # Вычисление потерь MSE
            lpips_loss = self.perceptual_loss(generated_img, target_img_tensor).mean()  # Вычисление перцептивной потери
            # noise_reg_loss = noise_regularize(noises) * noise_strength  # (опционально) Шумовая регуляризация

            # Суммарные потери
            loss = mse_loss + alpha * lpips_loss + beta * noise_loss # Общая потеря: сумма MSE и перцептивной потери с учетом коэффициента alpha
            loss.backward()  # Обратное распространение ошибки

            optimizer_w.step()  # Обновление параметров W-вектора

            self.noise_normalize_(noises)

            losses['mse'].append(mse_loss.item())
            losses['lpips'].append(lpips_loss.item())
            losses['noise_reg'].append(noise_loss.item())
            losses['all'].append(loss.item())
                
        print("Инверсия в W-пространстве с регуляризацией шумов завершена.")  # Сообщение о завершении процесса

        # Посмотрим на конечный результат
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow((generated_img.cpu().detach().squeeze().permute(1, 2, 0) + 1) / 2)
        axs[0].set_title("Сгенерированное изображение")
        axs[0].axis('off')

        axs[1].imshow((target_img_tensor.cpu().squeeze().permute(1, 2, 0).clip(-1,1) + 1) / 2)
        axs[1].set_title("Целевое изображение")
        axs[1].axis('off')

        plt.show()

        # Сохранение результатов

        save_dir = 'optimization_results_lat_opt'
        filename = "real_image_w+_optimization_lat_opt"

        # Проверка, что создана папка для сохранения результатов
        os.makedirs(save_dir, exist_ok=True)

        # Save the latents to a .pt file.
        latent_path = os.path.join(save_dir, filename + ".pt")
        torch.save(latent_in, latent_path)

        # Save the image to a .png file.
        image_path = os.path.join(save_dir, filename + ".png")
        save_image(generated_img, image_path)

        print(f"Результаты успешно сохранены в папку {save_dir} с именем {filename}")
