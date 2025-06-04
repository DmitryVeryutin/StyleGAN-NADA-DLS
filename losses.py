from torch import linalg as LA
from torchvision import transforms
import torch
import torch.nn as nn
#https://github.com/openai/CLIP

# https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/criteria/clip_loss.py

class DirectionalCLIPLoss(torch.nn.Module):
    """
    Этот класс определяет пользовательскую функцию потерь на основе CLIP по целевому направлению в ее пространстве.
    Он измеряет сходство между изображением и текстовым описанием по целевому направлению.
    """

    def __init__(self, stylegan_size=1024, global_loss_impact=0, clip_model=0, clip_preprocess=0):
        """
        Инициализирует класс CLIPLoss.

        Args:
            opts: Объект, содержащий различные параметры, включая размер изображения StyleGAN.
        """
        super(DirectionalCLIPLoss, self).__init__()
        # Загружаем предварительно обученную модель CLIP и функцию предварительной обработки
        self.model = clip_model
        self.model.requires_grad_(False) #с ним не вычисляется баквард

        self.global_loss_impact = global_loss_impact

        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])

        #self.upsample = torch.nn.Upsample(scale_factor=7)
        #self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image, source_text, frozen_image, target_text):
        """
        Вычисляет потери CLIP между изображением и текстом.

        Args:
            image: Входной тензор изображения.
            text: Тензор текстового описания.

        Returns:
            Значение потерь CLIP.
        """
        # Меняем размерность изображения для получения нужного разрешения для CLIP

        image_vec = self.model.encode_image(self.preprocess(image))
        image_vec /= LA.vector_norm(image_vec.clone(), ord=2, dim=-1, keepdim=True)

        source_text_vec = self.model.encode_text(source_text)
        source_text_vec /= LA.vector_norm(source_text_vec.clone(), ord=2, dim=-1, keepdim=True)

        frozen_image_vec = self.model.encode_image(self.preprocess(frozen_image))
        frozen_image_vec /= LA.vector_norm(frozen_image_vec.clone(), ord=2, dim=-1, keepdim=True)

        target_text_vec = self.model.encode_text(target_text)
        target_text_vec /= LA.vector_norm(target_text_vec.clone(), ord=2, dim=-1, keepdim=True)

        di = (image_vec - frozen_image_vec) #edit_direction averaging is not needed 'cause it is done on direction step

        '''
        # zero division problem, add const to frozen image?
        const = 0
        if LA.vector_norm(di.clone(), ord=2, dim=-1, keepdim=True) == 0:
            const = 1e-8
        di /= LA.vector_norm(di.clone() + const, ord=2, dim=-1, keepdim=True) # 512
        '''
        if di.sum() == 0:
            frozen_image_vec = self.model.encode_image(self.preprocess(frozen_image + 1e-6))
            di = (image_vec - frozen_image_vec)

        di /= LA.vector_norm(di.clone(), ord=2, dim=-1, keepdim=True) # 512

        dt = (target_text_vec - source_text_vec).mean(axis=0, keepdim=True) #[0] #target_direction
        dt /= LA.vector_norm(dt.clone(), ord=2, dim=-1, keepdim=True) # 512

        cos = nn.CosineSimilarity(dim=0, eps=1e-8)

        if self.global_loss_impact == 0:
            direction = (1 - cos(di, dt)).mean()
        else:
            direction = (1 - cos(di, dt)).mean() + self.global_loss_impact * (1 - cos(image_vec, target_text_vec)).mean() #added global loss

        return direction
    
class GlobalCLIPLoss(torch.nn.Module):
    """
    Этот класс определяет пользовательскую функцию потерь на основе CLIP (Contrastive Language–Image Pre-training).
    Он измеряет сходство между изображением и текстовым описанием.
    """

    def __init__(self, stylegan_size=1024, clip_model=0, clip_preprocess=0):
        """
        Инициализирует класс CLIPLoss.

        Args:
            opts: Объект, содержащий различные параметры, включая размер изображения StyleGAN.
        """
        super(GlobalCLIPLoss, self).__init__()
        # Загружаем предварительно обученную модель CLIP и функцию предварительной обработки
        self.model = clip_model
        self.model.requires_grad_(False) #с ним не вычисляется баквард
        self.preprocess = clip_preprocess

        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image, text):
        """
        Вычисляет потери CLIP между изображением и текстом.

        Args:
            image: Входной тензор изображения.
            text: Тензор текстового описания.

        Returns:
            Значение потерь CLIP.
        """
        # Меняем размерность изображения для получения нужного разрешения для CLIP
        image = self.avg_pool(self.upsample(image))

        similarity = (1 - self.model(image, text)[0] / 100).mean() # added 1 -

        return similarity
