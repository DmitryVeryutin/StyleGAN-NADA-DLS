import torch
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

class DomainAdapter(nn.Module):
    def __init__(self, generator=generator, fixed_generator=fixed_generator,
                 clipglobal=clipglobal, generator_optimizer = generator_optimizer,
                 clipdirect=clipdirect, pic_batch = 4, pic_sample = 4,
                 layers_to_train=layers_to_train, num_steps=200, device=device,
                 text_inputs = text_inputs, start_text_inputs = start_text_inputs):

        super(DomainAdapter, self).__init__()
        self.generator=generator
        self.fixed_generator=fixed_generator
        fixed_generator.eval()

        self.clipglobal=clipglobal
        self.clipdirect=clipdirect

        self.text_inputs = text_inputs
        self.start_text_inputs = start_text_inputs

        self.device = device
        self.num_steps = num_steps
        self.generator_optimizer = generator_optimizer
        self.all_layers = list(self.generator.children())
        self.trainable_layers = list(self.all_layers)[1:3] + list(self.all_layers[4][:])
        self.layer_numbers = [i for i in range(len(self.trainable_layers))]
        self.pic_sample = pic_sample
        self.pic_batch = pic_batch
        self.layers_to_train=layers_to_train
        self.losses = {
          'clipglobal': [],
          'clipdirect': [],
          'clipglobal_metrics': []
        }

        self.const_z = torch.randn(self.pic_sample, 512, device=device)

    def freeze_all_layers(self):
        # морозим все
        for l in self.all_layers:
          for p in l.parameters():
            p.requires_grad = False

    def unfreeze_opt_layers(self):
        # opt layers
        if self.layers_to_train < 18:
          sample_z = torch.randn(self.pic_batch, 512, device=self.device) #был батч 8
          initial_w_codes = self.fixed_generator.style(sample_z) #8x512

          initial_w_codes = initial_w_codes.unsqueeze(1).repeat(1, self.fixed_generator.n_latent, 1) #8x18x512

          w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(self.device) #8x18x512
          w_codes.requires_grad = True

          w_optim = torch.optim.SGD([w_codes], lr=0.01)

          w_codes_for_gen = w_codes.unsqueeze(0) #1x8x18x512, чтобы скормить это все генератору

          generated_from_w = self.generator(w_codes_for_gen, input_is_latent=True)[0] #8x3x1024x1024

          clipglobal_value = self.clipglobal(generated_from_w, self.text_inputs)
          self.losses['clipglobal'].append(clipglobal_value.item())

          w_optim.zero_grad()
          clipglobal_value.backward()
          w_optim.step()

          layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
          layer_numbers = torch.topk(layer_weights, self.layers_to_train)[1].cpu().numpy()

        #end of opt layers

        # разморозим только нужное нам
        for i in layer_numbers:
          for p in self.trainable_layers[i].parameters():
            p.requires_grad = True

    def forward(self):

        for step in tqdm(range(self.num_steps)):

            gc.collect()
            torch.cuda.empty_cache()

            self.generator.train()

            direct_sample_z = torch.randn(self.pic_batch, 512, device=self.device)

            # морозим все
            self.freeze_all_layers()
            # opt layers
            self.unfreeze_opt_layers()

            if not self.losses['clipglobal']:
              self.losses['clipglobal'] = [0]

            with torch.no_grad():
              latent = [self.fixed_generator.style(direct_sample_z)]
              frozen_image = self.fixed_generator(latent, input_is_latent=True)[0]

            image = self.generator(latent, input_is_latent=True)[0] # []


            clipdirect_value = self.clipdirect(image, self.start_text_inputs, frozen_image, self.text_inputs)
            self.losses['clipdirect'].append(clipdirect_value.item())

            # use cosine simularity (global loss) as a metric
            with torch.no_grad():
              clipglobal_metric = self.clipglobal(image, self.text_inputs)
              self.losses['clipglobal_metrics'].append(1 - clipglobal_metric.item())

            self.generator.zero_grad()
            clipdirect_value.backward()
            self.generator_optimizer.step()


            # Визуализация и вывод потерь
            if step % 50 == 0 or step == self.num_steps - 1:  # Каждые 50 шагов и последний

                self.generator.eval()
                const_z_style = self.generator.style(self.const_z)
                image = self.generator([const_z_style], input_is_latent=True)[0]

                print(f"inversion_generated_z/Step [{step}/{self.num_steps}], Global_loss: {self.losses['clipglobal'][-1]}, Direct_loss: {self.losses['clipdirect'][-1]}")  # Вывод текущих потерь

                fig, axs = plt.subplots(1, self.pic_sample, figsize=(16, 32))  # Создание подграфиков для визуализации
                for i in range(self.pic_batch):
                  axs[i].imshow((image.cpu().detach()[i].clip(-1,1).permute(1, 2, 0) + 1) / 2)
                  axs[i].axis('off')  # Отключение осей

                plt.show()
        return self.generator, self.fixed_generator, self.losses