# -*- coding: utf-8 -*-
import PIL
import tqdm
import torch
from torchvision import transforms
from torchvision import models

from nnattacks.attacks import FastGradientSignTargeted


def main():
    img = PIL.Image.open("../images/cavy.jpg")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    x = transform(img).unsqueeze(0)
    # x.requires_grad = True

    inceptionv3 = models.inception_v3(pretrained=True)
    inceptionv3.eval()

    y_true = torch.max(inceptionv3(x), 1)[1]
    y_target = torch.LongTensor([385])

    fgsm_targeted = FastGradientSignTargeted(n_iter=10)
    x_adv, pertb = fgsm_targeted.generate_perturbation(
        inceptionv3,
        x,
        y_target)

    y_adv = torch.max(inceptionv3(x_adv), 1)[1][0]

    print(y_true)
    print(y_adv)


if __name__ == "__main__":
    main()
