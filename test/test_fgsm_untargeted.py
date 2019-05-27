import PIL
import torch
from torchvision import transforms
from torchvision import models

from nnattacks.attacks import FastGradientSignUntargeted


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

    inceptionv3 = models.inception_v3(pretrained=True)
    inceptionv3.eval()

    y_true = torch.max(inceptionv3(x), 1)[1]

    fgsm_targeted = FastGradientSignUntargeted(n_iter=10)
    x_adv, pertb = fgsm_targeted.generate_perturbation(
        inceptionv3,
        x,
        y_true)

    y_adv = torch.max(inceptionv3(x_adv), 1)[1][0]

    print(y_true)
    print(y_adv)


if __name__ == "__main__":
    main()
