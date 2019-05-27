# nnattacks

## Installation

### from GitHub

```bash
$ git clone https://github.com/nocotan/nnattacks.git
$ cd nnattacks
$ python setup.py install
```

### from PyPi

```bash
$ pip install nnattacks
```

## Basic Usage

```python
import PIL
import torch
from torchvision import transforms

from nnattacks.attacks import FastGradientSignTargeted

img = PIL.Image.open("<path-to-image>")
tensor_img = transforms.ToTensor(img).unsqueeze(0) # input tensor

y_target = torch.LongTensor([385]) # target label

fgsm_targeted = FastGradientSignTargeted()
x_adv, pertb = fgsm_targeted.generate_perturbation(
    model=model,
    x=tensor_img,
    y_target=y_target,
)

x_adv # adversarial example
pertb # adversarial perturbation
```

## Attacks

* FastGradientSignTargeted
* FastGradientSignUntargeted

## References
* Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).
