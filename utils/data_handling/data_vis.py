import matplotlib.pyplot as plt
from torchvision import transforms as T


def plot_img_and_mask(img, mask):

    fig, ax = plt.subplots(1, 2)
    img = T.ToPILImage()(img).convert('LA')
    mask = T.ToPILImage()(mask)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title(f'Output mask')
    ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
