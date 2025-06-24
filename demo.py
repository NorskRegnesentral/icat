import os
import numpy as np
from PIL import Image
from icat.__main__ import run_icat

def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'demo')
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    n_imgs = 100
    img_size = (128, 128)
    sigma = 10
    files = []
    xy = np.zeros((n_imgs, 2), dtype=np.float32)

    for i in range(n_imgs):
        mus = np.random.uniform(0, 255, 3)
        img = np.random.normal(mus.reshape(1, 1, 3), sigma, size=img_size + (3,))
        img = np.clip(img, 0, 255).astype(np.uint8)
        img_path = os.path.join(img_dir, f'image_{i:04d}.png')
        Image.fromarray(img).save(img_path)
        files.append(img_path)
        xy[i, 0] = mus[0]  # R channel mean
        xy[i, 1] = mus[2]  # B channel mean
        print(img_path)

    np.savez(os.path.join(out_dir, 'data.npz'), files=np.array(files), xy=xy)

    run_icat(os.path.join(out_dir, 'data.npz'), classes=['blue', 'red'])

if __name__ == '__main__':
    main()
