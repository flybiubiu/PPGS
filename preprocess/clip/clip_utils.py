import os
from torchvision import transforms

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

def get_image_paths(directory):
    image_paths = []
    valid_image_extensions = [".jpg", ".JPG" ".jpeg", ".png", ".gif", ".bmp"]
    #print(os.walk(directory))<generator object walk at 0x7819aaae9d60>
    #print(directory)/home/ps/Desktop/2t/LEGS/mipnerf360/kitchen/images

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_image_extensions):
                image_paths.append(os.path.join(root, file))

    return sorted(image_paths)