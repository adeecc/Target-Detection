import os

path = 'mod'

for count, name in enumerate(os.listdir(path)):
    dst = os.path.join(path, f"mod_{count}.jpg")
    src = os.path.join(path, name)

    print(f"{src} -> {dst}")
    os.rename(src, dst)