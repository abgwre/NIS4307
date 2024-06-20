from classify import ViolenceClass
from model import load_images_from_folder

if __name__ == "__main__":

    #路径根据图片位置更改
    folder_path = 'violence_224/test'
    img_list = load_images_from_folder(folder_path)


    v = ViolenceClass()
    preds = v.classify(img_list)

    print(preds)