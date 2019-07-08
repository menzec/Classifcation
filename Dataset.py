import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


def createlist():
    '''
    在 train、val 文件夹下生成训练样本列表 listfile
    listfile 中文件的类别号为ClsName2id.txt 类别号减 1
    '''
    ClsName2id = r'G:\data\rssrai2019_scene_classification\ClsName2id.txt'
    classdict = dict()
    imageroot = r'G:\data\rssrai2019_scene_classification\val'
    listfile = r'G:\data\rssrai2019_scene_classification\val\list.txt'
    listfn = open(listfile, 'w')
    for classinfo in open(ClsName2id, encoding='UTF-8'):
        classinfo = classinfo.strip().split(':')
        classdict[classinfo[0]] = classinfo[1:]
    for cls in os.listdir(imageroot):
        if os.path.isdir(os.path.join(imageroot,
                                      cls)) and cls in classdict.keys():
            clsid = int(classdict[cls][1]) - 1
        else:
            print("%s not in ClsName2id.txt." % (os.path.join(imageroot, cls)))
            continue
        for img in os.listdir(os.path.join(imageroot, cls)):
            listfn.write("%s/%s %d\n" % (cls, img, clsid))
        '''
        生成info信息，每种类别的数量
        '''
        # listfn.write("%s,%s,num,%d,id,%d\n" %
        #              (cls, classdict[cls][0],
        #               len(os.listdir(os.path.join(imageroot, cls))), clsid))
    listfn.close()


class ImageFolder(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        fh = open(txt, 'r', encoding='UTF-8')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.root = os.path.split(txt)[0]

    def loader(self, image_name):
        return Image.open('%s/%s' % (self.root, image_name))

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class Sencedata():
    def __init__(self, root, BATCH_SIZE, ClsName2id='ClsName2id.txt'):
        self.root = root
        self.datafolder = ['/train/list.txt', '/val/list.txt']
        self.batch_size = BATCH_SIZE
        self.ClsName2id = os.path.join(self.root, ClsName2id)

    def getdata(self, training=False):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.5, 1.0), ratio=(0.75, 1.3)),
            transforms.RandomHorizontalFlip(p=0.5),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4842, 0.4901, 0.4505),
                (0.2168, 0.2013, 0.1948)),  # R,G,B每层的归一化用到的均值和方差
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4842, 0.4901, 0.4505),
                                 (0.2168, 0.2013, 0.1948)),
        ])
        train_data = ImageFolder(
            self.root + self.datafolder[0], transform=transform_train)
        test_data = ImageFolder(
            self.root + self.datafolder[1], transform=transform_test)
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4)
        test_loader = DataLoader(
            dataset=test_data, batch_size=self.batch_size, num_workers=2)
        return train_loader, test_loader

    def label(self):
        return [
            classinfo.split(':')
            for classinfo in open(self.ClsName2id, encoding='UTF-8')
        ]


if __name__ == "__main__":
    dataroot = r'G:\data\rssrai2019_scene_classification'
    traindata, testdata = Sencedata(dataroot, 10).getdata()
    print(len(traindata), len(testdata))
    for data in traindata:
        print(data[0].shape)
        break
    for data in testdata:
        print(data[0].shape)
        break