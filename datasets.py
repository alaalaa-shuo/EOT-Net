import torch.utils.data
import scipy.io as sio
import torchvision.transforms as transforms
import torch.nn.functional as F


class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, target, transform=None, target_transform=None):
        self.img = img.float()
        self.target = target.float()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img)


class Data:
    def __init__(self, dataset, device):
        super(Data, self).__init__()

        data_path = "./data/" + dataset + "_dataset.mat"
        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95
            data_path = "./data/samson_data.mat"
            bundle_path = './data/bundles_samson.mat'
        elif dataset == 'jasper':
            self.P, self.L, self.col = 4, 198, 100
            bundle_path = './data/bundles_ridge.mat'

        data = sio.loadmat(data_path)
        self.Y = torch.from_numpy(data['Y'].T)
        # N*L
        self.A = torch.from_numpy(data['A'].T)
        # N*P
        self.M = torch.from_numpy(data['M'])
        # GroundTruth L*P
        self.M1 = torch.from_numpy(data['M1'])
        # Initial Endmember L*P
        bundle_data = sio.loadmat(bundle_path)
        bundle = bundle_data['bundleLibs']
        self.bundle = torch.from_numpy(bundle).permute(1, 2, 0).contiguous()
        # BundleNum*P*L -> P*L*BundleNum

    def get(self, typ):
        if typ == "hs_img":
            return self.Y.float()
        elif typ == "abd_map":
            return self.A.float()
        elif typ == "end_mem":
            return self.M
        elif typ == "init_weight":
            return self.M1
        elif typ == "bundle":
            return self.bundle

    def get_loader(self, batch_size=1):
        train_dataset = TrainData(img=self.Y, target=self.A, transform=transforms.Compose([]))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)

        return train_loader

