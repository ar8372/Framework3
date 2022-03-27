# --> to pass it to pytorch dataloader
# > __init__
# > __len__
# > __getitem__

# each competition may require different
# preprocessing and scaling

from torch.utils.data import Dataset


class BengaliDataset(Dataset):
    def __init__(self, csv, img_height, img_width, transform):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
            self.locker = pickle.load(f)

        self.csv = csv.reset_index()
        self.img_ids = csv[self.locker["id_name"]]
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __get__(self, index):
        img_id = self.img_ids[idx]
        img = joblib.load(
            f"../input_{self.locker['comp_name']}/train_images/{img_id}.pkl"
        )

        # reshape
        img = img.reshape(self.img_height, self.img_width).astype(np.uint8)
        img = 255 - img  #

        # make it 3dimensional (X,Y, RGB) if not
        img = img[:, :, np.newaxis]

        # np.repeat(item, no_times, along axis)
        # it repeats item along an axis
        # duplicates whole image 3 times to create RGB channels
        img = img[img, 3, 2]

        if self.transform is not None:
            img = self.transfrom(image=img)["image"]
        target_list = self.locker["target_name"]
        target_1 = self.csv.iloc[index][target_list[0]]
        target_2 = self.csv.iloc[index][target_list[1]]
        target_3 = self.csv.iloc[index][target_list[2]]

        return img, np.array([target_1, target_2, target_3])


class CutMixImageDataGenerator:
    def __init__(self, generator1, generator2, img_size, batch_size):
        self.batch_index = 0
        self.samples = generator1.samples
        self.class_indices = generator1.class_indices
        self.generator1 = generator1
        self.generator2 = generator2
        self.img_size = img_size
        self.batch_size = batch_size

    def reset_index(self):  # Ordering Reset (If Shuffle is True, Shuffle Again)
        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def reset(self):
        self.batch_index = 0
        self.generator1.reset()
        self.generator2.reset()
        self.reset_index()

    def get_steps_per_epoch(self):
        quotient, remainder = divmod(self.samples, self.batch_size)
        return (quotient + 1) if remainder else quotient

    def __len__(self):
        self.get_steps_per_epoch()

    def __next__(self):
        if self.batch_index == 0:
            self.reset()

        crt_idx = self.batch_index * self.batch_size
        if self.samples > crt_idx + self.batch_size:
            self.batch_index += 1
        else:  # If current index over number of samples
            self.batch_index = 0

        reshape_size = self.batch_size
        last_step_start_idx = (self.get_steps_per_epoch() - 1) * self.batch_size
        if crt_idx == last_step_start_idx:
            reshape_size = self.samples - last_step_start_idx

        X_1, y_1 = self.generator1.next()
        X_2, y_2 = self.generator2.next()

        cut_ratio = np.random.beta(a=1, b=1, size=reshape_size)
        cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
        label_ratio = cut_ratio.reshape(reshape_size, 1)
        cut_img = X_2

        X = X_1
        for i in range(reshape_size):
            cut_size = int((self.img_size - 1) * cut_ratio[i])
            y1 = random.randint(0, (self.img_size - 1) - cut_size)
            x1 = random.randint(0, (self.img_size - 1) - cut_size)
            y2 = y1 + cut_size
            x2 = x1 + cut_size
            cut_arr = cut_img[i][y1:y2, x1:x2]
            cutmix_img = X_1[i]
            cutmix_img[y1:y2, x1:x2] = cut_arr
            X[i] = cutmix_img

        y = y_1 * (1 - (label_ratio**2)) + y_2 * (label_ratio**2)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)
