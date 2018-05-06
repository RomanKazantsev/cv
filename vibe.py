import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from tqdm import tqdm


class Vibe:
    def __init__(self, sample_amount, radius, K, subsampling_time):
        """
        'Visual Background Extractor' algorithm of background subtraction

        :param sample_amount: number of samples per pixel
        :param radius: radius of the sphere
        :param K: number of close samples for being part of background
        :param subsampling_time: amount of random subsampling
        """
        self.sample_amount = sample_amount
        self.radius = radius
        self.K = K
        self.subsampling_time = subsampling_time
        self.samples = None


    def initialize(self, image):
        """
        Initialize the model with frame of video sequence.

        :param image: initializining frame
        """
        self.samples = np.zeros([self.sample_amount, image.shape[0], image.shape[1], image.shape[2]])
        inds_vert = np.arange(image.shape[0])
        inds_horiz = np.arange(image.shape[1])

        rows = np.arange(image.shape[0])
        rows = np.repeat(rows[:, np.newaxis], image.shape[1], axis=1)

        columns = np.arange(image.shape[1])
        columns = np.repeat(columns[np.newaxis, :], image.shape[0], axis=0)

        for i in range(0, self.sample_amount):
            # generate numbers for 8-neighbor connected set
            rel_rows = np.random.random_integers(-1, 1, rows.shape)
            rel_cols = np.random.random_integers(-1, 1, columns.shape)

            res_rows = np.add(rows, rel_rows)
            res_columns = np.add(columns, rel_cols)

            res_rows = np.clip(res_rows, 0, image.shape[0] - 1)
            res_columns = np.clip(res_columns, 0, image.shape[1] - 1)

            self.samples[i, :, :, :] = image[res_rows, res_columns, :]


    def apply(self, image):
        """
        Apply background subtraction algorithm to the next image,
        update internal parameters and return foreground mask.
        If model is not yet initialized, model must be initialized with this image.

        :param image: next image in video sequence
        :return: foreground mask
        """
        # initialize pixel model if it hasn't yet
        if self.samples is None:
            self.initialize(image)

        # predict foreground mask
        repeated_image = np.repeat(image[np.newaxis, :, : , :], self.sample_amount, axis=0)
        distances = np.sqrt(np.sum(np.square(np.subtract(repeated_image, self.samples)), axis=3))

        bg_fit_samples = np.zeros(distances.shape)
        bg_fit_samples[distances < self.radius] = 1
        sum_bg_fit_samples = np.sum(bg_fit_samples, axis=0)

        fg_mask = np.zeros(sum_bg_fit_samples.shape)
        fg_mask[sum_bg_fit_samples < self.K] = 255

        # update pixel model with temporal consistency
        num_bg_pixels = np.int(np.sum(fg_mask==0))
        ind_samples_for_replace = np.random.random_integers(0, self.sample_amount - 1, size=num_bg_pixels)
        rows_cols = np.argwhere(fg_mask==0)
        rows = rows_cols[:, 0]
        columns = rows_cols[:, 1]
        subsampling_inds = np.zeros(rows.shape, dtype=bool)
        subsampling_inds[np.random.random_sample(rows.shape[0]) < (1 / self.subsampling_time)] = True
        ind_samples_for_replace = ind_samples_for_replace[subsampling_inds]
        sub_rows = rows[subsampling_inds]
        sub_columns = columns[subsampling_inds]
        self.samples[ind_samples_for_replace, sub_rows, sub_columns, :] = image[sub_rows, sub_columns, :]

        # update pixel model with spatial consistency over 8 neighbors
        rel_rows = np.random.random_integers(-1, 1, rows.shape)
        rel_cols = np.random.random_integers(-1, 1, columns.shape)
        res_rows = np.clip(np.add(rows, rel_rows), 0, image.shape[0] - 1)
        res_columns = np.clip(np.add(columns, rel_cols), 0, image.shape[1] - 1)
        res_rows_cols = np.concatenate((res_rows[:, np.newaxis], res_columns[:, np.newaxis]), axis=1)
        res_rows_cols, res_rows_cols_inds = np.unique(res_rows_cols, return_index=True, axis=0)

        rows = rows[res_rows_cols_inds]
        columns = columns[res_rows_cols_inds]
        neigh_rows = res_rows_cols[:, 0]
        neigh_columns = res_rows_cols[:, 1]

        ind_samples_for_replace = np.random.random_integers(0, self.sample_amount - 1, size=len(rows))
        subsampling_inds = np.zeros(rows.shape, dtype=bool)
        subsampling_inds[np.random.random_sample(rows.shape[0]) < (1 / self.subsampling_time)] = True
        ind_samples_for_replace = ind_samples_for_replace[subsampling_inds]
        sub_rows = rows[subsampling_inds]
        sub_columns = columns[subsampling_inds]
        sub_neigh_rows = neigh_rows[subsampling_inds]
        sub_neigh_columns = neigh_columns[subsampling_inds]
        self.samples[ind_samples_for_replace, sub_neigh_rows, sub_neigh_columns, :] = image[sub_rows, sub_columns, :]

        return fg_mask


def image_generator(dirpath, first_frame=1, last_frame=None):
    """
    Generator of (frame_number, image, groundtruth) tuples.

    :param dirpath: Path to dir contained 'input' and 'groundtruth' subdirs
    :param first_frame: int, optional. Frame number from which the generator starts (inclusive)
    :param last_frame: int, optional. If provide, frame number  where the generator stops (inclusive)
    :return: (frame_number, image, groundtruth) tuples
    """

    input_format_name = 'input/in{:06d}.jpg'
    gt_format_name = 'groundtruth/gt{:06d}.png'

    numb = first_frame
    while (last_frame is None) or numb <= last_frame:
        input_path = os.path.join(dirpath, input_format_name.format(numb))
        gt_path = os.path.join(dirpath, gt_format_name.format(numb))

        if os.path.exists(input_path):
            input_image = skimage.io.imread(input_path)
            gt_image = skimage.io.imread(gt_path)
            if len(input_image.shape) == 2:
                input_image = input_image[..., np.newaxis]
            yield numb, input_image, gt_image
        else:
            break
        numb += 1
        

image_gen_tmp = image_generator('dataset/baseline/highway', 500, 700)
bg_substractor = Vibe(sample_amount=20, radius=50, K=3, subsampling_time=5)
for numb, frame, gt in tqdm(image_gen_tmp):
    mask = bg_substractor.apply(frame)

