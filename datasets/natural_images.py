from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import random
    import glob
    import torch
    import torchvision as tv
    import imageio.v3 as iio
    from benchmark_utils import inv_problems, general_utils


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "natural_images"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "n_samples": [4],
        "sigma": [0.01],
        "random_state": [27],
        "extension": ["png"],
        "inv_problem": ["inpainting"],
        "noise_model": ["gaussian"],
        "blur_sd": [(0.75, 0.75)],
        "prop_inpaint": [0.5],
        "img_size" : [64],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        image_path = "./benchmark_sampling/data/images/BSD/train/"

        device = general_utils.get_best_device()

        print("Using torch device: ", device)

        # Generate pseudorandom data using `numpy`.
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # Load the data
        file_list = list(glob.glob(image_path + "*." + self.extension))
        import os

        random.shuffle(file_list)

        # Load images into a list
        gt_img_list = []  # torch.zeros(self.n_samples)
        for it in range(self.n_samples):
            gt_img = np.array(iio.imread(file_list[it]))
            # Scale to [0,1]
            gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min())

            gt_img_list.append(gt_img)

        x_true = torch.tensor(np.array(gt_img_list), dtype=torch.float32, device=device)

        

        # Crop image to [img_size x img_size]
        if x_true.shape[-1] > self.img_size and self.img_size != 0:
            x_true = tv.transforms.CenterCrop(self.img_size)(x_true)

        # Add new channel dimension to 1
        x_true = x_true[:, None, :, :]

        # Define the forward model
        physics = inv_problems.define_physics(
            inv_problem=self.inv_problem,
            noise_model=self.noise_model,
            sigma=self.sigma,
            blur_sd=self.blur_sd,
            prop_inpaint=self.prop_inpaint,
            img_size=x_true.size()[-3:],
        )  # Eventually add more parameters required for other inverse problems

        # Generate the observations
        y = physics(x_true)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(x_true=x_true, y=y, physics=physics)
