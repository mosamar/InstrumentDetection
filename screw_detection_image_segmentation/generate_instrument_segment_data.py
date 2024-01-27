import os

from synthetic_data_generation.generate_synthetic_data_segmentation import generate_training_data, arrange
from visualization.visualize_segmentation import plot_random_samples

if __name__ == '__main__':
    data_dir = '../data'
    dataset_dir = './datasets'
    generate_training_data(data_dir, dataset_dir, screw=True)
    arrange()
    plot_random_samples()
