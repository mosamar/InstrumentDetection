from synthetic_data_generation.generate_veins import generate_training_data
from synthetic_data_generation.utils import arrange
from visualization.visualize_segmentation import plot_random_samples

if __name__ == '__main__':
    data_dir = '../data'
    dataset_dir = './datasets'
    generate_training_data(data_dir, dataset_dir)
    arrange()
    plot_random_samples()

