from synthetic_data_generation.generate_synthetic_data_bounding_box import generate_bbox_data
from synthetic_data_generation.utils import arrange

if __name__ == '__main__':
    data_dir = '../data'
    dataset_dir = './datasets'
    generate_bbox_data(data_dir, dataset_dir)
    arrange()
