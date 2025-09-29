import os

script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, "../data")

NYU40_MAPPING_FILE = os.path.join(data_dir, "nyu40_segmentation_mapping.csv")
HABITAT_CATEGORY_MAPPING_FILE = os.path.join(data_dir, "matterport_category_mappings.tsv")