from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='../data/caption_data/dataset_coco_kor.json',
                       image_folder='../data/captioin_data',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='outputs/',
                       max_len=50)
