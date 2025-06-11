DATA_dic = {
    'RSVQA_LR':{
        'train': {
            'image_folder_path': "DATA/RSVQA-LR/images",
            'questions_json_path': "DATA/RSVQA-LR/train/LR_split_train_questions.json",
            'answers_json_path': "DATA/RSVQA-LR/train/LR_split_train_answers.json",
            'images_json_path': "DATA/RSVQA-LR/train/LR_split_train_images.json"
        },
        'test': {
            'image_folder_path': "DATA/RSVQA-LR/images",
            'questions_json_path': "DATA/RSVQA-LR/test/LR_split_test_questions.json",
            'answers_json_path': "DATA/RSVQA-LR/test/LR_split_test_answers.json",
            'images_json_path': "DATA/RSVQA-LR/test/LR_split_test_images.json"
        },
        'image_size': (256, 256)
    }
}

def load_dataset(dataset_name, is_eval, add_instruct, load_num, type, processor, task='all'):
    assert dataset_name in DATA_dic, f"Dataset {dataset_name} not found in DATA_dic."
    dataset_info = DATA_dic[dataset_name]
    if dataset_name == 'RSVQA_LR':
        from dataset.RSVQA_LR_Dataset import RSVQA_LR_Dataset
        return RSVQA_LR_Dataset(
            processor=processor,
            image_folder_path=dataset_info[type]['image_folder_path'],
            questions_json_path=dataset_info[type]['questions_json_path'],
            answers_json_path=dataset_info[type]['answers_json_path'],
            images_json_path=dataset_info[type]['images_json_path'],
            image_size=dataset_info['image_size'],
            use_num=load_num,
            add_instruct=add_instruct,
            is_eval=is_eval,
            task=task
        )