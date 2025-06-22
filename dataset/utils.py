DATA_dic = {
    'RSVQA_LR':{
        'train': {
            'image_folder_path': "DATA/RSVQA_LR/images",
            'questions_json_path': "DATA/RSVQA_LR/train/LR_split_train_questions.json",
            'answers_json_path': "DATA/RSVQA_LR/train/LR_split_train_answers.json",
            'images_json_path': "DATA/RSVQA_LR/train/LR_split_train_images.json"
        },
        'test': {
            'image_folder_path': "DATA/RSVQA_LR/images",
            'questions_json_path': "DATA/RSVQA_LR/test/LR_split_test_questions.json",
            'answers_json_path': "DATA/RSVQA_LR/test/LR_split_test_answers.json",
            'images_json_path': "DATA/RSVQA_LR/test/LR_split_test_images.json"
        },
        'image_size': (256, 256)
        },
    'RSVQA_HR':{
        'train': {
            'image_folder_path': "DATA/RSVQA_HR/images",
            'questions_json_path': "DATA/RSVQA_HR/train/USGS_split_train_questions.json",
            'answers_json_path': "DATA/RSVQA_HR/train/USGS_split_train_answers.json",
            'images_json_path': "DATA/RSVQA_HR/train/USGS_split_train_images.json"
        },
        'test': {
            'image_folder_path': "DATA/RSVQA_HR/images",
            'questions_json_path': "DATA/RSVQA_HR/test/USGS_split_test_questions.json",
            'answers_json_path': "DATA/RSVQA_HR/test/USGS_split_test_answers.json",
            'images_json_path': "DATA/RSVQA_HR/test/USGS_split_test_images.json"
            },
            'image_size': (512, 512)
        },
    'RSIVQA': {
        'image_size': (768, 768),
        'AID': {
            'image_folder_path': "DATA/RSIVQA/AID/images",
            'qa_train_path': "DATA/RSIVQA/AID/rsicd_vqa.txt",
        },
        'DOTA': {
            'image_folder_path': "DATA/RSIVQA/DOTA/images",
            'qa_train_path': "DATA/RSIVQA/DOTA/dota_train_vqa2.txt",
            'qa_test_path': "DATA/RSIVQA/DOTA/dota_val_vqa2.txt",
        },
        'HRRSD': {
            'image_folder_path': "DATA/RSIVQA/HRRSD/images",
            'qa_train_path': "DATA/RSIVQA/HRRSD/opt_val_vqa.txt",
        },
        'Sydney': {
            'image_folder_path': "DATA/RSIVQA/Sydney/images",
            'qa_train_path': "DATA/RSIVQA/Sydney/sydney_vqa.txt",
        }
    }
}


def load_dataset(model_name, dataset_name, is_eval, add_instruct, load_num, type, processor, task='all'):
    assert dataset_name in DATA_dic, f"Dataset {dataset_name} not found in DATA_dic."
    dataset_info = DATA_dic[dataset_name]
    if dataset_name == 'RSVQA_LR':
        from dataset.RSVQA_LR_Dataset import RSVQA_LR_Dataset
        return RSVQA_LR_Dataset(
            processor=processor,
            model_name=model_name,
            image_folder_path=dataset_info[type]['image_folder_path'],
            questions_json_path=dataset_info[type]['questions_json_path'],
            answers_json_path=dataset_info[type]['answers_json_path'],
            images_json_path=dataset_info[type]['images_json_path'],
            image_size=dataset_info['image_size'],
            use_num=load_num,
            add_instruct=add_instruct,
            is_eval=is_eval,
            task=task,
        )
    elif dataset_name == 'RSVQA_HR':
        from dataset.RSVQA_HR_Dataset import RSVQA_HR_Dataset
        return RSVQA_HR_Dataset(
            processor=processor,
            model_name=model_name,
            image_folder_path=dataset_info[type]['image_folder_path'],
            questions_json_path=dataset_info[type]['questions_json_path'],
            answers_json_path=dataset_info[type]['answers_json_path'],
            images_json_path=dataset_info[type]['images_json_path'],
            image_size=dataset_info['image_size'],
            use_num=load_num,
            add_instruct=add_instruct,
            is_eval=is_eval,
            task=task,
        )
    elif dataset_name == 'RSIVQA':
        from dataset.RSIVQA_Dateset import CombinedVQADataset
        assert isinstance(load_num, dict), "load_num should be a dictionary with 'train' and 'test' keys."
        task_list = ['YesOrNo', 'Number', 'Other']
        if isinstance(load_num['train'], list):
            load_num['train'] = {task: load_num['train'][i] for i, task in enumerate(task_list)}
        if isinstance(load_num['test'], list):
            load_num['test'] = {task: load_num['test'][i] for i, task in enumerate(task_list)}
        combined_dataset_loader_scenario = CombinedVQADataset(
            processor=processor,
            image_size=dataset_info['image_size'],
            use_num=load_num,
            task=task,
            add_instruct=add_instruct,
            split_save_path='DATA/RSIVQA/split_data.json',
            model_name=model_name
        )
        for key, value in dataset_info.items():
            if key == 'image_size':
                continue
            if key == 'DOTA':
                is_split = True
            else:
                is_split = False
            combined_dataset_loader_scenario.add_dataset(
                image_folder_path=value['image_folder_path'],
                qa_file_path_train=value.get('qa_train_path'),
                qa_file_path_test=value.get('qa_test_path'),
                is_split=is_split
            )
        
        # combined_dataset_loader_scenario.save_splits()
        train_dataset = combined_dataset_loader_scenario.get_dataset(split='train')
        test_dataset = combined_dataset_loader_scenario.get_dataset(split='test')
        print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} testing samples from RSIVQA dataset.")
        return train_dataset, test_dataset
        