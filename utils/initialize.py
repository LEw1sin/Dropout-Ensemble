import logging

def setup_logging(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_log_dir(net, dataset_mode, train_eval='', ed_es_only=''):
        from collections import Counter
        from datetime import datetime
        import os
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        type_ = set()
        model_list = []
        for model in net.models:
            model_name = f"{model.__class__.__name__}"
            model_list.append(model_name.split('_')[0]+model_name.split('_')[1])
            type_.add(model_name.split('_')[1])
        count = Counter(model_list)
        result = '_'.join([f"{v}{k}" for k, v in count.items()])
        log_dir_suffix = datetime.now().strftime("%m-%d-%H-%M-%S")
        train_eval = f'{train_eval}_' if train_eval else ''
        ed_es_only = f'_{ed_es_only}' if ed_es_only else ''
        log_dir = f'../../de_logistics/{train_eval}{dataset_mode}{ed_es_only}_{result}_{log_dir_suffix}/'
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    


