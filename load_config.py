from typing import Dict
from optparse import OptionParser

from .data_io import read_conf, str_to_bool


def load(config_file_path):
    options = read_conf(['--cfg', config_file_path])

    parser = OptionParser()
    cfg, _ = parser.parse_args([])

    # [data]
    cfg.tr_lst = options.tr_lst
    cfg.te_lst = options.te_lst
    cfg.pt_file = options.pt_file
    cfg.class_dict_file = options.lab_dict
    cfg.data_folder = options.data_folder.rstrip('/') + '/'
    cfg.output_folder = options.output_folder.rstrip('/') + '/'

    # [windowing]
    cfg.fs = int(options.fs)
    cfg.cw_len = int(options.cw_len)
    cfg.cw_shift = int(options.cw_shift)

    # [cnn]
    cfg.cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
    cfg.cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
    cfg.cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
    cfg.cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
    cfg.cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
    cfg.cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
    cfg.cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
    cfg.cnn_act = list(map(str, options.cnn_act.split(',')))
    cfg.cnn_drop = list(map(float, options.cnn_drop.split(',')))

    # [dnn]
    cfg.fc_lay = list(map(int, options.fc_lay.split(',')))
    cfg.fc_drop = list(map(float, options.fc_drop.split(',')))
    cfg.fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
    cfg.fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
    cfg.fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
    cfg.fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
    cfg.fc_act = list(map(str, options.fc_act.split(',')))

    # [class]
    cfg.class_lay = list(map(int, options.class_lay.split(',')))
    cfg.class_drop = list(map(float, options.class_drop.split(',')))
    cfg.class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
    cfg.class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
    cfg.class_use_batchnorm = list(
        map(str_to_bool, options.class_use_batchnorm.split(','))
    )
    cfg.class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
    cfg.class_act = list(map(str, options.class_act.split(',')))

    # [optimization]
    cfg.lr = float(options.lr)
    cfg.batch_size = int(options.batch_size)
    cfg.N_epochs = int(options.N_epochs)
    cfg.N_batches = int(options.N_batches)
    cfg.N_eval_epoch = int(options.N_eval_epoch)
    cfg.seed = int(options.seed)

    return cfg

def make_sinc_net_init_options(config) -> Dict:
    options = config

    fs = options.fs
    cw_len = options.cw_len

    cnn_N_filt = options.cnn_N_filt
    cnn_len_filt = options.cnn_len_filt
    cnn_max_pool_len = options.cnn_max_pool_len
    cnn_use_laynorm_inp = options.cnn_use_laynorm_inp
    cnn_use_batchnorm_inp = options.cnn_use_batchnorm_inp
    cnn_use_laynorm = options.cnn_use_laynorm
    cnn_use_batchnorm = options.cnn_use_batchnorm
    cnn_act = options.cnn_act
    cnn_drop = options.cnn_drop

    # converting context and shift in samples
    wlen = int(fs * cw_len / 1000.00)

    init_arg = {
        'input_dim': wlen,
        'fs': fs,
        'cnn_N_filt': cnn_N_filt,
        'cnn_len_filt': cnn_len_filt,
        'cnn_max_pool_len': cnn_max_pool_len,
        'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
        'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
        'cnn_use_laynorm': cnn_use_laynorm,
        'cnn_use_batchnorm': cnn_use_batchnorm,
        'cnn_act': cnn_act,
        'cnn_drop': cnn_drop,
    }
    
    return init_arg