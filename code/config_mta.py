class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def config_mta():

    config = Config({

        #数据集
        "dataset_name": "JUFE_10K", # JUFE_10K; OIQ_10K; OIQA; CVIQ;
        
        # optimization
        "batch_size": 8,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 50,
        "val_freq": 1,
        "num_workers": 32,
        "split_seed": 0,

        # model
        "embed_dim": 128,

        # 多卡训练
        "CUDA_VISIBLE_DEVICES": "0",
        "device_ids": [0],

        # 是否生成预测文件
        "print_pred_file": False,
        
        # 训练输出文件
        "train_out_file":"../file_pred_train/6-13-qrd-JUFE-10K-output_train_quality.csv",
        "train_out_type_file":"../file_pred_train/6-13-qrd-JUFE-10K-output_train_type.csv",
        "train_out_range_file":"../file_pred_train/6-13-qrd-JUFE-10K-output_train_range.csv",
        "train_out_degree_file":"../file_pred_train/6-13-qrd-JUFE-10K-output_train_degree.csv",

        # 测试输出文件
        "test_out_file":"../file_pred_test/6-13-qrd-JUFE-10K-output_test_quality.csv",
        "test_out_type_file":"../file_pred_test/6-13-qrd-JUFE-10K-output_test_type.csv",
        "test_out_range_file":"../file_pred_test/6-13-qrd-JUFE-10K-output_test_range.csv",
        "test_out_degree_file":"../file_pred_test/6-13-qrd-JUFE-10K-output_test_degree.csv",

        # 数据库失真的种类 JUFE-10K: t:4, r:2, d:3;  OIQ-10K: t:1 r:4 d:1;(1:没有相关类别)  OIQA: t:4, d:5; CVIQ: t:3, d:11
        "type_num": 4,
        "range_num": 2,
        "degree_num": 3,

        # 可选任务，全部False表示只有quality一个分支
        "qrtd": True,
        "qrt": False,
        "qrd": False,
        "qtd": False,
        "qr": False,
        "qt":False,
        "qd": False,

        # 是否加载预训练
        "pretrained": True,
        # 预训练权重路径
        "model_weight_path": "/mnt/10T/rjl/Multitask_Auxiliary_OIQA/pt/qrtd/4-12-JUFE-10K-qrtd/best_ckpt_23.pt",

        # 训练、测试数据
        # "train_dataset": "/mnt/10T/rjl/Multitask_Auxiliary_OIQA/file/JUFE-10K/train_final_viewport8_mos_type_range_degree_1.csv",
        "test_dataset": "/mnt/10T/rjl/Multitask_Auxiliary_OIQA/file/JUFE-10K/test_final_viewport8_mos_type_range_degree_1.csv",
        # "train_dataset": "/mnt/10T/rjl/Multitask_Auxiliary_OIQA/file/OIQ-10K/OIQ-10K_train_info.csv",
        # "test_dataset": "/mnt/10T/rjl/Multitask_Auxiliary_OIQA/file/OIQ-10K/OIQ-10K_test_info.csv",
        # "train_dataset": "/home/d310/10t/rjl/Multitask_Auxiliary_OIQA/file/OIQA/train_OIQA_viewport_8.csv",
        # "test_dataset": "/home/d310/10t/rjl/Multitask_Auxiliary_OIQA/file/OIQA/test_OIQA_viewport_8.csv",
        # "train_dataset": "/home/d310/10t/rjl/Multitask_Auxiliary_OIQA/file/CVIQ/train_dis_CVIQ_viewport_8.csv",
        # "test_dataset": "/home/d310/10t/rjl/Multitask_Auxiliary_OIQA/file/CVIQ/test_dis_CVIQ_viewport_8.csv",

        # "train_dataset": "/home/d310/10t/rjl/Multitask_Auxiliary_OIQA/file/OIQ-10K/OIQ-10K_train_info.csv",
        # "test_dataset": "/home/d310/10t/rjl/Multitask_Auxiliary_OIQA/file/OIQ-10K/OIQ-10K_test_info.csv",


        # load & save checkpoint
        "model_name": "11-8-JUFE-10K-q",
        "type_name": "q",
        "ckpt_path": "/mnt/10T/rjl/Multitask_Auxiliary_OIQA/pt",               # directory for saving checkpoint
        "log_path": "/mnt/10T/rjl/Multitask_Auxiliary_OIQA/output",
    })
    
        
    return config