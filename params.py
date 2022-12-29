class params():

    bpm_flag = False
    k_fold_flag = False
    model_save_flag = False
    log_flag = True
    wandb_flag = True
    random_seed = 0
    save_img_flag = False

    __TIME__ = 0  # 1: print time, 0: not print time
    __PREPROCESSING__ = 1  # 1: preprocessing, 0: not preprocessing
    __MODEL_SUMMARY__ = 0  # 1: print model summary, 0: not print model summary

    model = "DeepPhys"
    model_list = ["DeepPhys", "DeepPhys_DA", "PhysNet", "PhysNet_LSTM", "PPNet", "GCN", "AxisNet", "RhythmNet"]

    # preprocessing parameters
    # dataset
    dataset_name = "UBFC"
    # UBFC / PURE / V4V / VIPL_HR
    save_root_path = "/media/hdd1/dy_dataset/"
    data_root_path = "/media/hdd1/"
    model_root_path = "/media/hdd1/dy/model/"
    train_ratio = 0.8
    face_detect_algorithm = 0
    divide_flag = 1  # 1 : divide by number 0: divide by subject
    fixed_position = 1  # 1 : fixed position 0: not fixed position
    time_length = 32  # The number of frames in dataset.__GetItem__
    chunk_size = 4  # The number of subjects processed at one time

    # train paramaters

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_batch_size = 32
    train_shuffle = 1

    test_batch_size = 32
    test_shuffle = 1

    loss_fn = "fft"
    '''
    "mse","L1","neg_pearson","multi_margin","bce","huber","cosine_embedding",
                   "cross_entropy","ctc","bce_with_logits","gaussian_nll","hinge_embedding",
                   "KLDiv","margin_ranking","multi_label_margin","multi_label_soft_margin",
                   "nll","nll2d","pairwise","poisson_nll","smooth_l1","soft_margin",
                   "triplet_margin","triplet_margin_distance",
                   "PPNET : MSE"
    '''
    optimizer = "adam"
    '''
    "adam","sgd","rms_prop","ada_delta","ada_grad","ada_max",
                    "ada_mw","a_sgd","lbfgs","n_adam","r_adam","rprop","sparse_adam",
                    "PPNET : adam"
    '''
    lr = 0.001
    '''
   "DeepPhys : lr = 1",
    "PhysNet : lr = 0.001",
    "PPNet : lr = 0.001",
    "GCN : lr = 0.003"
    '''
    epoch = 500

    #wandb params
    wandb_project_name = "SeqNet"
    wandb_entity = "daeyeolkim"


params = params()
