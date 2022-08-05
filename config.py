import torch

class DefaultConfigs(object):
    #1.string parameters
    train_data = "./data/train/"
    #test_data = "./data/testcrop5测试epoch18/"
    test_data = "./data/MIAepoch240/"
    val_data = "./data/val/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")
    model_name0 = "net0"
    model_name1 = "net1"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0"
    upscale_factor = 2

    #2.numeric parameters
    epochs = 100
    batch_size = 32
    img_height = 448
    img_weight = 448
    num_classes = 2
    seed = 888
    lr = 1e-4
    lr_decay = 0.1
    weight_decay = 1e-4

config = DefaultConfigs()
