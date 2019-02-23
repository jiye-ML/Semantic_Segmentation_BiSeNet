import sys, os
import subprocess

sys.path.append("models")
from models.BiSeNet import build_bisenet

SUPPORTED_MODELS = ["FC-DenseNet56", "BiSeNet"]

SUPPORTED_FRONTENDS = ["ResNet50", "ResNet101", "ResNet152"]

# 下载与训练的ResNet模型
def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])

# 建立 BisNet模型，前端使用 ResNet
def build_model(model_name, net_input, num_classes, frontend="ResNet101", is_training=True):
    print("Preparing the model ...")

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            "The model you selected is not supported. The following models are currently supported: {0}".format(
                SUPPORTED_MODELS))
    # 前端
    if frontend not in SUPPORTED_FRONTENDS:
        raise ValueError(
            "The frontend you selected is not supported. The following models are currently supported: {0}".format(
                SUPPORTED_FRONTENDS))

    # 基础模型
    if "ResNet50" == frontend and not os.path.isfile("models/resnet_v2_50.ckpt"):
        download_checkpoints("ResNet50")
    if "ResNet101" == frontend and not os.path.isfile("models/resnet_v2_101.ckpt"):
        download_checkpoints("ResNet101")

    if model_name == "BiSeNet":
        # BiSeNet requires pre-trained ResNet weights
        network, init_fn = build_bisenet(net_input, frontend=frontend, num_classes=num_classes, is_training=is_training)
    else:
        raise ValueError(
            "Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

    return network, init_fn
