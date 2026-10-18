from models.TSViT.TSViTdense import TSViT, TSViT_DINOv3
from models.TSViT.TSViTcls import TSViTcls, TSViTcls_DINOv3


def get_model(config, device):
    model_config = config['MODEL']
    arch = model_config['architecture']

    if arch in ["TSViTcls", "TSViTcls_DINOv3", "TSViTcls-DINOv3"]:
        model_config['device'] = device
        return TSViTcls(model_config).to(device)

    elif arch in ["TSViT", "TSViT_DINOv3", "TSViT-DINOv3"]:
        return TSViT(model_config).to(device)

    else:
        raise NameError(
            f"Model architecture '{arch}' not recognized. "
            f"Available architectures: 'TSViT', 'TSViT_DINOv3', 'TSViTcls', 'TSViTcls_DINOv3'"
        )
