from torchvision.models import (
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
    EfficientNet,
)


def create_model(pretrained: bool = True) -> EfficientNet:
    if pretrained:
        model = efficientnet_v2_m(
            weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1
        )
    else:
        model = efficientnet_v2_m()

    return model
