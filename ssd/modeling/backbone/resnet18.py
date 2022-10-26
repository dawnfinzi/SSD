# Inspired by this ResNet50 implementation: https://github.com/lufficc/SSD/pull/160
from ssd.modeling import registry

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class ResNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        # load pytorch backbone
        backbone = resnet18(pretrained=pretrained)

        # create a list of all layers except the final two (avg. pool & fc)
        self.feature_provider = nn.Sequential(*list(backbone.children())[:-2])

        # change stride
        conv4_block1 = self.feature_provider[-1][0]
        conv4_block1.conv1.stride = (1, 1)  # changing the stride to 1x1
        conv4_block1.conv2.stride = (1, 1)  # changing the stride to 1x1
        conv4_block1.downsample[0].stride = (1, 1)  # changing the stride to 1x1

    def forward(self, x):
        # provides a feature map in a forward pass
        x = self.feature_provider(x)
        return x  # [512,19,19]


class resnet18_SSD300(nn.Module):
    def __init__(self, pretrained, cfg):
        super().__init__()

        self.feature_provider = ResNet(
            pretrained
        )  # initialising our feature provider backbone
        self.label_num = cfg.MODEL.NUM_CLASSES  # number of classes

        features_list = ["19x19", "10x10", "5x5", "3x3", "1x1"]
        feature_channel_dict = {
            "19x19": 512,
            "10x10": 512,
            "5x5": 256,
            "3x3": 256,
            "1x1": 256,
        }

        # proposed priors per feature in a feature map
        num_prior_box_dict = {"19x19": 6, "10x10": 6, "5x5": 6, "3x3": 4, "1x1": 4}

        intermediate_channel_dict = {"10x10": 256, "5x5": 128, "3x3": 128, "1x1": 128}

        # intermediate channels for the additional layers
        self._make_additional_features_maps(
            features_list, feature_channel_dict, intermediate_channel_dict
        )

        self.loc = []
        self.conf = []

        # Generating localization and classification heads
        for feature_map_name in features_list:
            priors_boxes = num_prior_box_dict[feature_map_name]
            output_channel_from_feature_map = feature_channel_dict[feature_map_name]
            self.loc.append(
                nn.Conv2d(
                    output_channel_from_feature_map,
                    priors_boxes * 4,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.conf.append(
                nn.Conv2d(
                    output_channel_from_feature_map,
                    priors_boxes * self.label_num,
                    kernel_size=3,
                    padding=1,
                )
            )

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _make_additional_features_maps(
        self, features_list, feature_channel_dict, intermediate_channel_dict
    ):

        input_list = features_list[:-1]
        output_list = features_list[1:]

        self.additional_blocks = []

        for i, (prev_feature_name, current_feature_name) in enumerate(
            zip(input_list, output_list)
        ):
            if i < 2:  # for the first 3 additional features maps (19x19 , 10x10, 5x5)
                padding = 1
                stride = 2
            else:  # for 3x3 and 1x1
                padding = 0
                stride = 1

            layer = nn.Sequential(
                nn.Conv2d(
                    feature_channel_dict[prev_feature_name],
                    intermediate_channel_dict[current_feature_name],
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    intermediate_channel_dict[current_feature_name]
                ),  # an additional implementation by NVIDIA
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    intermediate_channel_dict[current_feature_name],
                    feature_channel_dict[current_feature_name],
                    kernel_size=3,
                    padding=padding,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    feature_channel_dict[current_feature_name]
                ),  # an additional implementation by NVIDIA
                nn.ReLU(inplace=True),
            )
            # adding the new feature map generator block to our arsenal
            self.additional_blocks.append(layer)

        # converting into nn modules so that they can be added to pytorchs
        # computational graph and backprop can be performed
        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    # Xavier initializing the weights
    def _init_weights(self):
        # making a list of all blocks in our SSD300 models
        # note that the backbone already has weights initialised so we are
        # initialising only the newly created layer's weights
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.feature_provider(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        return detection_feed


@registry.BACKBONES.register("resnet18_SSD300")
def resnet18_SSD(cfg, pretrained=True):
    model = resnet18_SSD300(pretrained, cfg)
    return model
