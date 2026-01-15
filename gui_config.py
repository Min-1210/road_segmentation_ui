import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_ROOT = BASE_DIR
MODEL_DB = {
    "DeepLabV3Plus": {
        "mobileone_s0": {
            "DeepGlobal": "weight_data/DeepGlobal/DeepLabV3Plus/model/model_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s0.pt",
            "Massachusetts": "weight_data/Massachusetts/DeepLabV3Plus/model/model_Massachusetts_CrossEntropyLoss_DeepLabV3Plus_mobileone_s0.pt",
            "TGRS_Road": "weight_data/TGRS_Road/DeepLabV3Plus/model/model_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "weight_data/DeepGlobal/DeepLabV3Plus/model/model_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s1.pt",
            "Massachusetts": "weight_data/Massachusetts/DeepLabV3Plus/model/model_Massachusetts_CrossEntropyLoss_DeepLabV3Plus_mobileone_s1.pt",
            "TGRS_Road": "weight_data/TGRS_Road/DeepLabV3Plus/model/model_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "weight_data/DeepGlobal/DeepLabV3Plus/model/model_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s2.pt",
            "Massachusetts": "weight_data/Massachusetts/DeepLabV3Plus/model/model_Massachusetts_CrossEntropyLoss_DeepLabV3Plus_mobileone_s2.pt",
            "TGRS_Road": "weight_data/TGRS_Road/DeepLabV3Plus/model/model_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "weight_data/DeepGlobal/DeepLabV3Plus/model/model_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s3.pt",
            "Massachusetts": "weight_data/Massachusetts/DeepLabV3Plus/model/model_Massachusetts_CrossEntropyLoss_DeepLabV3Plus_mobileone_s3.pt",
            "TGRS_Road": "weight_data/TGRS_Road/DeepLabV3Plus/model/model_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s3.pt"
        }     
    },
    "EfficientViT-Seg": {
        "efficientvit-seg-l1-ade20k": {
            "DeepGlobal": "weight_data/DeepGlobal/EfficientViT_Seg/model/model_DeepGlobal_CrossEntropyLoss_EfficientViT-Seg_l1-ade20k.pt",
            "Massachusetts": "weight_data/Massachusetts/EfficientViT_Seg/model/model_Massachusetts_CrossEntropyLoss_EfficientViT-Seg_l1-ade20k.pt",
            "TGRS_Road": "weight_data/TGRS_Road/EfficientViT_Seg/model/model_TGRS_Road_CrossEntropyLoss_EfficientViT-Seg_l1-ade20k.pt"
        },
        "efficientvit-seg-l2-ade20k": {
            "DeepGlobal": "weight_data/DeepGlobal/EfficientViT_Seg/model/model_DeepGlobal_CrossEntropyLoss_EfficientViT-Seg_l2-ade20k.pt",
            "Massachusetts": "weight_data/Massachusetts/EfficientViT_Seg/model/model_Massachusetts_CrossEntropyLoss_EfficientViT-Seg_l2-ade20k.pt",
            "TGRS_Road": "weight_data/TGRS_Road/EfficientViT_Seg/model/model_TGRS_Road_CrossEntropyLoss_EfficientViT-Seg_l2-ade20k.pt"
        },
        "efficientvit-seg-l1-cityscapes": {
            "DeepGlobal": "weight_data/DeepGlobal/EfficientViT_Seg/model/model_DeepGlobal_CrossEntropyLoss_EfficientViT-Seg_l1-cityscapes.pt",
            "Massachusetts": "weight_data/Massachusetts/EfficientViT_Seg/model/model_Massachusetts_CrossEntropyLoss_EfficientViT-Seg_l1-cityscapes.pt",
            "TGRS_Road": "weight_data/TGRS_Road/EfficientViT_Seg/model/model_TGRS_Road_CrossEntropyLoss_EfficientViT-Seg_l1-cityscapes.pt"
        },
        "efficientvit-seg-l2-cityscapes": {
            "DeepGlobal": "weight_data/DeepGlobal/EfficientViT_Seg/model/model_DeepGlobal_CrossEntropyLoss_EfficientViT-Seg_l2-cityscapes.pt",
            "Massachusetts": "weight_data/Massachusetts/EfficientViT_Seg/model/model_Massachusetts_CrossEntropyLoss_EfficientViT-Seg_l2-cityscapes.pt",
            "TGRS_Road": "weight_data/TGRS_Road/EfficientViT_Seg/model/model_TGRS_Road_CrossEntropyLoss_EfficientViT-Seg_l2-cityscapes.pt"
        }
    },
    "FPN": {
        "mobileone_s0": {
            "DeepGlobal": "weight_data/DeepGlobal/FPN/model/model_DeepGlobal_CrossEntropyLoss_FPN_mobileone_s0.pt",
            "Massachusetts": "weight_data/Massachusetts/FPN/model/model_Massachusetts_CrossEntropyLoss_FPN_mobileone_s0.pt",
            "TGRS_Road": "weight_data/TGRS_Road/FPN/model/model_TGRS_Road_CrossEntropyLoss_FPN_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "weight_data/DeepGlobal/FPN/model/model_DeepGlobal_CrossEntropyLoss_FPN_mobileone_s1.pt",
            "Massachusetts": "weight_data/Massachusetts/FPN/model/model_Massachusetts_CrossEntropyLoss_FPN_mobileone_s1.pt",
            "TGRS_Road": "weight_data/TGRS_Road/FPN/model/model_TGRS_Road_CrossEntropyLoss_FPN_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "weight_data/DeepGlobal/FPN/model/model_DeepGlobal_CrossEntropyLoss_FPN_mobileone_s2.pt",
            "Massachusetts": "weight_data/Massachusetts/FPN/model/model_Massachusetts_CrossEntropyLoss_FPN_mobileone_s2.pt",
            "TGRS_Road": "weight_data/TGRS_Road/FPN/model/model_TGRS_Road_CrossEntropyLoss_FPN_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "weight_data/DeepGlobal/FPN/model/model_DeepGlobal_CrossEntropyLoss_FPN_mobileone_s3.pt",
            "Massachusetts": "weight_data/Massachusetts/FPN/model/model_Massachusetts_CrossEntropyLoss_FPN_mobileone_s3.pt",
            "TGRS_Road": "weight_data/TGRS_Road/FPN/model/model_TGRS_Road_CrossEntropyLoss_FPN_mobileone_s3.pt"
        }     
    },
    "MAnet": {
        "mobileone_s0": {
            "DeepGlobal": "weight_data/DeepGlobal/MAnet/model/model_DeepGlobal_CrossEntropyLoss_MAnet_mobileone_s0.pt",
            "Massachusetts": "weight_data/Massachusetts/MAnet/model/model_Massachusetts_CrossEntropyLoss_MAnet_mobileone_s0.pt",
            "TGRS_Road": "weight_data/TGRS_Road/MAnet/model/model_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "weight_data/DeepGlobal/MAnet/model/model_DeepGlobal_CrossEntropyLoss_MAnet_mobileone_s1.pt",
            "Massachusetts": "weight_data/Massachusetts/MAnet/model/model_Massachusetts_CrossEntropyLoss_MAnet_mobileone_s1.pt",
            "TGRS_Road": "weight_data/TGRS_Road/MAnet/model/model_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "weight_data/DeepGlobal/MAnet/model/model_DeepGlobal_CrossEntropyLoss_MAnet_mobileone_s2.pt",
            "Massachusetts": "weight_data/Massachusetts/MAnet/model/model_Massachusetts_CrossEntropyLoss_MAnet_mobileone_s2.pt",
            "TGRS_Road": "weight_data/TGRS_Road/MAnet/model/model_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "weight_data/DeepGlobal/MAnet/model/model_DeepGlobal_CrossEntropyLoss_MAnet_mobileone_s3.pt",
            "Massachusetts": "weight_data/Massachusetts/MAnet/model/model_Massachusetts_CrossEntropyLoss_MAnet_mobileone_s3.pt",
            "TGRS_Road": "weight_data/TGRS_Road/MAnet/model/model_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s3.pt"
        }     
    },
    "PAN": {
        "mobileone_s0": {
            "DeepGlobal": "weight_data/DeepGlobal/PAN/model/model_DeepGlobal_CrossEntropyLoss_PAN_mobileone_s0.pt",
            "Massachusetts": "weight_data/Massachusetts/PAN/model/model_Massachusetts_CrossEntropyLoss_PAN_mobileone_s0.pt",
            "TGRS_Road": "weight_data/TGRS_Road/PAN/model/model_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "weight_data/DeepGlobal/PAN/model/model_DeepGlobal_CrossEntropyLoss_PAN_mobileone_s1.pt",
            "Massachusetts": "weight_data/Massachusetts/PAN/model/model_Massachusetts_CrossEntropyLoss_PAN_mobileone_s1.pt",
            "TGRS_Road": "weight_data/TGRS_Road/PAN/model/model_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "weight_data/DeepGlobal/PAN/model/model_DeepGlobal_CrossEntropyLoss_PAN_mobileone_s2.pt",
            "Massachusetts": "weight_data/Massachusetts/PAN/model/model_Massachusetts_CrossEntropyLoss_PAN_mobileone_s2.pt",
            "TGRS_Road": "weight_data/TGRS_Road/PAN/model/model_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "weight_data/DeepGlobal/PAN/model/model_DeepGlobal_CrossEntropyLoss_PAN_mobileone_s3.pt",
            "Massachusetts": "weight_data/Massachusetts/PAN/model/model_Massachusetts_CrossEntropyLoss_PAN_mobileone_s3.pt",
            "TGRS_Road": "weight_data/TGRS_Road/PAN/model/model_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s3.pt"
        }     
    },
    "PSPNet": {
        "mobileone_s0": {
            "DeepGlobal": "weight_data/DeepGlobal/PSPNet/model/model_DeepGlobal_CrossEntropyLoss_PSPNet_mobileone_s0.pt",
            "Massachusetts": "weight_data/Massachusetts/PSPNet/model/model_Massachusetts_CrossEntropyLoss_PSPNet_mobileone_s0.pt",
            "TGRS_Road": "weight_data/TGRS_Road/PSPNet/model/model_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "weight_data/DeepGlobal/PSPNet/model/model_DeepGlobal_CrossEntropyLoss_PSPNet_mobileone_s1.pt",
            "Massachusetts": "weight_data/Massachusetts/PSPNet/model/model_Massachusetts_CrossEntropyLoss_PSPNet_mobileone_s1.pt",
            "TGRS_Road": "weight_data/TGRS_Road/PSPNet/model/model_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "weight_data/DeepGlobal/PSPNet/model/model_DeepGlobal_CrossEntropyLoss_PSPNet_mobileone_s2.pt",
            "Massachusetts": "weight_data/Massachusetts/PSPNet/model/model_Massachusetts_CrossEntropyLoss_PSPNet_mobileone_s2.pt",
            "TGRS_Road": "weight_data/TGRS_Road/PSPNet/model/model_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "weight_data/DeepGlobal/PSPNet/model/model_DeepGlobal_CrossEntropyLoss_PSPNet_mobileone_s3.pt",
            "Massachusetts": "weight_data/Massachusetts/PSPNet/model/model_Massachusetts_CrossEntropyLoss_PSPNet_mobileone_s3.pt",
            "TGRS_Road": "weight_data/TGRS_Road/PSPNet/model/model_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s3.pt"
        }     
    },
    "UPerNet": {
        "mobileone_s0": {
            "DeepGlobal": "weight_data/DeepGlobal/UPerNet/model/model_DeepGlobal_CrossEntropyLoss_UPerNet_mobileone_s0.pt",
            "Massachusetts": "weight_data/Massachusetts/UPerNet/model/model_Massachusetts_CrossEntropyLoss_UPerNet_mobileone_s0.pt",
            "TGRS_Road": "weight_data/TGRS_Road/UPerNet/model/model_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "weight_data/DeepGlobal/UPerNet/model/model_DeepGlobal_CrossEntropyLoss_UPerNet_mobileone_s1.pt",
            "Massachusetts": "weight_data/Massachusetts/UPerNet/model/model_Massachusetts_CrossEntropyLoss_UPerNet_mobileone_s1.pt",
            "TGRS_Road": "weight_data/TGRS_Road/UPerNet/model/model_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "weight_data/DeepGlobal/UPerNet/model/model_DeepGlobal_CrossEntropyLoss_UPerNet_mobileone_s2.pt",
            "Massachusetts": "weight_data/Massachusetts/UPerNet/model/model_Massachusetts_CrossEntropyLoss_UPerNet_mobileone_s2.pt",
            "TGRS_Road": "weight_data/TGRS_Road/UPerNet/model/model_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "weight_data/DeepGlobal/UPerNet/model/model_DeepGlobal_CrossEntropyLoss_UPerNet_mobileone_s3.pt",
            "Massachusetts": "weight_data/Massachusetts/UPerNet/model/model_Massachusetts_CrossEntropyLoss_UPerNet_mobileone_s3.pt",
            "TGRS_Road": "weight_data/TGRS_Road/UPerNet/model/model_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s3.pt"
        }     
    },
}
def get_available_datasets(arch, encoder):
    if arch in MODEL_DB and encoder in MODEL_DB[arch]:
        return list(MODEL_DB[arch][encoder].keys())
    return []

def get_weight_path(arch, encoder, dataset):
    if arch in MODEL_DB and encoder in MODEL_DB[arch] and dataset in MODEL_DB[arch][encoder]:
        filename = MODEL_DB[arch][encoder][dataset]
        filename = filename.replace("/", os.sep)
        
        return os.path.join(WEIGHTS_ROOT, filename)
    return None