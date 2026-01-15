import os
import argparse
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("❌ Lỗi: Chưa cài thư viện segmentation_models_pytorch. Chạy: pip install segmentation-models-pytorch")
    smp = None

try:
    from efficientvit.seg_model_zoo import create_efficientvit_seg_model
except ImportError:
    create_efficientvit_seg_model = None

def build_model(arch, encoder, num_classes, weight_path, device):
    model = None
    print(f"🏗️ Đang khởi tạo model: {arch} (Encoder: {encoder})...")

    if arch.lower() == "efficientvit-seg" or "efficientvit" in arch.lower():
        if create_efficientvit_seg_model is None:
            raise ImportError("Chưa cài thư viện efficientvit để chạy model này.")
        
        model_zoo_name = encoder if encoder else "efficientvit-seg-l1-ade20k"
        model = create_efficientvit_seg_model(
            name=model_zoo_name,
            pretrained=False,
            num_classes=num_classes
        )
    
    elif smp is not None and hasattr(smp, arch):
        model_fn = getattr(smp, arch)
        model = model_fn(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Kiến trúc model '{arch}' không được hỗ trợ hoặc chưa import thư viện.")

    model = model.to(device)
    if os.path.exists(weight_path):
        print(f"📥 Loading weights từ: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[6:]
            new_state_dict[name] = v
        
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("✅ Load weights thành công (Strict mode).")
        except Exception as e:
            print(f"⚠️ Warning: Load strict thất bại, thử strict=False. Lỗi: {e}")
            model.load_state_dict(new_state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Không tìm thấy file weight: {weight_path}")

    return model

def overlay_mask(image, mask, alpha=0.5):
    vis_image = image.copy()
    if mask.max() == 0: return vis_image

    color_mask = np.zeros_like(image)
    
    if mask.max() == 1:
        color_mask[mask == 1] = [0, 255, 0] # Green
        mask_bool = mask == 1
    else:
        np.random.seed(42)
        unique_classes = np.unique(mask)
        mask_bool = mask > 0
        for cls in unique_classes:
            if cls == 0: continue
            color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
            color_mask[mask == cls] = color
            
    vis_image[mask_bool] = cv2.addWeighted(image[mask_bool], 1-alpha, color_mask[mask_bool], alpha, 0)
    return vis_image

def main():
    parser = argparse.ArgumentParser(description="Predict Segmentation (Standalone)")
    parser.add_argument("--input", type=str, required=True, help="Path ảnh hoặc thư mục ảnh")
    parser.add_argument("--weight", type=str, required=True, help="Path file .pt")
    parser.add_argument("--output", type=str, default="predictions", help="Thư mục output")
    parser.add_argument("--arch", type=str, default="DeepLabV3Plus", help="Kiến trúc (Unet, DeepLabV3Plus, EfficientViT-Seg...)")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder (resnet34, mobileone_s0, efficientvit-seg-l1...)")
    parser.add_argument("--classes", type=int, default=2, help="Số class (1=Binary)")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    
    model = build_model(args.arch, args.encoder, args.classes, args.weight, device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.isdir(args.input):
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = [p for p in glob.glob(os.path.join(args.input, "*")) if os.path.splitext(p)[1].lower() in valid_exts]
    else:
        image_paths = [args.input]

    if not image_paths:
        print("❌ Không tìm thấy ảnh nào.")
        return

    print(f"🚀 Bắt đầu dự đoán {len(image_paths)} ảnh...")

    for img_path in image_paths:
        try:
            pil_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size
            img_tensor = transform(pil_img).unsqueeze(0).to(device) # (1, C, H, W)

            with torch.no_grad():
                output = model(img_tensor)
                
                if output.shape[-2:] != img_tensor.shape[-2:]:
                    output = F.interpolate(output, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
                
                if args.classes == 1:
                    pred_mask = (torch.sigmoid(output) > 0.5).long().squeeze().cpu().numpy()
                else:
                    pred_mask = torch.argmax(output, dim=1).long().squeeze().cpu().numpy()

            vis_img = np.array(pil_img)
            result_img = overlay_mask(vis_img, pred_mask)
            
            filename = os.path.basename(img_path)
            save_path = os.path.join(args.output, f"pred_{filename}")
            Image.fromarray(result_img).save(save_path)
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý ảnh {img_path}: {e}")

    print(f"✅ Hoàn tất. Kết quả lưu tại: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()




# python demo_efficientvit_seg_model.py \
#     --model "efficientvit-seg-l1-ade20k" \
#     --weight_path "/home/weed/Pictures/road_segmentation/model/model_All_CrossEntropyLoss_EfficientViT-Seg_l1.pt" \
#     --image_path "/home/weed/Pictures/road_segmentation/Satellite_Datasets/Test/images/Test/image (72).png" \
#     --output_path "/home/weed/Pictures/road_segmentation/prediction.png"

