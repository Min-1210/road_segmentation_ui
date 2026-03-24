import os
import argparse
import cv2
import glob
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
    # [Giữ nguyên code hàm build_model của bạn]
    model = None
    print(f"🏗️ Đang khởi tạo model: {arch} (Encoder: {encoder})...")

    if arch.lower() == "efficientvit-seg" or "efficientvit" in arch.lower():
        if create_efficientvit_seg_model is None:
            raise ImportError("Chưa cài thư viện efficientvit để chạy model này.")
        
        model_zoo_name = encoder if encoder else "efficientvit-seg-l1-ade20k"
        model = create_efficientvit_seg_model(name=model_zoo_name, pretrained=False, num_classes=num_classes)
    elif smp is not None and hasattr(smp, arch):
        model_fn = getattr(smp, arch)
        model = model_fn(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=num_classes, activation=None)
    else:
        raise ValueError(f"Kiến trúc model '{arch}' không được hỗ trợ.")

    model = model.to(device)
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        new_state_dict = {k[6:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except Exception:
            model.load_state_dict(new_state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Không tìm thấy file weight: {weight_path}")
    return model

def overlay_mask(image, mask, alpha=0.5):
    # [Giữ nguyên code hàm overlay_mask của bạn]
    vis_image = image.copy()
    if mask.max() == 0: return vis_image
    color_mask = np.zeros_like(image)
    if mask.max() == 1:
        color_mask[mask == 1] = [0, 255, 0]
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

def process_single_video(input_path, output_path, model, transform, device, args):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎬 Xử lý video: {os.path.basename(input_path)} ({total_frames} frames)...")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                if output.shape[-2:] != img_tensor.shape[-2:]:
                    output = F.interpolate(output, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
                
                if args.classes == 1:
                    pred_mask = (torch.sigmoid(output) > 0.5).long().squeeze().cpu().numpy()
                else:
                    pred_mask = torch.argmax(output, dim=1).long().squeeze().cpu().numpy()

            vis_img = np.array(pil_img)
            result_img_rgb = overlay_mask(vis_img, pred_mask)
            result_img_bgr = cv2.cvtColor(result_img_rgb, cv2.COLOR_RGB2BGR)
            out.write(result_img_bgr)
            
        except Exception as e:
            print(f"❌ Lỗi frame {frame_count} video {os.path.basename(input_path)}: {e}")

    cap.release()
    out.release()
    print(f"✅ Đã lưu: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict Segmentation on Video or Folder")
    parser.add_argument("--input", type=str, required=True, help="Path tới video hoặc THƯ MỤC chứa video")
    parser.add_argument("--weight", type=str, required=True, help="Path file .pt")
    parser.add_argument("--output", type=str, required=True, help="Path lưu video đầu ra hoặc THƯ MỤC đầu ra")
    parser.add_argument("--arch", type=str, default="DeepLabV3Plus")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--classes", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, args.encoder, args.classes, args.weight, device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # KIỂM TRA ĐẦU VÀO LÀ FILE HAY FOLDER
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True) # Tạo folder output nếu chưa có
        valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(valid_exts)]
        
        if not video_files:
            print(f"⚠️ Không tìm thấy video nào trong thư mục: {args.input}")
            return
            
        print(f"🚀 Tìm thấy {len(video_files)} video. Bắt đầu xử lý hàng loạt...")
        for video_path in video_files:
            filename = os.path.basename(video_path)
            output_path = os.path.join(args.output, f"pred_{filename}")
            process_single_video(video_path, output_path, model, transform, device, args)
            
        print(f"🎉 Hoàn tất toàn bộ folder! Kết quả lưu tại: {os.path.abspath(args.output)}")
        
    elif os.path.isfile(args.input):
        # Chạy file đơn lẻ
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        process_single_video(args.input, args.output, model, transform, device, args)
    else:
        print(f"❌ Đường dẫn input không tồn tại: {args.input}")

if __name__ == "__main__":
    main()
    
    
    # python3 inference_video.py     
    # --arch "efficientvit-seg"     
    # --encoder "efficientvit-seg-l1-cityscapes"     
    # --weight "/home/weed/Pictures/road_segmentation_v1/Result/Version2/DeepGlobal/EfficientViT_Seg/model/model_DeepGlobal_CrossEntropyLoss_EfficientViT-Seg_l1-cityscapes.pt"     
    # --input "/home/weed/Pictures/road_segmentation_v1/Satellite_Datasets/video/video_1.mp4"     
    # --output "/home/weed/Pictures/road_segmentation_v1/Satellite_Datasets/predicted_video_1.mp4"
