import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

def unnormalize(tensor):
    """Khôi phục tensor ảnh đã normalize về lại dạng ảnh RGB để xem"""
    # Lấy device hiện tại của tensor để tránh lỗi khác device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1) # Giới hạn giá trị màu trong khoảng [0, 1]
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return img

def visualize_cast_pooling(model, image_tensor, segment_tensor, device):
    
    images = image_tensor.unsqueeze(0).to(device, non_blocking=True)
    segments = segment_tensor.unsqueeze(0).to(device, non_blocking=True)
    
    # Khôi phục ảnh gốc để làm nền mờ (overlay)
    raw_image = unnormalize(images[0])

    # Chạy model với AMP (Automatic Mixed Precision)
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            *_, intermediates = model(images, segments, return_intermediates=True) 

    # Tính toán Assignment Matrix (S)
    S1 = F.softmax(intermediates['logit1'], dim=-1) # (1, 196, 64)
    S2 = F.softmax(intermediates['logit2'], dim=-1) # (1, 64, 32)
    S3 = F.softmax(intermediates['logit3'], dim=-1) # (1, 32, 16)

    # Do dùng autocast(), các biến S1, S2, S3 có thể đang là Float16.
    # ép về Float32 trước khi nhân ma trận để đảm bảo không bị lỗi sai số.
    S1 = S1.float()
    S2 = S2.float()
    S3 = S3.float()

    # Nhân ma trận (Back-projection)
    Map_L1 = S1 
    Map_L2 = torch.bmm(S1, S2) 
    Map_L3 = torch.bmm(Map_L2, S3) 

    # Hàm hiển thị (Plotting)
    def show_top_clusters(map_tensor, title, num_show=5):
        bs, num_patches, num_clusters = map_tensor.shape
        grid_h = grid_w = int(np.sqrt(num_patches)) # 14x14 = 196 patches
        
        # Tìm các clusters có tổng cường độ cao nhất
        cluster_strength = map_tensor.sum(dim=1).squeeze()
        _, top_indices = torch.topk(cluster_strength, k=min(num_show, num_clusters))
        
        plt.figure(figsize=(18, 4))
        plt.suptitle(f"{title}", fontsize=16, fontweight='bold')
        
        plt.subplot(1, num_show + 1, 1)
        plt.imshow(raw_image)
        plt.axis('off')
        plt.title("Original Image")
        
        # In các segments đè lên ảnh
        for i, cluster_idx in enumerate(top_indices):
            # Lấy bản đồ 14x14
            heatmap = map_tensor[0, :, cluster_idx].reshape(grid_h, grid_w).cpu().numpy()
            
            # Phóng to lên 224x224 cho khớp ảnh thật
            heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
            
            # Chuẩn hóa độ sáng 
            heat_min, heat_max = heatmap_resized.min(), heatmap_resized.max()
            heatmap_norm = (heatmap_resized - heat_min) / (heat_max - heat_min + 1e-8)
            
            plt.subplot(1, num_show + 1, i + 2)
            plt.imshow(raw_image)
            plt.imshow(heatmap_norm, cmap='jet', alpha=0.55) # Lớp phủ nhiệt
            plt.axis('off')
            plt.title(f"Segment ID: {cluster_idx.item()}")
            
        plt.tight_layout()
        plt.show()

    # vẽ 3 cấp độ
    print("Đang render bản đồ Pooling...")
    show_top_clusters(Map_L1, "Level 1: Fine-Grained Features (Species/Superpixels)")
    show_top_clusters(Map_L2, "Level 2: Mid-Level Grouping (Family/Parts)")
    show_top_clusters(Map_L3, "Level 3: Global Shape (Order/Background)")