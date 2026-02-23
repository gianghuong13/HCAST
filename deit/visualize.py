import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

def unnormalize(tensor):
    """Khôi phục tensor ảnh đã normalize"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy()

def generate_random_colors(num_colors):
    """Tạo bảng màu ngẫu nhiên (RGB) đặc trưng cho từng cluster"""
    np.random.seed(42)
    colors = np.random.rand(num_colors, 3)
    return colors

def get_boundaries_numpy(label_img):
    # pad ảnh để xử lý các pixel ở viền
    padded = np.pad(label_img, pad_width=1, mode='edge')
    # So sánh từng pixel với 4 hàng xóm xung quanh, nếu khác -> boundary
    boundaries = (label_img != padded[:-2, 1:-1]) | \
                 (label_img != padded[2:, 1:-1]) | \
                 (label_img != padded[1:-1, :-2]) | \
                 (label_img != padded[1:-1, 2:])
    return boundaries

def create_segmentation_map(superpixel_mask, patch_cluster_ids, colors):
    """
    Tạo ảnh segmentation bằng cách ánh xạ ID của từng patch vào superpixel mask.
    - superpixel_mask: (224, 224) chứa ID của superpixel (0 -> 195)
    - patch_cluster_ids: (196,) chứa ID cụm mà superpixel đó thuộc về
    """
    H, W = superpixel_mask.shape
    colored_map = np.zeros((H, W, 3), dtype=np.float32)
    cluster_map_2d = np.zeros((H, W), dtype=np.int32)
    
    # Số lượng patch thực tế (thường là 196)
    num_patches = patch_cluster_ids.shape[0]
    
    for patch_id in range(num_patches):
        # Tìm tất cả các pixel thuộc về superpixel này
        pixel_indices = (superpixel_mask == patch_id)
        
        # Superpixel này thuộc về cluster nào?
        cluster_id = patch_cluster_ids[patch_id]
        
        # Tô màu cho vùng đó
        colored_map[pixel_indices] = colors[cluster_id]
        
        # Lưu lại bản đồ ID để vẽ viền
        cluster_map_2d[pixel_indices] = cluster_id
        
    # Vẽ đường viền trắng 
    boundaries = get_boundaries_numpy(cluster_map_2d)
    colored_map[boundaries] = [1.0, 1.0, 1.0] # Màu trắng
    
    return colored_map

def visualize_cast_pooling(model, image_tensor, segment_tensor, device, img_index=0):
    images = image_tensor.unsqueeze(0).to(device, non_blocking=True)
    segments = segment_tensor.unsqueeze(0).to(device, non_blocking=True)
    
    raw_image = unnormalize(images[0])
    
    # Lấy mask của superpixel (224x224), giá trị từ 0 đến 195
    superpixel_mask = segments[0].cpu().numpy()

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            *_, intermediates = model(images, segments, return_intermediates=True) 

    # 1. Tính Assignment Matrix (Mềm)
    S1 = F.softmax(intermediates['logit1'].float(), dim=-1) # (1, 196, 64)
    S2 = F.softmax(intermediates['logit2'].float(), dim=-1) # (1, 64, 32)
    S3 = F.softmax(intermediates['logit3'].float(), dim=-1) # (1, 32, 16)
    
    # 2. Nhân ma trận để lấy đường đi từ Patch -> các cấp
    Map_L1 = S1                          # Patch -> L1 (1, 196, 64)
    Map_L2 = torch.bmm(S1, S2)           # Patch -> L2 (1, 196, 32)
    Map_L3 = torch.bmm(Map_L2, S3)       # Patch -> L3 (1, 196, 16)
    
    # 3. Chuyển sang "Hard Assignment" bằng argmax 
    # Output shape: (196,) - Mỗi giá trị là ID của cluster mà patch đó thuộc về
    labels_L1 = torch.argmax(Map_L1[0], dim=-1).cpu().numpy()
    labels_L2 = torch.argmax(Map_L2[0], dim=-1).cpu().numpy()
    labels_L3 = torch.argmax(Map_L3[0], dim=-1).cpu().numpy()
    
    # 4. Tô màu
    # Tạo tối đa 64 màu cho L1, 32 màu cho L2, 16 màu cho L3
    colors_L1 = generate_random_colors(64)
    colors_L2 = generate_random_colors(32)
    colors_L3 = generate_random_colors(16)
    
    seg_L1 = create_segmentation_map(superpixel_mask, labels_L1, colors_L1)
    seg_L2 = create_segmentation_map(superpixel_mask, labels_L2, colors_L2)
    seg_L3 = create_segmentation_map(superpixel_mask, labels_L3, colors_L3)

    # 5. Vẽ hình giống Figure 6 và Figure 10 trong paper
    plt.figure(figsize=(16, 4))
    
    # Cột 1: Ảnh gốc
    plt.subplot(1, 4, 1)
    plt.imshow(raw_image)
    plt.axis('off')
    plt.title("Original Image")
    
    # Cột 2: Cấp 1 (Fine-grained)
    plt.subplot(1, 4, 2)
    plt.imshow(seg_L1)
    plt.axis('off')
    plt.title("Level 1 (Fine Segments)")
    
    # Cột 3: Cấp 2
    plt.subplot(1, 4, 3)
    plt.imshow(seg_L2)
    plt.axis('off')
    plt.title("Level 2 (Mid-Level Groups)")
    
    # Cột 4: Cấp 3 (Coarse)
    plt.subplot(1, 4, 4)
    plt.imshow(seg_L3)
    plt.axis('off')
    plt.title("Level 3 (Coarse Segments)")
    
    plt.tight_layout()
    plt.show()
    # Nếu chạy hàng loạt, comment dòng plt.show() và dùng 2 dòng dưới:
    # plt.savefig(f"vis_hierarchy_{img_index}.png", bbox_inches='tight', dpi=150)
    # plt.close()