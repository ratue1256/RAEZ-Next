# hand_bone_tracker/scripts/generate_synthetic_hands.py
import h5py
import numpy as np
import cv2
import random
from pathlib import Path
from tqdm import tqdm

# Target image size
TARGET_SIZE = (256, 256)

# Camera intrinsic matrix (FreiHAND equivalent for 256x256)
FOCAL_LENGTH = 500.0
CX = 128.0
CY = 128.0
K = np.array([
    [FOCAL_LENGTH,          0.0,   CX],
    [         0.0, FOCAL_LENGTH,   CY],
    [         0.0,          0.0,  1.0]
], dtype=np.float32)

# Helper function to get rotation matrix around an axis
def rotation_matrix(axis, angle):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], dtype=np.float32)

def get_rotation_x(angle):
    return rotation_matrix([1, 0, 0], angle)

def get_rotation_y(angle):
    return rotation_matrix([0, 1, 0], angle)

def get_rotation_z(angle):
    return rotation_matrix([0, 0, 1], angle)

# Forward kinematics for a single hand
def generate_hand_pose():
    # Base bone lengths (meters)
    # Wrist to MCP/CMC lengths
    L_wrist_to_mcp = {
        'thumb': 0.07,
        'index': 0.09,
        'middle': 0.09,
        'ring': 0.085,
        'pinky': 0.08
    }
    
    # Finger joint segment lengths (MCP->PIP, PIP->DIP, DIP->TIP)
    L_finger_segments = {
        'thumb':  [0.035, 0.030, 0.025],
        'index':  [0.040, 0.028, 0.022],
        'middle': [0.045, 0.032, 0.025],
        'ring':   [0.040, 0.028, 0.022],
        'pinky':  [0.032, 0.022, 0.018]
    }
    
    # Base spread/tilt rotations relative to wrist (in radians)
    # Thumb: spread out and tilted forward
    # Index/Middle/Ring/Pinky: spread out in the palm plane
    base_spread = {
        'thumb':  (-np.pi / 4.5, np.pi / 6.0), # (spread, tilt)
        'index':  (-np.pi / 15.0, 0.0),
        'middle': (0.0, 0.0),
        'ring':   (np.pi / 15.0, 0.0),
        'pinky':  (np.pi / 7.5, 0.0)
    }
    
    # Randomize global hand size factor
    size_factor = random.uniform(0.85, 1.15)
    
    # 21 Keypoints list
    joints_3d = np.zeros((21, 3), dtype=np.float32)
    # Joint 0 is Wrist (origin)
    joints_3d[0] = [0.0, 0.0, 0.0]
    
    # Generate joint rotations (biomechanically correlated)
    # If MCP bends, PIP and DIP bend as well
    flexion_master = random.uniform(-0.1, 1.6) # master bending factor
    
    finger_keys = ['thumb', 'index', 'middle', 'ring', 'pinky']
    finger_offsets = {
        'thumb':  1,
        'index':  5,
        'middle': 9,
        'ring':   13,
        'pinky':  17
    }
    
    for f in finger_keys:
        idx = finger_offsets[f]
        
        # Local lengths scaled
        l_wrist_mcp = L_wrist_to_mcp[f] * size_factor
        l_seg = [seg * size_factor for seg in L_finger_segments[f]]
        
        # Base spread/tilt
        spread_ang, tilt_ang = base_spread[f]
        # Add small random spread variation
        spread_ang += random.uniform(-0.05, 0.05)
        tilt_ang += random.uniform(-0.05, 0.05)
        
        R_base = get_rotation_z(spread_ang) @ get_rotation_x(tilt_ang)
        
        # Joint 1 (MCP or CMC for thumb)
        joints_3d[idx] = R_base @ np.array([0.0, l_wrist_mcp, 0.0], dtype=np.float32)
        
        # Generate flexion angles for MCP, PIP, DIP
        # Correlated with flexion_master
        if f == 'thumb':
            mcp_flex = max(-0.2, min(0.6, flexion_master * 0.4 + random.uniform(-0.1, 0.1)))
            pip_flex = max(0.0, min(1.0, flexion_master * 0.6 + random.uniform(-0.1, 0.1)))
            dip_flex = max(0.0, min(0.8, flexion_master * 0.5 + random.uniform(-0.1, 0.1)))
        else:
            mcp_flex = max(-0.2, min(1.4, flexion_master * 0.8 + random.uniform(-0.1, 0.1)))
            pip_flex = max(0.0, min(1.6, flexion_master * 1.0 + random.uniform(-0.1, 0.1)))
            dip_flex = max(0.0, min(1.2, flexion_master * 0.8 + random.uniform(-0.1, 0.1)))
            
        # MCP rotation (includes minor adduction/abduction)
        mcp_abd = random.uniform(-0.15, 0.15) if f != 'middle' else 0.0
        R_mcp = R_base @ get_rotation_x(mcp_flex) @ get_rotation_z(mcp_abd)
        
        # Joint 2 (PIP)
        joints_3d[idx+1] = joints_3d[idx] + R_mcp @ np.array([0.0, l_seg[0], 0.0], dtype=np.float32)
        
        # PIP rotation
        R_pip = R_mcp @ get_rotation_x(pip_flex)
        
        # Joint 3 (DIP)
        joints_3d[idx+2] = joints_3d[idx+1] + R_pip @ np.array([0.0, l_seg[1], 0.0], dtype=np.float32)
        
        # DIP rotation
        R_dip = R_pip @ get_rotation_x(dip_flex)
        
        # Joint 4 (TIP)
        joints_3d[idx+3] = joints_3d[idx+2] + R_dip @ np.array([0.0, l_seg[2], 0.0], dtype=np.float32)

    # Apply global hand rotation (random pose relative to camera)
    pitch = random.uniform(-np.pi / 3, np.pi / 3)
    roll = random.uniform(-np.pi / 2, np.pi / 2)
    yaw = random.uniform(-np.pi, np.pi)
    
    R_global = get_rotation_z(roll) @ get_rotation_x(pitch) @ get_rotation_y(yaw)
    joints_3d_rot = (R_global @ joints_3d.T).T
    
    # Translate hand to a random depth Z and offsets X, Y
    # Ensure it stays in view of the camera
    depth = random.uniform(0.40, 0.75) # 40cm to 75cm away
    # Compute maximum offsets based on depth to keep hand in center
    max_offset_x = (depth * 0.15)
    max_offset_y = (depth * 0.15)
    offset_x = random.uniform(-max_offset_x, max_offset_x)
    offset_y = random.uniform(-max_offset_y, max_offset_y)
    
    translation = np.array([offset_x, offset_y, depth], dtype=np.float32)
    joints_3d_final = joints_3d_rot + translation
    
    return joints_3d_final

# Drawing / Rendering functions
def draw_volumetric_bone(img, pA, zA, pB, zB, rA_phys, rB_phys, color, highlight_color):
    """Draws a perspective-correct 3D cylinder bone with lighting highlights."""
    # Compute projected thicknesses at joint A and B
    tA = int(rA_phys * FOCAL_LENGTH / zA)
    tB = int(rB_phys * FOCAL_LENGTH / zB)
    tA = max(1, tA)
    tB = max(1, tB)
    
    dp = pB - pA
    dist = np.linalg.norm(dp)
    if dist < 1e-5:
        cv2.circle(img, (int(pA[0]), int(pA[1])), tA, color, -1, cv2.LINE_AA)
        return
        
    d_hat = dp / dist
    n_hat = np.array([-d_hat[1], d_hat[0]], dtype=np.float32)
    
    # Calculate corners of the perspective trapezoid
    pts = np.array([
        pA + tA * n_hat,
        pA - tA * n_hat,
        pB - tB * n_hat,
        pB + tB * n_hat
    ], dtype=np.int32)
    
    # Draw cylinder base
    cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)
    
    # Draw cylinder highlight (cylindrical shading effect)
    h_tA = max(1, int(tA * 0.35))
    h_tB = max(1, int(tB * 0.35))
    h_offset = n_hat * 0.25 # Offset highlight to simulate a light source from top-left
    
    h_pts = np.array([
        pA + h_offset + h_tA * n_hat,
        pA + h_offset - h_tA * n_hat,
        pB + h_offset - h_tB * n_hat,
        pB + h_offset + h_tB * n_hat
    ], dtype=np.int32)
    
    cv2.fillConvexPoly(img, h_pts, highlight_color, cv2.LINE_AA)

def render_synthetic_hand(joints_3d, bg_img):
    # Project joints to 2D
    uv_proj = (K @ joints_3d.T).T
    uv = uv_proj[:, :2] / uv_proj[:, 2:3]
    
    # Prepare canvas
    img = bg_img.copy()
    
    # Style selector
    style = random.choice(['skin', 'robot_grey', 'robot_gold', 'cyan_neon'])
    
    # Base color schemes (BGR)
    if style == 'skin':
        # Skin tone range (randomized)
        h = random.uniform(10, 20)
        s = random.uniform(50, 130)
        v = random.uniform(180, 240)
        # Convert HSV color to BGR
        skin_color_hsv = np.uint8([[[h, s, v]]])
        base_color = tuple(int(c) for c in cv2.cvtColor(skin_color_hsv, cv2.COLOR_HSV2BGR)[0][0])
        # highlight: slightly lighter/whiter
        highlight_color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, max(10, s-40), min(255, v+30)]]]), cv2.COLOR_HSV2BGR)[0][0])
        joint_color = base_color
    elif style == 'robot_grey':
        base_color = (110, 110, 110)
        highlight_color = (200, 200, 200)
        joint_color = (230, 120, 30) # Orange joints for mechanical look
    elif style == 'robot_gold':
        base_color = (30, 140, 200) # Gold-ish BGR
        highlight_color = (130, 210, 255)
        joint_color = (150, 40, 40) # Dark red joints
    else: # cyan_neon
        base_color = (200, 200, 30) # Darker cyan base
        highlight_color = (255, 255, 100) # Glowing bright cyan
        joint_color = (0, 255, 255) # Yellow joints
        
    # Physical bone radii (in meters)
    radii = {
        'wrist': 0.024,
        'mcp': 0.012,
        'pip': 0.010,
        'dip': 0.009,
        'tip': 0.007
    }
    
    # Mapping joint indices to joint types for radius lookup
    def get_joint_radius(j_idx):
        if j_idx == 0:
            return radii['wrist']
        finger_pos = (j_idx - 1) % 4
        if finger_pos == 0:
            return radii['mcp']
        elif finger_pos == 1:
            return radii['pip']
        elif finger_pos == 2:
            return radii['dip']
        else:
            return radii['tip']

    # Draw palm polygon first
    palm_indices = [0, 1, 5, 9, 13, 17]
    palm_pts = np.array([uv[idx] for idx in palm_indices], dtype=np.int32)
    cv2.fillConvexPoly(img, palm_pts, base_color, cv2.LINE_AA)
    
    # spec highlight for palm
    palm_center = np.mean(palm_pts, axis=0)
    palm_h_pts = np.array([pt + (palm_center - pt)*0.3 for pt in palm_pts], dtype=np.int32)
    cv2.fillConvexPoly(img, palm_h_pts, highlight_color, cv2.LINE_AA)

    # Bone connections list
    BONE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    
    # Draw volumetric bones
    for A, B in BONE_CONNECTIONS:
        draw_volumetric_bone(
            img, 
            uv[A], joints_3d[A, 2], 
            uv[B], joints_3d[B, 2],
            get_joint_radius(A), get_joint_radius(B),
            base_color, highlight_color
        )
        
    # Draw joint spheres on top of bones
    for idx in range(21):
        z = joints_3d[idx, 2]
        r = get_joint_radius(idx)
        t = max(1, int(r * FOCAL_LENGTH / z))
        cv2.circle(img, (int(uv[idx, 0]), int(uv[idx, 1])), t, joint_color, -1, cv2.LINE_AA)
        
        # Specular highlight on joint
        cv2.circle(img, (int(uv[idx, 0] - t*0.2), int(uv[idx, 1] - t*0.2)), max(1, int(t * 0.3)), highlight_color, -1, cv2.LINE_AA)
        
    return img, uv

def generate_dataset(num_samples, output_file, bg_folder, desc="Generating synthetic dataset"):
    bg_paths = list(Path(bg_folder).glob("*.jpg"))
    
    # Initialize H5
    with h5py.File(output_file, 'w') as f:
        images_ds = f.create_dataset('images',
                                     shape=(num_samples, 256, 256, 3),
                                     dtype=np.uint8,
                                     chunks=(32, 256, 256, 3),
                                     compression='lzf')
        
        kpts_ds = f.create_dataset('keypoints_3d',
                                   shape=(num_samples, 21, 3),
                                   dtype=np.float32)
        
        kpts_2d_ds = f.create_dataset('keypoints_2d',
                                      shape=(num_samples, 21, 2),
                                      dtype=np.float32)
        
        for i in tqdm(range(num_samples), desc=desc):
            # 1. Load random background
            if bg_paths:
                bg_path = random.choice(bg_paths)
                bg = cv2.imread(str(bg_path))
                if bg is not None:
                    bg = cv2.resize(bg, TARGET_SIZE)
                    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                else:
                    bg = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                bg = np.zeros((256, 256, 3), dtype=np.uint8)
                # Draw random background gradient
                color1 = np.array([random.randint(20, 80), random.randint(20, 80), random.randint(20, 80)])
                color2 = np.array([random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)])
                for y in range(256):
                    factor = y / 255.0
                    bg[y, :] = (1.0 - factor) * color1 + factor * color2
            
            # 2. Generate 3D joints
            joints_3d = generate_hand_pose()
            
            # 3. Render
            img, uv = render_synthetic_hand(joints_3d, bg)
            
            # Normalize annotations
            # 2D coordinates in [0, 1]
            uv_norm = uv / 256.0
            
            # 3D coordinates relative to wrist
            joints_3d_norm = joints_3d - joints_3d[0:1, :]
            
            # Save
            images_ds[i] = img
            kpts_ds[i] = joints_3d_norm.astype(np.float32)
            kpts_2d_ds[i] = uv_norm.astype(np.float32)
            
            if i % 500 == 0:
                f.flush()

if __name__ == '__main__':
    print("Starting synthetic hand pose dataset generator...")
    
    bg_folder = 'data/raw/training/rgb'
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = processed_dir / 'synthetic_train.h5'
    val_file = processed_dir / 'synthetic_val.h5'
    
    # We will generate a lightweight but robust dataset:
    # 15,000 training images, 1,500 validation images
    generate_dataset(15000, train_file, bg_folder, desc="Generating RHD-Synthetic Train")
    generate_dataset(1500, val_file, bg_folder, desc="Generating RHD-Synthetic Val")
    
    print("\nDataset generation complete!")
    print(f"Train H5: {train_file.absolute()}")
    print(f"Val H5: {val_file.absolute()}")
