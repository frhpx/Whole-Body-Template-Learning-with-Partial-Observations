import os
import gc
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
import pyvista as pv

from model import TemplateNet, DeformNet
from missingloss import masked_sdf_loss, deformation_loss, latent_loss

gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 预处理 ────────────────────────────────────────────────────────────────────

def load_mesh_raw(nii_path):
    data = nib.load(nii_path).get_fdata()
    mask = (data > 0.5).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    return trimesh.Trimesh(vertices=verts, faces=faces)

def apply_norm(mesh, center, scale):
    verts = (np.array(mesh.vertices) - center) / scale
    return trimesh.Trimesh(vertices=verts, faces=np.array(mesh.faces))

def sample_points(meshes, n_vol=10000, area_coef=20000, min_per_mesh=2000):
    # 每个器官按自己的表面积 * 系数 直接决定点数，不再从固定预算里分摊
    samples = [np.random.uniform(-1, 1, (n_vol, 3))]
    for m in meshes:
        ni    = max(int(m.area * area_coef), min_per_mesh)
        verts = np.array(m.vertices)
        samples.append(np.random.uniform(verts.min(0), verts.max(0), (ni, 3)))
    return np.vstack(samples)

def compute_sdf(points, mesh):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    sdf, _, _ = pcu.signed_distance_to_mesh(points, verts, faces)
    return torch.tensor(sdf, dtype=torch.float32)

# ── 配置：扫描每个病人实际有哪些器官标注 ──────────────────────────────────────

ROOT          = r'E:\Totalsegmentator_dataset_small_v201'
SUBJECT_IDS   = ['s0011', 's0058', 's0223', 's0250', 's0310']

subject_organ_files = []
for sid in SUBJECT_IDS:
    seg_dir = os.path.join(ROOT, sid, 'segmentations')
    files   = glob.glob(os.path.join(seg_dir, '*.nii.gz'))
    organ_map = {}
    for f in files:
        name = os.path.basename(f)[:-len('.nii.gz')]
        data = nib.load(f).get_fdata()
        if (data > 0.5).sum() > 0:          # 文件存在但 mask 全空 = 视为缺失
            organ_map[name] = f
    subject_organ_files.append(organ_map)

# channel 顺序 = 所有病人出现过的器官名的并集（按字母排序，保证可复现），先只训前 20 个
organ_names = sorted(set().union(*[m.keys() for m in subject_organ_files]))[:20]
C = len(organ_names)

SUBJECTS = []
for sid, organ_map in zip(SUBJECT_IDS, subject_organ_files):
    paths  = [organ_map[name] for name in organ_names if name in organ_map]
    organs = [c for c, name in enumerate(organ_names) if name in organ_map]
    SUBJECTS.append({'paths': paths, 'organs': organs})
    missing = [name for name in organ_names if name not in organ_map]
    print(f"{sid}: 有 {len(organs)}/{C} 个器官, 缺失 {missing}")
LATENT_DIM = 256
N_EPOCHS   = 30000
BATCH_SIZE = 4096
LR_MODEL   = 1e-4
LR_LATENT  = 1e-4

# ── 刚性预对齐：用所有 subject 共有的器官算质心，平移对齐，再统一缩放 ──────────

common_organs = set(organ_names)
for organ_map in subject_organ_files:
    common_organs &= set(organ_map.keys())
common_organs = [name for name in organ_names if name in common_organs]
print(f"用于对齐的共有器官: {common_organs}")

raw_meshes_list = [[load_mesh_raw(p) for p in subj['paths']] for subj in SUBJECTS]

if common_organs:
    subject_centers = []
    for subj, raw_meshes in zip(SUBJECTS, raw_meshes_list):
        ref_verts = np.vstack([
            np.array(raw_meshes[local_idx].vertices)
            for local_idx, global_ch in enumerate(subj['organs'])
            if organ_names[global_ch] in common_organs
        ])
        subject_centers.append(ref_verts.mean(axis=0))
else:
    print("没有所有 subject 共有的器官，退化为全局统一中心")
    all_verts = np.vstack([np.array(m.vertices) for raw in raw_meshes_list for m in raw])
    subject_centers = [all_verts.mean(axis=0)] * len(SUBJECTS)

all_centered_verts = np.vstack([
    np.array(m.vertices) - subject_centers[i]
    for i, raw_meshes in enumerate(raw_meshes_list)
    for m in raw_meshes
])
global_scale = np.abs(all_centered_verts).max()

# ── 数据加载 ──────────────────────────────────────────────────────────────────

point_list      = []
sdf_list        = []
mask_list       = []
gt_meshes_list  = []   # 每个 subject: {global_ch: normalized trimesh}

for subj, raw_meshes, center in zip(SUBJECTS, raw_meshes_list, subject_centers):
    meshes     = [apply_norm(m, center, global_scale) for m in raw_meshes]
    pts        = sample_points(meshes)
    N          = len(pts)

    sdf_gt     = torch.zeros(N, C)
    organ_mask = torch.zeros(N, C, dtype=torch.bool)
    gt_meshes  = {}

    for local_idx, global_ch in enumerate(subj['organs']):
        sdf_gt[:, global_ch]     = compute_sdf(pts, meshes[local_idx])
        organ_mask[:, global_ch] = True
        gt_meshes[global_ch]     = meshes[local_idx]

    point_list.append(torch.tensor(pts, dtype=torch.float32))
    sdf_list.append(sdf_gt)
    mask_list.append(organ_mask)
    gt_meshes_list.append(gt_meshes)

subject_ids_list = [
    torch.full((len(pts),), i, dtype=torch.long)
    for i, pts in enumerate(point_list)
]

point       = torch.cat(point_list,      dim=0).to(device)
sdf         = torch.cat(sdf_list,        dim=0).to(device)
organ_mask  = torch.cat(mask_list,       dim=0).to(device)
subject_ids = torch.cat(subject_ids_list).to(device)

# ── 模型 ──────────────────────────────────────────────────────────────────────

template_net = TemplateNet(C, hidden_dim=256).to(device)
deform_net   = DeformNet(latent_dim=LATENT_DIM, hidden_dim=256).to(device)
latent_codes = nn.Embedding(len(SUBJECTS), LATENT_DIM).to(device)
nn.init.normal_(latent_codes.weight, mean=0, std=0.01)

optimizer = optim.Adam([
    {'params': template_net.parameters(), 'lr': LR_MODEL},
    {'params': deform_net.parameters(),   'lr': LR_MODEL},
    {'params': latent_codes.parameters(), 'lr': LR_LATENT},
])


loss_history = []

for epoch in range(N_EPOCHS):
    idx      = torch.randint(0, len(point), (BATCH_SIZE,))
    x        = point[idx].detach().requires_grad_(True)
    gt       = sdf[idx]
    mask     = organ_mask[idx]
    subj_ids = subject_ids[idx]

    z        = latent_codes(subj_ids)
    delta_x  = deform_net(torch.cat([x, z], dim=-1))
    x_prime  = x + delta_x
    pred_sdf = template_net(x_prime)

    l_sdf  = masked_sdf_loss(pred_sdf, gt, mask)
    l_def  = deformation_loss(delta_x, x)
    l_lat  = latent_loss(z)
    loss   = l_sdf + 1e-3 * l_def + 1e-4 * l_lat

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(template_net.parameters()) + list(deform_net.parameters()), max_norm=1.0
    )
    torch.nn.utils.clip_grad_norm_(latent_codes.parameters(), max_norm=0.1)
    optimizer.step()

    loss_history.append(l_sdf.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | sdf={l_sdf.item():.4f} | def={l_def.item():.6f} | lat={l_lat.item():.6f}")


# ── 可视化 ────────────────────────────────────────────────────────────────────

plt.figure()
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('SDF Loss')
plt.title('Training Loss (missing organs)')
plt.savefig('loss_curve_missing.png')
plt.close()

resolution = 128
grid_1d = torch.linspace(-1.2, 1.2, resolution)
gx, gy, gz = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1).to(device)

with torch.no_grad():
    sdf_vals = []
    for i in range(0, len(grid), 4096):
        sdf_vals.append(template_net(grid[i:i+4096]).cpu())
    sdf_vol = torch.cat(sdf_vals).reshape(resolution, resolution, resolution, C).numpy()

cmap = plt.colormaps['tab20']
organ_colors = [cmap(i % 20) for i in range(C)]

for c, name in enumerate(organ_names):
    print(f"{name} sdf_vol: min={sdf_vol[...,c].min():.4f}, max={sdf_vol[...,c].max():.4f}")

def safe_level(vol):
    if vol.min() <= 0.0 <= vol.max():
        return 0.0
    return float(vol.min()) + 1e-5   # 全正/全负时退化为提取最内层等值面

plotter = pv.Plotter(off_screen=True)
for c, (name, color) in enumerate(zip(organ_names, organ_colors)):
    vol = sdf_vol[..., c]
    try:
        verts, faces, _, _ = marching_cubes(vol, level=safe_level(vol))
        faces_pv = np.column_stack([np.full(len(faces), 3), faces]).flatten()
        mesh = pv.PolyData(verts.astype(np.float32), faces_pv)
        plotter.add_mesh(mesh, color=color, opacity=0.5, label=name)
    except Exception as e:
        print(f"{name} marching cubes failed: {e}")

plotter.add_legend()
plotter.export_html('template_combined_missing.html')
plotter.close()

# ── 每个 subject：预测重建 vs GT 对比 ─────────────────────────────────────────

def verts_to_coords(verts, resolution, vmin, vmax):
    return verts / (resolution - 1) * (vmax - vmin) + vmin

vmin, vmax = grid_1d[0].item(), grid_1d[-1].item()

for i, subj in enumerate(SUBJECTS):
    try:
        z_i = latent_codes.weight[i].detach()

        with torch.no_grad():
            pred_vals = []
            for j in range(0, len(grid), 4096):
                gj = grid[j:j+4096]
                zj = z_i.unsqueeze(0).expand(len(gj), -1)
                dx = deform_net(torch.cat([gj, zj], dim=-1))
                pred_vals.append(template_net(gj + dx).cpu())
            pred_vol = torch.cat(pred_vals).reshape(resolution, resolution, resolution, C).numpy()

        plotter = pv.Plotter(off_screen=True)

        for c, (name, color) in enumerate(zip(organ_names, organ_colors)):
            try:
                vol_c = pred_vol[..., c]
                verts, faces, _, _ = marching_cubes(vol_c, level=safe_level(vol_c))
                verts = verts_to_coords(verts, resolution, vmin, vmax)
                faces_pv = np.column_stack([np.full(len(faces), 3), faces]).flatten()
                mesh = pv.PolyData(verts.astype(np.float32), faces_pv)
                plotter.add_mesh(mesh, color=color, opacity=0.4, label=f'{name} (pred)')
            except Exception as e:
                print(f"subject {i} {name} pred marching cubes failed: {e}")

        for global_ch, gt_mesh in gt_meshes_list[i].items():
            gt_verts = np.array(gt_mesh.vertices).astype(np.float32)
            gt_faces = np.array(gt_mesh.faces)
            faces_pv = np.column_stack([np.full(len(gt_faces), 3), gt_faces]).flatten()
            gt_pv = pv.PolyData(gt_verts, faces_pv)
            plotter.add_mesh(gt_pv, color='black', style='wireframe',
                             label=f'{organ_names[global_ch]} (GT)')

        plotter.add_legend()
        plotter.export_html(f'subject_{i}_recon_vs_gt.html')
        plotter.close()
    except Exception as e:
        print(f"subject {i} 整体处理失败，跳过: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
