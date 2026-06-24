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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_mesh_raw(nii_path):
    data = nib.load(nii_path).get_fdata()
    mask = (data > 0.5).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    return trimesh.Trimesh(vertices=verts, faces=faces)

def apply_norm(mesh, center, scale):
    verts = (np.array(mesh.vertices) - center) / scale
    return trimesh.Trimesh(vertices=verts, faces=np.array(mesh.faces))

def sample_points(meshes, n=20000):
    n_vol      = n // 2
    n_per_mesh = (n - n_vol) // len(meshes)
    samples = [np.random.uniform(-1, 1, (n_vol, 3))]
    for m in meshes:
        verts = np.array(m.vertices)
        samples.append(np.random.uniform(verts.min(0), verts.max(0), (n_per_mesh, 3)))
    return np.vstack(samples)

def compute_sdf(points, mesh):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    sdf, _, _ = pcu.signed_distance_to_mesh(points, verts, faces)
    return torch.tensor(sdf, dtype=torch.float32)

# ── 配置 ──────────────────────────────────────────────────────────────────────

ROOT = r'E:\Totalsegmentator_dataset_small_v201'

# channel 0: lung_lower_lobe_left  channel 1: lung_lower_lobe_right  channel 2: heart
# 缺失矩阵:  s0011=[1,1,0]  s0058=[1,0,1]  s0223=[0,1,1]
SUBJECTS = [
    {'paths': [f'{ROOT}\\s0011\\segmentations\\lung_lower_lobe_left.nii.gz',
               f'{ROOT}\\s0011\\segmentations\\lung_lower_lobe_right.nii.gz'],
     'organs': [0, 1]},
    {'paths': [f'{ROOT}\\s0058\\segmentations\\lung_lower_lobe_left.nii.gz',
               f'{ROOT}\\s0058\\segmentations\\heart.nii.gz'],
     'organs': [0, 2]},
    {'paths': [f'{ROOT}\\s0223\\segmentations\\lung_lower_lobe_right.nii.gz',
               f'{ROOT}\\s0223\\segmentations\\heart.nii.gz'],
     'organs': [1, 2]},
]

C          = 3       # 总 channel 数（所有器官）
LATENT_DIM = 256
N_EPOCHS   = 5000
BATCH_SIZE = 2048
LR_MODEL   = 1e-4
LR_LATENT  = 1e-3

# ── 全局归一化：收集所有 subject 所有可用器官的顶点 ──────────────────────────

all_raw_flat = [load_mesh_raw(p) for subj in SUBJECTS for p in subj['paths']]
all_verts    = np.vstack([np.array(m.vertices) for m in all_raw_flat])
global_center = all_verts.mean(axis=0)
global_scale  = np.abs(all_verts - global_center).max()

# ── 数据加载 ──────────────────────────────────────────────────────────────────

point_list      = []
sdf_list        = []
mask_list       = []
gt_meshes_list  = []   # 每个 subject: {global_ch: normalized trimesh}

for subj in SUBJECTS:
    raw_meshes = [load_mesh_raw(p) for p in subj['paths']]
    meshes     = [apply_norm(m, global_center, global_scale) for m in raw_meshes]
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

    l_sdf = masked_sdf_loss(pred_sdf, gt, mask)
    l_def = deformation_loss(delta_x, x)
    l_lat = latent_loss(z)
    loss  = l_sdf + 1e-3 * l_def + 1e-4 * l_lat

    optimizer.zero_grad()
    loss.backward()
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
plt.show()

resolution = 128
grid_1d = torch.linspace(-1.2, 1.2, resolution)
gx, gy, gz = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1).to(device)

with torch.no_grad():
    sdf_vals = []
    for i in range(0, len(grid), 4096):
        sdf_vals.append(template_net(grid[i:i+4096]).cpu())
    sdf_vol = torch.cat(sdf_vals).reshape(resolution, resolution, resolution, C).numpy()

organ_names   = ['lung_lower_lobe_left', 'lung_lower_lobe_right', 'heart']
organ_colors  = ['steelblue', 'tomato', 'mediumseagreen']

for c, name in enumerate(organ_names):
    print(f"{name} sdf_vol: min={sdf_vol[...,c].min():.4f}, max={sdf_vol[...,c].max():.4f}")

plotter = pv.Plotter(off_screen=True)
for c, (name, color) in enumerate(zip(organ_names, organ_colors)):
    vol = sdf_vol[..., c]
    try:
        verts, faces, _, _ = marching_cubes(vol, level=0.0)
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
            verts, faces, _, _ = marching_cubes(pred_vol[..., c], level=0.0)
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
