import os
import gc
import glob
import ssl
import traceback

# Windows 证书库里有损坏的证书，会导致 aiohttp（trame 依赖，export_html 需要）
# 在 import 时调用 ssl.create_default_context() 崩溃。这里把该异常吞掉，
# 不影响我们自己的网络请求（本脚本不发起任何网络请求）。
_orig_load_default_certs = ssl.SSLContext.load_default_certs
def _safe_load_default_certs(self, *args, **kwargs):
    try:
        return _orig_load_default_certs(self, *args, **kwargs)
    except ssl.SSLError:
        pass
ssl.SSLContext.load_default_certs = _safe_load_default_certs

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
from missingloss import deformation_loss, latent_loss



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

def sample_points(mesh, n_vol=10000, area_coef=20000, min_points=2000):
    # near-surface 点数按表面积 * 系数直接决定，不再是固定数量
    n_surf = max(int(mesh.area * area_coef), min_points)
    verts  = np.array(mesh.vertices)
    p_vol  = np.random.uniform(-1.2, 1.2, (n_vol, 3))
    p_surf = np.random.uniform(verts.min(0), verts.max(0), (n_surf, 3))
    return np.vstack([p_vol, p_surf])

def compute_sdf(points, mesh):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    sdf, _, _ = pcu.signed_distance_to_mesh(points, verts, faces)
    return torch.tensor(sdf, dtype=torch.float32)

def safe_level(vol):
    if vol.min() <= 0.0 <= vol.max():
        return 0.0
    return float(vol.min()) + 1e-5

def verts_to_coords(verts, resolution, vmin, vmax):
    return verts / (resolution - 1) * (vmax - vmin) + vmin

# ── 配置：扫描每个病人实际有哪些器官标注 ──────────────────────────────────────

ROOT        = r'E:\Totalsegmentator_dataset_small_v201'
SUBJECT_IDS = ['s0011', 's0058', 's0223', 's0250', 's0344']
# SUBJECT_IDS = ['s0011', 's0058']

# 指定要训练哪些器官：填了就只跑这几个；留空则按字母顺序取前 N_ORGANS 个
ORGAN_NAMES   = []
N_ORGANS      = 16    # ORGAN_NAMES 为空时生效，取前 N 个
OUTLIER_RATIO = 2.5   # voxel 数偏离该器官中位数超过这个倍数（或不到 1/这个倍数）就当缺失
MIN_SUBJECTS  = 2     # 一个器官至少要有这么多 subject 才训练模板

subject_organ_voxels = []   # 每个 subject: {organ_name: nvox}
subject_organ_paths  = []   # 每个 subject: {organ_name: path}
for sid in SUBJECT_IDS:
    seg_dir = os.path.join(ROOT, sid, 'segmentations')
    files   = glob.glob(os.path.join(seg_dir, '*.nii.gz'))
    voxel_map, path_map = {}, {}
    for f in files:
        name = os.path.basename(f)[:-len('.nii.gz')]
        nvox = int((nib.load(f).get_fdata() > 0.5).sum())
        if nvox > 0:
            voxel_map[name] = nvox
            path_map[name]  = f
    subject_organ_voxels.append(voxel_map)
    subject_organ_paths.append(path_map)

_all_organ_names = sorted(set().union(*[m.keys() for m in subject_organ_voxels]))
if ORGAN_NAMES:
    organ_names = [n for n in ORGAN_NAMES if n in set(_all_organ_names)]
else:
    organ_names = _all_organ_names[:N_ORGANS]

# 对每个器官，按体素数跟其他有这个器官的人比，离群就当这个 subject 缺这个器官
organ_subject_ids = {}   # 每个器官: 通过筛选的 subject_id 列表
for name in organ_names:
    have = [(i, subject_organ_voxels[i][name]) for i in range(len(SUBJECT_IDS)) if name in subject_organ_voxels[i]]
    if not have:
        continue
    median_nvox = np.median([n for _, n in have])
    kept = []
    for i, nvox in have:
        ratio = nvox / median_nvox if median_nvox > 0 else 1.0
        if ratio > OUTLIER_RATIO or ratio < 1 / OUTLIER_RATIO:
            print(f"{SUBJECT_IDS[i]}: {name} voxel 数离群 ({nvox} vs 中位数 {median_nvox:.0f})，视为缺失")
            continue
        kept.append(SUBJECT_IDS[i])
    organ_subject_ids[name] = kept
    print(f"{name}: {len(kept)}/{len(SUBJECT_IDS)} 个 subject 有效")

LATENT_DIM = 64    # 每个器官各自独立训练，subject 数量很少，不需要很大的 latent
N_EPOCHS   = 10000
BATCH_SIZE = 4096
LR_MODEL   = 1e-4
LR_LATENT  = 1e-4

CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── 全局共享归一化：让所有器官最终能拼到同一个坐标系里对比 ────────────────────
# （网络仍然每个器官独立训练，这里只是统一坐标约定，不影响训练本身）

all_raw_verts = []
for name, sids in organ_subject_ids.items():
    for sid in sids:
        path = subject_organ_paths[SUBJECT_IDS.index(sid)][name]
        all_raw_verts.append(np.array(load_mesh_raw(path).vertices))
all_raw_verts = np.vstack(all_raw_verts)
shared_center = all_raw_verts.mean(axis=0)
shared_scale  = np.abs(all_raw_verts - shared_center).max()

all_organ_sdf_vols      = {}   # organ_name -> sdf_vol，留着最后拼合可视化
all_organ_gt_meshes     = {}   # organ_name -> [normalized trimesh, ...]，原始未变形的 GT（仅作记录用）
all_organ_gt_meshes_warped = {}  # organ_name -> [(verts, faces), ...]，经 deform_net 变形后、真正能跟模板对比的 GT
all_organ_gt_meshes_orig   = {}  # organ_name -> [(verts, faces), ...]，未加 dx 的原始 GT（归一化后，与 template 同坐标系）
all_organ_offsets       = {}   # organ_name -> 这个器官在共享坐标系里"平均该在哪"的偏移量

# ── 每类器官单独训练一套模板 ───────────────────────────────────────────────────

for organ_name in organ_names:
    sids = organ_subject_ids.get(organ_name, [])
    if len(sids) < MIN_SUBJECTS:
        print(f"=== {organ_name}: 只有 {len(sids)} 个有效 subject，跳过 ===")
        continue

    print(f"=== 训练器官: {organ_name} ({len(sids)} 个 subject) ===")

    try:
        paths      = [subject_organ_paths[SUBJECT_IDS.index(sid)][organ_name] for sid in sids]
        raw_meshes = [load_mesh_raw(p) for p in paths]
        # 训练前先对齐：每个 subject 用自己这个器官的质心去中心化，避免不同 subject
        # 位置差太远导致模板裂成多份；scale 仍用全局共享的，保证跟其他器官可比
        subject_centers = [np.array(m.vertices).mean(axis=0) for m in raw_meshes]
        meshes = [apply_norm(m, c, shared_scale) for m, c in zip(raw_meshes, subject_centers)]
        all_organ_gt_meshes[organ_name] = meshes

        # 记录这个器官真实的平均位置相对共享坐标系原点的偏移，留着合并可视化时加回去
        organ_world_center = np.mean(subject_centers, axis=0)
        all_organ_offsets[organ_name] = (organ_world_center - shared_center) / shared_scale

        point_list, sdf_list = [], []
        for mesh in meshes:
            pts = sample_points(mesh)
            point_list.append(torch.tensor(pts, dtype=torch.float32))
            sdf_list.append(compute_sdf(pts, mesh).unsqueeze(-1))

        subject_ids_list = [torch.full((len(p),), i, dtype=torch.long) for i, p in enumerate(point_list)]

        point       = torch.cat(point_list, dim=0).to(device)
        sdf         = torch.cat(sdf_list,   dim=0).to(device)
        subject_ids = torch.cat(subject_ids_list).to(device)

        template_net = TemplateNet(1, hidden_dim=256).to(device)
        deform_net   = DeformNet(latent_dim=LATENT_DIM, hidden_dim=128).to(device)
        latent_codes = nn.Embedding(len(sids), LATENT_DIM).to(device)
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
            subj_ids = subject_ids[idx]

            z        = latent_codes(subj_ids)
            delta_x  = deform_net(torch.cat([x, z], dim=-1))
            x_prime  = x + delta_x
            pred_sdf = template_net(x_prime)

            grad_sdf = torch.autograd.grad(pred_sdf.sum(), x_prime, create_graph=True)[0]
            l_eik    = ((grad_sdf.norm(dim=-1) - 1) ** 2).mean()
            l_center = delta_x.mean(dim=0).pow(2).mean()

            l_sdf = torch.mean((pred_sdf - gt) ** 2)
            l_def = deformation_loss(delta_x, x)
            l_lat = latent_loss(z)
            loss  = l_sdf + 1e-3 * l_def + 1e-4 * l_lat + 1e-2 * l_eik + 1e-2 * l_center


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(template_net.parameters()) + list(deform_net.parameters()), max_norm=1.0
            )
            torch.nn.utils.clip_grad_norm_(latent_codes.parameters(), max_norm=0.1)
            optimizer.step()

            loss_history.append(l_sdf.item())
            if epoch % 1000 == 0:
                print(f"  epoch {epoch:5d} | sdf={l_sdf.item():.4f} | def={l_def.item():.6f} | lat={l_lat.item():.6f} | eik={l_eik.item():.4f} | center={l_center.item():.6f}")

        # plt.figure()
        # plt.plot(loss_history)
        # plt.xlabel('Epoch')
        # plt.ylabel('SDF Loss')
        # plt.title(f'Training Loss - {organ_name}')
        # plt.savefig(f'loss_curve_{organ_name}.png')
        # plt.close()

        torch.save({
            'organ_name':    organ_name,
            'sids':          sids,
            'template_net':  template_net.state_dict(),
            'deform_net':    deform_net.state_dict(),
            'latent_codes':  latent_codes.state_dict(),
            'latent_dim':    LATENT_DIM,
            'organ_offset':  all_organ_offsets[organ_name],
            'shared_scale':  shared_scale,
        }, os.path.join(CHECKPOINT_DIR, f'{organ_name}.pt'))

        # ── 可视化 ────────────────────────────────────────────────────────────

        resolution = 256
        grid_1d = torch.linspace(-1.2, 1.2, resolution)
        gx, gy, gz = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1).to(device)
        vmin, vmax = grid_1d[0].item(), grid_1d[-1].item()

        
        with torch.no_grad():
            sdf_vals = []
            for j in range(0, len(grid), 4096):
                sdf_vals.append(template_net(grid[j:j+4096]).cpu())
            sdf_vol = torch.cat(sdf_vals).reshape(resolution, resolution, resolution).numpy()

        all_organ_sdf_vols[organ_name] = sdf_vol

        warped_meshes_this_organ = []
        orig_meshes_this_organ   = []
        for i, sid in enumerate(sids):
            z_i = latent_codes.weight[i].detach()
            gt_verts = np.array(meshes[i].vertices).astype(np.float32)
            gt_faces = np.array(meshes[i].faces)
            with torch.no_grad():
                gt_x  = torch.tensor(gt_verts, dtype=torch.float32, device=device)
                gt_z  = z_i.unsqueeze(0).expand(len(gt_x), -1)
                gt_dx = deform_net(torch.cat([gt_x, gt_z], dim=-1))

                gt        = gt_x.cpu().numpy()
                gt_warped = (gt_x + gt_dx).cpu().numpy()

            warped_meshes_this_organ.append((gt_warped, gt_faces))
            orig_meshes_this_organ.append((gt, gt_faces))

        all_organ_gt_meshes_warped[organ_name] = warped_meshes_this_organ
        all_organ_gt_meshes_orig[organ_name]   = orig_meshes_this_organ

    except Exception as e:
        print(f"=== {organ_name} 训练/可视化失败，跳过: {e} ===")
        traceback.print_exc()
    finally:
        gc.collect()
        torch.cuda.empty_cache()

# ── 把所有器官的模板拼到同一张图里（共享坐标系，相对位置有意义） ──────────────
# 注意：这里用的是 all_organ_gt_meshes_warped（已经过各自 deform_net 变形），
# 不是原始的 all_organ_gt_meshes——后者跟 template 的 canonical 坐标系不在一个空间，
# 直接拿来画会出现"模板和 GT 对不上"的问题。

cmap = plt.colormaps['tab20']
combined_colors = [cmap(i % 20) for i in range(len(all_organ_sdf_vols))]

plotter = pv.Plotter(off_screen=True)
for (name, vol), color in zip(all_organ_sdf_vols.items(), combined_colors):
    offset = all_organ_offsets.get(name, np.zeros(3))
    try:
        verts, faces, _, _ = marching_cubes(vol, level=safe_level(vol))
        verts = verts_to_coords(verts, vol.shape[0], -1.2, 1.2) + offset
        faces_pv = np.column_stack([np.full(len(faces), 3), faces]).flatten()
        mesh_pv = pv.PolyData(verts.astype(np.float32), faces_pv)
        plotter.add_mesh(mesh_pv, color='red', opacity=0.5, label=name)
    except Exception as e:
        print(f"{name} combined marching cubes failed: {e}")
for name, warped_list in all_organ_gt_meshes_warped.items():
    offset = all_organ_offsets.get(name, np.zeros(3))
    for j, (gt_verts, gt_faces) in enumerate(warped_list):
        verts_off = gt_verts.astype(np.float32) + offset
        faces_pv = np.column_stack([np.full(len(gt_faces), 3), gt_faces]).flatten()
        gt_pv = pv.PolyData(verts_off, faces_pv)
        plotter.add_mesh(gt_pv, color='black', style='wireframe',
                         label=f'{name} subj{j} (GT warped)' if j == 0 else None)

plotter.add_legend()

plotter.export_html('temp+dx.html')
plotter.close()

# ── 合并图 2：template + 原始 GT（不加 dx），对比变形前后的差异 ──────────────────
plotter2 = pv.Plotter(off_screen=True)
for (name, vol), color in zip(all_organ_sdf_vols.items(), combined_colors):
    offset = all_organ_offsets.get(name, np.zeros(3))
    try:
        verts, faces, _, _ = marching_cubes(vol, level=safe_level(vol))
        verts = verts_to_coords(verts, vol.shape[0], -1.2, 1.2) + offset
        faces_pv = np.column_stack([np.full(len(faces), 3), faces]).flatten()
        mesh_pv = pv.PolyData(verts.astype(np.float32), faces_pv)
        plotter2.add_mesh(mesh_pv, color='red', opacity=0.5, label=name)
    except Exception as e:
        print(f"{name} combined (orig) marching cubes failed: {e}")
for name, orig_list in all_organ_gt_meshes_orig.items():
    offset = all_organ_offsets.get(name, np.zeros(3))
    for j, (gt_verts, gt_faces) in enumerate(orig_list):
        verts_off = gt_verts.astype(np.float32) + offset
        faces_pv = np.column_stack([np.full(len(gt_faces), 3), gt_faces]).flatten()
        gt_pv = pv.PolyData(verts_off, faces_pv)
        plotter2.add_mesh(gt_pv, color='black', style='wireframe',
                          label=f'{name} subj{j} (GT original)' if j == 0 else None)

plotter2.add_legend()

plotter2.export_html('temp+gt.html')
plotter2.close()

# ── 纯模板合并图（不含 GT wireframe，看各器官模板形状更清晰） ──────────────────

plotter2 = pv.Plotter(off_screen=True)
for (name, vol), color in zip(all_organ_sdf_vols.items(), combined_colors):
    offset = all_organ_offsets.get(name, np.zeros(3))
    try:
        verts, faces, _, _ = marching_cubes(vol, level=safe_level(vol))
        verts = verts_to_coords(verts, vol.shape[0], -1.2, 1.2) + offset
        faces_pv = np.column_stack([np.full(len(faces), 3), faces]).flatten()
        mesh_pv = pv.PolyData(verts.astype(np.float32), faces_pv) 
        plotter2.add_mesh(mesh_pv, color=color, opacity=0.7, label=name)
    except Exception as e:
        print(f"{name} template-only marching cubes failed: {e}")
plotter2.add_legend()

plotter2.export_html('tmponly.html')
plotter2.close()
