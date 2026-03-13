# cloth_tools.py (improved to match cloth_z_up.obj topology and winding)
import os, sys, math, argparse
import numpy as np

# --------------------- I/O ---------------------
def read_obj(path):
    """
    Read vertices (v), faces (f raw lines), texture coords (vt) and normals (vn).
    Faces are kept as raw strings to preserve indices like v/vt/vn.
    """
    V, F, VT, VN, OTH = [], [], [], [], []
    with open(path, "r") as f:
        for ln in f:
            if ln.startswith("v "):
                _, x, y, z, *rest = ln.strip().split()
                V.append([float(x), float(y), float(z)])
            elif ln.startswith("vt "):
                _, u, v, *rest = ln.strip().split()
                VT.append([float(u), float(v)])
            elif ln.startswith("vn "):
                _, nx, ny, nz, *rest = ln.strip().split()
                VN.append([float(nx), float(ny), float(nz)])
            elif ln.startswith("f "):
                F.append(ln.strip())
            else:
                OTH.append(ln.rstrip("\n"))
    return (np.array(V, float), F, np.array(VT, float) if len(VT) else None, np.array(VN, float) if len(VN) else None, OTH)

def write_obj(path, V, F, header="# generated obj\n", vt=None, vn=None, extras=None):
    """
    Write an OBJ. If vt and/or vn are provided, they will be written before faces.
    Faces may be raw strings (written verbatim) or 3-tuples of vertex indices (v only).
    `extras` is an optional list of lines to include (e.g., 'usemtl None', 's 1').
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        for v in V:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        if vt is not None:
            for t in vt:
                f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        if vn is not None:
            for n in vn:
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        if extras:
            for line in extras:
                f.write(str(line).rstrip("\n") + "\n")
        for fl in F:
            if isinstance(fl, str):
                f.write(fl + "\n")
            else:
                a, b, c = fl
                f.write(f"f {a} {b} {c}\n")

# --------------------- complexity tag ---------------------
def complexity_tag(nverts, nfaces):
    return int((nverts + nfaces) / 100)

# --------------------- grid & shape generators ---------------------
def _emit_obj(out_path, V, triangles, vt=None, vn_dir=(0.0,0.0,-1.0), extras=None, header="# generated obj\n"):
    V = np.asarray(V, float)
    vn = np.array([vn_dir], float) if vn_dir is not None else None
    # default UV: one per-vertex if not supplied
    if vt is None:
        # try to infer rectangular bounds for normalized uv
        minv, maxv = V.min(0), V.max(0)
        size = np.maximum(maxv - minv, 1e-12)
        uv = (V[:, :2] - minv[:2]) / size[:2]
        vt = uv.tolist()
    # faces as v/vt/vn strings
    F = []
    for (a,b,c) in triangles:
        if vn is not None:
            F.append(f"f {a}/{a}/1 {b}/{b}/1 {c}/{c}/1")
        else:
            F.append((a,b,c))
    write_obj(out_path, V, F, header=header, vt=vt, vn=vn, extras=extras)
    return out_path


def _build_grid(n, halfx, halfy, z=0.0):
    V=[]; vt=[]
    for j in range(n):
        y = -halfy + (2*halfy)*j/(n-1)
        for i in range(n):
            x = -halfx + (2*halfx)*i/(n-1)
            V.append([x,y,z])
            vt.append([i/(n-1), 1.0 - j/(n-1)])
    return np.array(V,float), vt


def _build_rect_grid(nx, ny, halfx, halfy, z=0.0):
    """
    Build a rectangular grid of vertices and UVs.
    """
    V = []; vt = []
    for j in range(ny):
        y = -halfy + (2*halfy) * j / (ny - 1)
        for i in range(nx):
            x = -halfx + (2*halfx) * i / (nx - 1)
            V.append([x, y, z])
            vt.append([i/(nx-1), 1.0 - j/(ny-1)])
    return np.array(V, float), vt


def _triangulate_grid(n, diagonal="A"):
    def vid(i,j): return j*n + i + 1  # 1-based
    tris=[]
    for j in range(n-1):
        for i in range(n-1):
            v00=vid(i,j); v10=vid(i+1,j); v01=vid(i,j+1); v11=vid(i+1,j+1)
            if diagonal=="A":
                tris.append((v00,v11,v10)); tris.append((v00,v01,v11))
            elif diagonal=="B":
                tris.append((v00,v10,v01)); tris.append((v10,v11,v01))
            elif diagonal=="checker":
                if (i+j)%2==0:
                    tris.append((v00,v11,v10)); tris.append((v00,v01,v11))
                else:
                    tris.append((v00,v10,v01)); tris.append((v10,v11,v01))
            elif diagonal=="row-alt":
                if j%2==0:
                    tris.append((v00,v11,v10)); tris.append((v00,v01,v11))
                else:
                    tris.append((v00,v10,v01)); tris.append((v10,v11,v01))
            elif diagonal=="col-alt":
                if i%2==0:
                    tris.append((v00,v11,v10)); tris.append((v00,v01,v11))
                else:
                    tris.append((v00,v10,v01)); tris.append((v10,v11,v01))
            else:
                tris.append((v00,v11,v10)); tris.append((v00,v01,v11))
    return tris


def _triangulate_rect(nx, ny, diagonal="A"):
    """
    Triangulate a rectangle grid (nx × ny) with diagonal pattern.
    """
    def vid(i, j): return j * nx + i + 1  # 1-based indexing
    tris = []
    for j in range(ny-1):
        for i in range(nx-1):
            v00 = vid(i, j); v10 = vid(i+1, j)
            v01 = vid(i, j+1); v11 = vid(i+1, j+1)
            if diagonal == "A":
                tris.append((v00, v11, v10)); tris.append((v00, v01, v11))
            elif diagonal == "B":
                tris.append((v00, v10, v01)); tris.append((v10, v11, v01))
            else:
                tris.append((v00, v11, v10)); tris.append((v00, v01, v11))
    return tris


def _affine(V, shear_x=0.0, shear_y=0.0, rot_deg=0.0, scale_x=1.0, scale_y=1.0):
    th = math.radians(rot_deg)
    c,s = math.cos(th), math.sin(th)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1.0]])
    S = np.array([[scale_x,0,0],[0,scale_y,0],[0,0,1]])
    Sh = np.array([[1,shear_x,0],[shear_y,1,0],[0,0,1]])
    return (V @ (S@Sh@R).T)


def _edge_ruffle(V, n, amp=0.0, freq=8):
    if amp==0: return V
    V = V.copy()
    N=n
    def is_edge(i,j): return (i==0 or j==0 or i==N-1 or j==N-1)
    for j in range(N):
        for i in range(N):
            if is_edge(i,j):
                t = (i + j) / (N-1)
                V[j*N+i,2] += amp * math.sin(2*math.pi*freq*t)
    return V


def _jitter(V, sigma=0.0):
    if sigma<=0: return V
    J = np.random.normal(0.0, sigma, size=V.shape)
    J[:,2]*=0.2
    return V + J


def write_grid_obj(path, n=9, edge_len=1.0, z=0.0, thickness_mm=0.0,
                   blender_compat=True, object_name=None, mtllib=None,
                   diagonal="A", shear_x=0.0, shear_y=0.0, rot_deg=0.0,
                   scale_x=1.0, scale_y=1.0, edge_ruffle_amp=0.0, edge_ruffle_freq=8,
                   jitter_mm=0.0, uv_scale_u=1.0, uv_scale_v=1.0,
                   uv_offset_u=0.0, uv_offset_v=0.0,
                   uv_rot_deg=0.0, uv_mirror_u=False, uv_mirror_v=False,
                   uv_noise=0.0, uv_wrap=False, uv_match_like=None):
    """
    Generate n×n grid with many DR options.
    """
    verts_count = n*n
    faces_count = 2*(n-1)*(n-1)
    comp = complexity_tag(verts_count, faces_count)

    base, ext = os.path.splitext(path)
    is_dir = (ext == "" and (path.endswith(os.sep) or os.path.isdir(path)))
    if is_dir:
        out_dir = path
        fname = f"mj_square_n{n}_v{verts_count}_f{faces_count}_complex{comp}.obj"
        out_path = os.path.join(out_dir, fname)
    else:
        if ext.lower() != ".obj":
            out_dir = os.path.dirname(path) or "."
            stem = os.path.basename(path)
            fname = f"{stem}_n{n}_v{verts_count}_f{faces_count}_complex{comp}.obj"
            out_path = os.path.join(out_dir, fname)
        else:
            out_dir = os.path.dirname(path) or "."
            out_path = path

    # geometry
    V, vt = _build_grid(n, edge_len, edge_len, z)
    V = _edge_ruffle(V, n, amp=edge_ruffle_amp, freq=edge_ruffle_freq)
    V = _affine(V, shear_x=shear_x, shear_y=shear_y, rot_deg=rot_deg, scale_x=scale_x, scale_y=scale_y)
    V = _jitter(V, sigma=jitter_mm*1e-3)

    # triangles
    tris = _triangulate_grid(n, diagonal=diagonal)

    # UV transform
    vt2=[]
    for (u,v) in vt:
        u2 = u*uv_scale_u + uv_offset_u
        v2 = v*uv_scale_v + uv_offset_v
        vt2.append([u2,v2])
    vt2 = _uv_transform(vt2,
                        uv_scale_u=1.0, uv_scale_v=1.0,
                        uv_offset_u=0.0, uv_offset_v=0.0,
                        uv_rot_deg=uv_rot_deg, uv_mirror_u=uv_mirror_u, uv_mirror_v=uv_mirror_v,
                        uv_noise=uv_noise, uv_wrap=uv_wrap, uv_match_like=uv_match_like)

    # extras
    extras=[]
    if mtllib: extras.append(f"mtllib {mtllib}")
    if object_name: extras.append(f"o {object_name}")
    extras += ["usemtl None","s 1"]

    _emit_obj(out_path, V, tris, vt=vt2, vn_dir=(0,0,-1), extras=extras,
              header="# mj DR grid\n")
    print(f"Wrote {out_path} (verts={verts_count}, faces={faces_count}, complexity={comp})")
    return out_path


def write_poncho_obj(path, n=33, edge_len=1.0, hole_radius=0.25,
                    diagonal="A",
                    uv_scale_u=1.0, uv_scale_v=1.0,
                    uv_offset_u=0.0, uv_offset_v=0.0,
                    uv_rot_deg=0.0, uv_mirror_u=False, uv_mirror_v=False,
                    uv_noise=0.0, uv_wrap=False, uv_match_like=None,
                    object_name=None, mtllib=None):
    """Square grid with a circular head hole (removed faces)."""
    V, vt = _build_grid(n, edge_len, edge_len, 0.0)
    tris = _triangulate_grid(n, diagonal=diagonal)
    # mask: keep triangles only if all three verts are outside hole
    def keep_vid(vid):
        i0 = vid-1
        p = V[i0]
        r = math.hypot(p[0], p[1])
        return r >= hole_radius
    keep=[]
    for (a,b,c) in tris:
        if keep_vid(a) and keep_vid(b) and keep_vid(c):
            keep.append((a,b,c))
    # reindex to compact vertex list
    used = sorted({i for tri in keep for i in tri})
    remap = {old:i+1 for i,old in enumerate(used)}
    V2 = V[np.array(used)-1]
    vt2 = [vt[i-1] for i in used]
    vt2 = _uv_transform(vt2,
                        uv_scale_u=uv_scale_u, uv_scale_v=uv_scale_v,
                        uv_offset_u=uv_offset_u, uv_offset_v=uv_offset_v,
                        uv_rot_deg=uv_rot_deg, uv_mirror_u=uv_mirror_u, uv_mirror_v=uv_mirror_v,
                        uv_noise=uv_noise, uv_wrap=uv_wrap, uv_match_like=uv_match_like)
    tris2 = [(remap[a], remap[b], remap[c]) for (a,b,c) in keep]
    extras=[]
    if mtllib: extras.append(f"mtllib {mtllib}")
    if object_name: extras.append(f"o {object_name}")
    extras += ["usemtl None","s 1"]
    out_path = path if path.endswith('.obj') else os.path.join(path, f"poncho_n{n}.obj")
    _emit_obj(out_path, V2, tris2, vt=vt2, vn_dir=(0,0,-1), extras=extras, header="# mj poncho\n")
    print(f"Wrote {out_path} (verts={len(V2)}, faces={len(tris2)})")
    return out_path


def write_skirt_obj(path, na=64, nr=16, r_inner=0.1, r_outer=1.0, flare_pow=1.0,
                     jitter_mm=0.0, uv_tile_u=1.0, uv_tile_v=1.0,
                     uv_rot_deg=0.0, uv_mirror_u=False, uv_mirror_v=False,
                     uv_noise=0.0, uv_wrap=False, uv_match_like=None,
                     object_name=None, mtllib=None):
    """Polar annulus (skirt-like) with radial rings."""
    # verts
    V=[]; vt=[]
    for j in range(nr):
        t = j/(nr-1)
        r = r_inner + (r_outer - r_inner)*(t**flare_pow)
        for i in range(na):
            a = 2*math.pi * i/na
            x = r*math.cos(a); y = r*math.sin(a)
            V.append([x,y,0.0])
            vt.append([(i/na)*uv_tile_u, (1.0-t)*uv_tile_v])
    V = np.array(V,float)
    # faces
    def vid(i,j): return j*na + (i%na) + 1
    tris=[]
    for j in range(nr-1):
        for i in range(na):
            v00=vid(i,j); v10=vid(i+1,j); v01=vid(i,j+1); v11=vid(i+1,j+1)
            tris.append((v00,v11,v10)); tris.append((v00,v01,v11))
    # jitter
    if jitter_mm>0:
        V = _jitter(V, sigma=jitter_mm*1e-3)
    vt2 = _uv_transform(vt,
                        uv_scale_u=1.0, uv_scale_v=1.0,
                        uv_offset_u=0.0, uv_offset_v=0.0,
                        uv_rot_deg=uv_rot_deg, uv_mirror_u=uv_mirror_u, uv_mirror_v=uv_mirror_v,
                        uv_noise=uv_noise, uv_wrap=uv_wrap, uv_match_like=uv_match_like)
    # extras
    extras=[]
    if mtllib: extras.append(f"mtllib {mtllib}")
    if object_name: extras.append(f"o {object_name}")
    extras += ["usemtl None","s 1"]
    out_path = path if path.endswith('.obj') else os.path.join(path, f"skirt_na{na}_nr{nr}.obj")
    _emit_obj(out_path, V, tris, vt=vt2, vn_dir=(0,0,-1), extras=extras, header="# mj skirt annulus\n")
    print(f"Wrote {out_path} (verts={len(V)}, faces={len(tris)})")
    return out_path


def write_scarf_obj(path, nx=33, ny=17, half_w=1.0, half_h=0.5,
                    diagonal="A",
                    uv_scale_u=1.0, uv_scale_v=1.0,
                    uv_offset_u=0.0, uv_offset_v=0.0,
                    uv_rot_deg=0.0, uv_mirror_u=False, uv_mirror_v=False,
                    uv_noise=0.0, uv_wrap=False, uv_match_like=None,
                    object_name=None, mtllib=None):
    """Rectangular cloth (scarf)."""
    V, vt = _build_rect_grid(nx, ny, half_w, half_h, 0.0)
    tris = _triangulate_rect(nx, ny, diagonal=diagonal)
    # UV transform
    vt2 = _uv_transform(vt,
                        uv_scale_u=uv_scale_u, uv_scale_v=uv_scale_v,
                        uv_offset_u=uv_offset_u, uv_offset_v=uv_offset_v,
                        uv_rot_deg=uv_rot_deg, uv_mirror_u=uv_mirror_u, uv_mirror_v=uv_mirror_v,
                        uv_noise=uv_noise, uv_wrap=uv_wrap, uv_match_like=uv_match_like)
    # extras
    extras = []
    if mtllib: extras.append(f"mtllib {mtllib}")
    if object_name: extras.append(f"o {object_name}")
    extras += ["usemtl None", "s 1"]
    # output
    out_path = path if path.endswith(".obj") else os.path.join(path, f"scarf_{nx}x{ny}.obj")
    _emit_obj(out_path, V, tris, vt=vt2, vn_dir=(0,0,-1), extras=extras, header="# mj scarf\n")
    print(f"Wrote {out_path} (verts={len(V)}, faces={len(tris)})")
    return out_path


def write_cape_obj(path, n=33, half=1.0, diagonal="A",
                   cut_tl=0.0, cut_tr=0.0, cut_bl=0.0, cut_br=0.0,
                   hem_delete_prob=0.0,
                   uv_scale_u=1.0, uv_scale_v=1.0,
                   uv_offset_u=0.0, uv_offset_v=0.0,
                   uv_rot_deg=0.0, uv_mirror_u=False, uv_mirror_v=False,
                   uv_noise=0.0, uv_wrap=False, uv_match_like=None,
                   object_name=None, mtllib=None):
    """
    Square cloth (n x n) with right-triangle corner cutouts, plus optional irregular hem.
    cut_* are in meters (0..half); any triangle whose *all vertices* fall inside the cut area is removed.
    If hem_delete_prob>0, boundary triangles are randomly dropped.
    """
    V, vt = _build_grid(n, half, half, 0.0)
    tris = _triangulate_grid(n, diagonal=diagonal)

    # helpers to test corner cutouts
    def in_tl(p):
        if cut_tl <= 0: return False
        return (p[0] <= -half + cut_tl) and (p[1] >= half - cut_tl) and ((p[0] + p[1]) <= (-half + half))
    def in_tr(p):
        if cut_tr <= 0: return False
        return (p[0] >= half - cut_tr) and (p[1] >= half - cut_tr) and ((-p[0] + p[1]) <= (-half + half))
    def in_bl(p):
        if cut_bl <= 0: return False
        return (p[0] <= -half + cut_bl) and (p[1] <= -half + cut_bl) and ((-p[0] + -p[1]) <= (half + half))
    def in_br(p):
        if cut_br <= 0: return False
        return (p[0] >= half - cut_br) and (p[1] <= -half + cut_br) and ((p[0] + -p[1]) <= (half + half))

    def in_cut(p):
        return in_tl(p) or in_tr(p) or in_bl(p) or in_br(p)

    keep = []
    rng = np.random.default_rng()
    N = n
    def is_edge_vid(vid):
        idx = vid - 1
        j, i = divmod(idx, n)
        return (i == 0 or j == 0 or i == N-1 or j == N-1)

    for (a, b, c) in tris:
        Pa, Pb, Pc = V[a-1], V[b-1], V[c-1]
        # remove if all three inside any cutout region
        if in_cut(Pa) and in_cut(Pb) and in_cut(Pc):
            continue
        # irregular hem: drop some boundary triangles
        if hem_delete_prob > 0.0 and (is_edge_vid(a) or is_edge_vid(b) or is_edge_vid(c)):
            if rng.random() < hem_delete_prob:
                continue
        keep.append((a, b, c))

    # compact reindex
    used = sorted({i for tri in keep for i in tri})
    remap = {old: i+1 for i, old in enumerate(used)}
    V2 = V[np.array(used)-1]
    vt2 = [vt[i-1] for i in used]
    tris2 = [(remap[a], remap[b], remap[c]) for (a, b, c) in keep]

    # UV transform
    vt2 = _uv_transform(vt2,
                        uv_scale_u=uv_scale_u, uv_scale_v=uv_scale_v,
                        uv_offset_u=uv_offset_u, uv_offset_v=uv_offset_v,
                        uv_rot_deg=uv_rot_deg, uv_mirror_u=uv_mirror_u, uv_mirror_v=uv_mirror_v,
                        uv_noise=uv_noise, uv_wrap=uv_wrap, uv_match_like=uv_match_like)

    extras = []
    if mtllib: extras.append(f"mtllib {mtllib}")
    if object_name: extras.append(f"o {object_name}")
    extras += ["usemtl None", "s 1"]
    out_path = path if path.endswith(".obj") else os.path.join(path, f"cape_n{n}.obj")
    _emit_obj(out_path, V2, tris2, vt=vt2, vn_dir=(0,0,-1), extras=extras, header="# mj cape\n")
    print(f"Wrote {out_path} (verts={len(V2)}, faces={len(tris2)})")
    return out_path

# --------------------- rotation helpers ---------------------
def Rx(a): c,s = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c,s = math.cos(a), math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c,s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def best_zup_rotation(V):
    deg = math.pi/2
    cands = [
        np.eye(3),
        Rx(+deg), Rx(-deg),
        Ry(+deg), Ry(-deg),
        Rz(+deg), Rz(-deg),
        Rx(deg)@Ry(deg), Rx(deg)@Ry(-deg),
        Ry(deg)@Rx(deg), Ry(-deg)@Rx(deg),
    ]
    best_R, best_span = None, 1e9
    for R in cands:
        V2 = V @ R.T
        zspan = V2[:,2].max() - V2[:,2].min()
        if zspan < best_span:
            best_span, best_R = zspan, R
    return best_R

def rotate_to_zup(src, dst=None, thickness_mm=1.0):
    """Rotate any OBJ to Z-up and compress thickness to ~thickness_mm (single surface).
       Preserves vt/vn (and writes them if present in source)."""
    V, F, VT, VN, OTH = read_obj(src)
    if V.size == 0:
        raise ValueError("No vertices found in source OBJ.")

    Vc = V - V.mean(0)
    R  = best_zup_rotation(Vc)
    V2 = (Vc @ R.T)

    # compress thickness to requested span
    V2[:,2] -= V2[:,2].mean()
    zspan = V2[:,2].ptp()
    if zspan > 1e-12:
        V2[:,2] *= ((float(thickness_mm) * 1e-3) / zspan)

    nverts, nfaces = len(V2), len(F)
    comp = complexity_tag(nverts, nfaces)

    base, ext = os.path.splitext(src)
    out_path = dst or f"{base}_zup_complex{comp}{ext if ext else '.obj'}"

    # Preserve extras that are harmless (filter comments)
    extras = [ln for ln in OTH if (ln.startswith("usemtl") or ln.startswith("s "))]

    write_obj(out_path, V2, F, header="# z-up flat export\n", vt=VT, vn=VN, extras=extras)

    size_est = np.linalg.norm(V2.max(0) - V2.min(0))
    print(f"Wrote: {out_path} (verts={nverts}, faces={nfaces}, size≈{size_est:.3f} m, complexity={comp}, thickness≈{float(thickness_mm)*1e-3:.6f} m)")
    return out_path

# --------------------- compare ---------------------
def _compare(a, b, tol=1e-6, ignore_winding=True):
    A = read_obj(a)
    B = read_obj(b)
    AV, AF, AVT, AVN, _ = A
    BV, BF, BVT, BVN, _ = B

    def key(p):
        inv = 1.0 / max(tol, 1e-12)
        return tuple(int(round(c * inv)) for c in p)

    # Build map from position->index for A
    amap = {key(v): i+1 for i, v in enumerate(AV)}

    # Check vertex sets
    missing = []
    for j, v in enumerate(BV, start=1):
        if key(v) not in amap:
            missing.append((j, v))
    same_vertices = (len(missing) == 0 and len(AV) == len(BV))

    def face_to_idx_set(F_raw, sort_vertices=True):
        tri = []
        for ln in F_raw:
            parts = ln.split()[1:]
            idx = []
            for p in parts:
                t = p.split('/')
                idx.append(int(t[0]))
            tri.append(tuple(sorted(idx)) if sort_vertices else tuple(idx))
        return set(tri)

    # Remap B faces into A index space via position
    b_to_a = {j: amap[key(v)] for j, v in enumerate(BV, start=1) if key(v) in amap}

    def remap_faces(F_raw, m, sort_vertices=True):
        tri = []
        for ln in F_raw:
            parts = ln.split()[1:]
            idx = []
            for p in parts:
                t = p.split('/')
                vi = int(t[0])
                idx.append(m[vi])
            tri.append(tuple(sorted(idx)) if sort_vertices else tuple(idx))
        return set(tri)

    Aset = face_to_idx_set(AF, sort_vertices=ignore_winding)
    Bset = remap_faces(BF, b_to_a, sort_vertices=ignore_winding) if same_vertices else set()

    same_tris = (Aset == Bset)

    # Bounding boxes (for quick visual sanity)
    bbA = (AV.min(0), AV.max(0))
    bbB = (BV.min(0), BV.max(0))

    # Differences / summaries
    print("Vertices:", len(AV), len(BV), "same_set=", same_vertices)
    print("Faces:", len(AF), len(BF), "same_tris=", same_tris, "(ignore_winding=" + str(ignore_winding) + ")")

    # vt/vn summaries
    print("UVs:", 0 if AVT is None else len(AVT), 0 if BVT is None else len(BVT))
    print("Normals:", 0 if AVN is None else len(AVN), 0 if BVN is None else len(BVN))

    # Pretty-print BBoxes (round to 6 decimals for readability)
    fmt = lambda a: [float(f"{v:.6f}") for v in a.tolist()]
    print("BBox A:", (fmt(bbA[0]), fmt(bbA[1])))
    print("BBox B:", (fmt(bbB[0]), fmt(bbB[1])))

    # Max bbox delta (absolute)
    bb_delta = np.maximum(np.abs(bbA[0]-bbB[0]), np.abs(bbA[1]-bbB[1])).max()
    print("BBox max|Δ|:", float(bb_delta))

    if not same_vertices:
        print("Missing in A (positions from B):", len(missing))
    if not same_tris:
        print("Tri diff count:", len(Aset ^ Bset))

    ok = same_vertices and same_tris
    print("Summary:", "MATCH" if ok else "DIFF")
    return ok

# --------------------- stats ---------------------
def obj_stats(path="cloth_z_up.obj"):
    V, F, VT, VN, OTH = read_obj(path)
    nverts, nfaces = len(V), len(F)
    comp = complexity_tag(nverts, nfaces)

    # try to infer n for a square grid
    import math
    n = None
    rt = int(math.isqrt(nverts))
    if rt * rt == nverts:
        # if it is a regular n×n grid, faces should match 2*(rt-1)^2
        if 2 * (rt - 1) * (rt - 1) == nfaces:
            n = rt

    print(f"path: {path}")
    if n is not None:
        print(f"n = {n}  (grid detected)")
    else:
        print("n = unknown (not a perfect n×n grid)")
    print(f"verts = {nverts}")
    print(f"faces = {nfaces}")
    print(f"complex = {comp}")
    if VT is not None:
        print(f"vt = {len(VT)} (texture coords present)")
    if VN is not None:
        print(f"vn = {len(VN)} (normals present)")

# --- UV utilities for DR ---
def _uv_transform(vt, *,
                  uv_scale_u=1.0, uv_scale_v=1.0,
                  uv_offset_u=0.0, uv_offset_v=0.0,
                  uv_rot_deg=0.0, uv_mirror_u=False, uv_mirror_v=False,
                  uv_noise=0.0, uv_wrap=False,
                  uv_match_like=None):
    """
    Apply UV transforms in this order:
      [match-like] -> mirror -> rotate(about 0.5,0.5) -> scale -> offset -> noise -> wrap
    If uv_match_like is set (path to an OBJ with vt), we first map base [0..1]^2 into that vt bbox.
    """
    import numpy as np
    U = np.array(vt, float)

    # optional match-like (fit base [0,1]^2 to reference vt bbox)
    if uv_match_like:
        try:
            V2, F2, VT2, VN2, OTH2 = read_obj(uv_match_like)
            if VT2 is not None and len(VT2) > 0:
                lo = VT2.min(0); hi = VT2.max(0)
                S = hi - lo; S[S == 0] = 1.0
                U[:, 0] = lo[0] + U[:, 0] * S[0]
                U[:, 1] = lo[1] + U[:, 1] * S[1]
        except Exception:
            pass  # ignore if reference can't be read

    # mirror about 0.5
    if uv_mirror_u: U[:, 0] = 1.0 - U[:, 0]
    if uv_mirror_v: U[:, 1] = 1.0 - U[:, 1]

    # rotate about center (0.5, 0.5)
    if abs(uv_rot_deg) > 1e-12:
        th = math.radians(uv_rot_deg)
        c, s = math.cos(th), math.sin(th)
        cx, cy = 0.5, 0.5
        x = U[:, 0] - cx
        y = U[:, 1] - cy
        U[:, 0] = cx + c * x - s * y
        U[:, 1] = cy + s * x + c * y

    # scale + offset
    U[:, 0] = U[:, 0] * uv_scale_u + uv_offset_u
    U[:, 1] = U[:, 1] * uv_scale_v + uv_offset_v

    # noise
    if uv_noise > 0.0:
        U += np.random.normal(0.0, uv_noise, size=U.shape)

    # wrap
    if uv_wrap:
        U = U % 1.0

    return U.tolist()

# --------------------- CLI ---------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Cloth mesh tools: make grid or rotate OBJ to Z-up.")
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("make-grid", help="Generate an n×n grid OBJ with complexity in filename (with DR options).")
    g.add_argument("--out", required=True, help="Output path OR directory OR stem.")
    g.add_argument("--n", type=int, default=9, help="Vertices per side (default 9).")
    g.add_argument("--edge", type=float, default=1.0, help="Half-extent in meters in blender-compat mode; full width in legacy mode.")
    g.add_argument("--z", type=float, default=0.0, help="Z height (default 0.0).")
    g.add_argument("--thickness-mm", type=float, default=0.0, help="Ignored (kept for back-compat).")
    g.add_argument("--no-blender-compat", action="store_true", help="(Back-compat only)")
    g.add_argument("--object-name", default=None, help="Write an 'o <name>' line before faces.")
    g.add_argument("--mtl", dest="mtllib", default=None, help="Write an 'mtllib <file>' line before faces.")
    g.add_argument("--diagonal", choices=["A","B","checker","row-alt","col-alt"], default="A")
    g.add_argument("--shear-x", type=float, default=0.0)
    g.add_argument("--shear-y", type=float, default=0.0)
    g.add_argument("--rot-deg", type=float, default=0.0)
    g.add_argument("--scale-x", type=float, default=1.0)
    g.add_argument("--scale-y", type=float, default=1.0)
    g.add_argument("--edge-ruffle-amp", type=float, default=0.0, help="Meters of sinusoidal z-offset along boundary")
    g.add_argument("--edge-ruffle-freq", type=int, default=8)
    g.add_argument("--jitter-mm", type=float, default=0.0)
    g.add_argument("--uv-scale", nargs=2, type=float, default=[1.0,1.0], metavar=("USCALE","VSCALE"))
    g.add_argument("--uv-offset", nargs=2, type=float, default=[0.0,0.0], metavar=("UOFF","VOFF"))
    g.add_argument("--uv-rot-deg", type=float, default=0.0)
    g.add_argument("--uv-mirror-u", action="store_true")
    g.add_argument("--uv-mirror-v", action="store_true")
    g.add_argument("--uv-noise", type=float, default=0.0)
    g.add_argument("--uv-wrap", action="store_true")
    g.add_argument("--uv-match-like", default=None)

    sc = sub.add_parser("make-scarf", help="Rectangular scarf cloth.")
    sc.add_argument("--out", required=True)
    sc.add_argument("--nx", type=int, default=33)
    sc.add_argument("--ny", type=int, default=17)
    sc.add_argument("--half-w", type=float, default=1.0)
    sc.add_argument("--half-h", type=float, default=0.5)
    sc.add_argument("--diagonal", choices=["A","B"], default="A")
    sc.add_argument("--uv-scale", nargs=2, type=float, default=[1.0,1.0])
    sc.add_argument("--uv-offset", nargs=2, type=float, default=[0.0,0.0])
    sc.add_argument("--uv-rot-deg", type=float, default=0.0)
    sc.add_argument("--uv-mirror-u", action="store_true")
    sc.add_argument("--uv-mirror-v", action="store_true")
    sc.add_argument("--uv-noise", type=float, default=0.0)
    sc.add_argument("--uv-wrap", action="store_true")
    sc.add_argument("--uv-match-like", default=None)
    sc.add_argument("--object-name", default=None)
    sc.add_argument("--mtl", dest="mtllib", default=None)

    cp = sub.add_parser("make-cape", help="Square cape with corner cutouts & irregular hem.")
    cp.add_argument("--out", required=True)
    cp.add_argument("--n", type=int, default=33)
    cp.add_argument("--half", type=float, default=1.0)
    cp.add_argument("--diagonal", choices=["A","B","checker","row-alt","col-alt"], default="A")
    cp.add_argument("--cut-tl", type=float, default=0.0)
    cp.add_argument("--cut-tr", type=float, default=0.0)
    cp.add_argument("--cut-bl", type=float, default=0.0)
    cp.add_argument("--cut-br", type=float, default=0.0)
    cp.add_argument("--hem-delete-prob", type=float, default=0.0)
    cp.add_argument("--uv-scale", nargs=2, type=float, default=[1.0,1.0])
    cp.add_argument("--uv-offset", nargs=2, type=float, default=[0.0,0.0])
    cp.add_argument("--uv-rot-deg", type=float, default=0.0)
    cp.add_argument("--uv-mirror-u", action="store_true")
    cp.add_argument("--uv-mirror-v", action="store_true")
    cp.add_argument("--uv-noise", type=float, default=0.0)
    cp.add_argument("--uv-wrap", action="store_true")
    cp.add_argument("--uv-match-like", default=None)
    cp.add_argument("--object-name", default=None)
    cp.add_argument("--mtl", dest="mtllib", default=None)

    r = sub.add_parser("rotate", help="Rotate an OBJ to Z-up and compress thickness.")
    r.add_argument("--src", required=True, help="Source OBJ.")
    r.add_argument("--dst", default=None, help="Destination OBJ (auto-named if omitted).")
    r.add_argument("--thickness-mm", type=float, default=1.0, help="Target total z-span in millimeters (default 1.0).")

    s = sub.add_parser("stats", help="Print n/verts/faces/complex for an OBJ.")
    s.add_argument("--src", required=True, help="Source OBJ to analyze.")

    c = sub.add_parser("compare", help="Compare two OBJs for vertex/face equality (ignoring winding).")
    c.add_argument("--a", required=True, help="First OBJ.")
    c.add_argument("--b", required=True, help="Second OBJ.")
    c.add_argument("--tol", type=float, default=1e-6, help="Position tolerance (default 1e-6).")
    c.add_argument("--ignore-winding", action="store_true", help="Ignore vertex winding when comparing.")

    p = sub.add_parser("make-poncho", help="Square cloth with circular head hole (domain randomization ready).")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=33)
    p.add_argument("--edge", type=float, default=1.0)
    p.add_argument("--hole-radius", type=float, default=0.25)

    k = sub.add_parser("make-skirt", help="Radial annulus skirt.")
    k.add_argument("--out", required=True)
    k.add_argument("--na", type=int, default=64)
    k.add_argument("--nr", type=int, default=16)
    k.add_argument("--r-inner", type=float, default=0.1)
    k.add_argument("--r-outer", type=float, default=1.0)
    k.add_argument("--flare-pow", type=float, default=1.0)
    k.add_argument("--jitter-mm", type=float, default=0.0)
    k.add_argument("--uv-tile", nargs=2, type=float, default=[1.0,1.0], metavar=("US","VS"))

    return ap.parse_args()

def main():
    args = parse_args()
    if args.cmd == "make-grid":
        write_grid_obj(
            args.out, n=args.n, edge_len=args.edge, z=args.z,
            thickness_mm=args.thickness_mm, blender_compat=(not args.no_blender_compat),
            object_name=args.object_name, mtllib=args.mtllib, diagonal=args.diagonal,
            shear_x=args.shear_x, shear_y=args.shear_y, rot_deg=args.rot_deg,
            scale_x=args.scale_x, scale_y=args.scale_y, edge_ruffle_amp=args.edge_ruffle_amp,
            edge_ruffle_freq=args.edge_ruffle_freq, jitter_mm=args.jitter_mm,
            uv_scale_u=args.uv_scale[0], uv_scale_v=args.uv_scale[1],
            uv_offset_u=args.uv_offset[0], uv_offset_v=args.uv_offset[1],
            uv_rot_deg=args.uv_rot_deg, uv_mirror_u=args.uv_mirror_u, uv_mirror_v=args.uv_mirror_v,
            uv_noise=args.uv_noise, uv_wrap=args.uv_wrap, uv_match_like=args.uv_match_like,
        )
    elif args.cmd == "rotate":
        rotate_to_zup(args.src, dst=args.dst, thickness_mm=args.thickness_mm)
    elif args.cmd == "stats":
        obj_stats(args.src)
    elif args.cmd == "compare":
        _ = _compare(args.a, args.b, tol=args.tol, ignore_winding=args.ignore_winding)
    elif args.cmd == "make-poncho":
        write_poncho_obj(args.out, n=args.n, edge_len=args.edge, hole_radius=args.hole_radius)
    elif args.cmd == "make-skirt":
        write_skirt_obj(args.out, na=args.na, nr=args.nr, r_inner=args.r_inner, r_outer=args.r_outer,
                        flare_pow=args.flare_pow, jitter_mm=args.jitter_mm,
                        uv_tile_u=args.uv_tile[0], uv_tile_v=args.uv_tile[1])
    elif args.cmd == "make-scarf":
        write_scarf_obj(
            args.out, nx=args.nx, ny=args.ny, half_w=args.half_w, half_h=args.half_h,
            diagonal=args.diagonal,
            uv_scale_u=args.uv_scale[0], uv_scale_v=args.uv_scale[1],
            uv_offset_u=args.uv_offset[0], uv_offset_v=args.uv_offset[1],
            uv_rot_deg=args.uv_rot_deg, uv_mirror_u=args.uv_mirror_u, uv_mirror_v=args.uv_mirror_v,
            uv_noise=args.uv_noise, uv_wrap=args.uv_wrap, uv_match_like=args.uv_match_like,
            object_name=args.object_name, mtllib=args.mtllib
        )
    elif args.cmd == "make-cape":
        write_cape_obj(
            args.out, n=args.n, half=args.half, diagonal=args.diagonal,
            cut_tl=args.cut_tl, cut_tr=args.cut_tr, cut_bl=args.cut_bl, cut_br=args.cut_br,
            hem_delete_prob=args.hem_delete_prob,
            uv_scale_u=args.uv_scale[0], uv_scale_v=args.uv_scale[1],
            uv_offset_u=args.uv_offset[0], uv_offset_v=args.uv_offset[1],
            uv_rot_deg=args.uv_rot_deg, uv_mirror_u=args.uv_mirror_u, uv_mirror_v=args.uv_mirror_v,
            uv_noise=args.uv_noise, uv_wrap=args.uv_wrap, uv_match_like=args.uv_match_like,
            object_name=args.object_name, mtllib=args.mtllib
        )
    else:
        print("No subcommand given. Examples:")
        print("  python cloth_tools.py make-grid --out assets/cloth/ --n 9 --edge 1.0")
        print("  python cloth_tools.py make-poncho --out assets/cloth/ --n 33 --edge 1.0 --hole-radius 0.25")
        print("  python cloth_tools.py make-skirt --out assets/cloth/ --na 64 --nr 16 --r-inner 0.1 --r-outer 1.0")
        print("  python cloth_tools.py make-scarf --out assets/cloth/ --nx 33 --ny 17 --half-w 1.0 --half-h 0.5")
        print("  python cloth_tools.py make-cape --out assets/cloth/ --n 33 --half 1.0 --cut-tl 0.2 --cut-br 0.2")
        print("  python cloth_tools.py rotate --src assets/cloth/your.obj --thickness-mm 1.0")
        print("  python cloth_tools.py compare --a cloth_z_up.obj --b assets/cloth/mj_square_n5_v25_f32_complex0.obj")

if __name__ == "__main__":
    main()
