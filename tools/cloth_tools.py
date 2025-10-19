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

# --------------------- grid generator ---------------------
def write_grid_obj(path, n=9, edge_len=1.0, z=0.0, thickness_mm=0.0,
                   blender_compat=True, object_name=None, mtllib=None):
    """
    Generate n×n grid (single triangulated sheet).

    If blender_compat=True (default):
      - Coordinates span [-edge_len, +edge_len] in X and Y (edge_len is *half-extent*).
      - Uniform diagonal split (v00--v11) for all quads (no checkerboard).
      - Face winding is CLOCKWISE when viewed from +Z so that normals point to -Z,
        matching Blender's default plane triangulation.
      - One shared vertex normal (0,0,-1) and a 0..1 UV grid are emitted.
      - 'usemtl None' and 's 1' are written for compatibility with cloth_z_up.obj.

    If blender_compat=False (legacy behavior):
      - Coordinates span [-edge_len/2, +edge_len/2].
      - Checkerboard diagonal split with CCW winding (normals +Z).
      - No vt/vn lines are emitted.
    """
    verts_count = n * n
    faces_count = 2 * (n - 1) * (n - 1)
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

    verts = []
    faces = []
    vt = None
    vn = None
    extras = None

    if blender_compat:
        # Half-extent coordinates to match cloth_z_up.obj scale when edge_len=1.0
        half = edge_len
        step = (2.0 * edge_len) / (n - 1)

        for j in range(n):
            y = -half + j * step
            for i in range(n):
                x = -half + i * step
                verts.append([x, y, z])

        # UVs 0..1 (u increases with x, v decreases with y to match -Z winding)
        vt = []
        for j in range(n):
            vcoord = 1.0 - (j / (n - 1))
            for i in range(n):
                u = i / (n - 1)
                vt.append([u, vcoord])

        # Single normal pointing -Z
        vn = np.array([[0.0, 0.0, -1.0]], float)

        def vid(i, j):  # 1-based
            return j * n + i + 1

        # Uniform 'A' diagonal (v00--v11) with CLOCKWISE order -> -Z normals
        f_lines = []
        for j in range(n - 1):
            for i in range(n - 1):
                v00 = vid(i, j)
                v10 = vid(i + 1, j)
                v01 = vid(i, j + 1)
                v11 = vid(i + 1, j + 1)
                f_lines.append(f"f {v00}/{v00}/1 {v11}/{v11}/1 {v10}/{v10}/1")
                f_lines.append(f"f {v00}/{v00}/1 {v01}/{v01}/1 {v11}/{v11}/1")
        faces = f_lines

        # Optional extras to mirror Blender export style
        extras = []
        if mtllib:
            extras.append(f"mtllib {mtllib}")
        if object_name:
            extras.append(f"o {object_name}")
        extras += ["usemtl None", "s 1"]

        # thickness_mm ignored in blender-compat mode (mesh is perfectly flat)
        hz = 0.0
    else:
        # Legacy mode (previous behavior)
        step = edge_len / (n - 1)
        half = edge_len / 2.0
        # half-thickness in meters
        hz = max(0.0, float(thickness_mm)) * 1e-3 * 0.5

        for j in range(n):
            y = -half + j * step
            for i in range(n):
                x = -half + i * step
                if hz > 0.0:
                    # Checkerboard: alternate +/- hz to create a tiny z-span
                    z_off = hz if ((i + j) % 2 == 0) else -hz
                    verts.append([x, y, z + z_off])
                else:
                    verts.append([x, y, z])

        def vid(i, j):  # 1-based for OBJ
            return j * n + i + 1

        for j in range(n - 1):
            for i in range(n - 1):
                v00 = vid(i, j)
                v10 = vid(i + 1, j)
                v01 = vid(i, j + 1)
                v11 = vid(i + 1, j + 1)

                if (i + j) % 2 == 0:
                    # split v00--v11, CCW (+Z)
                    faces.append((v00, v10, v11))
                    faces.append((v00, v11, v01))
                else:
                    # split v10--v01, CCW (+Z)
                    faces.append((v00, v10, v01))
                    faces.append((v10, v11, v01))

        vt = None
        vn = None
        extras = None

    V = np.array(verts, float)
    header = "# mj_match_square.obj (blender-compat)\n" if blender_compat else "# mj_match_square.obj\n"
    write_obj(out_path, V, faces, header=header, vt=vt, vn=vn, extras=extras)
    print(f"Wrote {out_path} (verts={verts_count}, faces={faces_count}, complexity={comp}, blender_compat={blender_compat})")
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

# --------------------- CLI ---------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Cloth mesh tools: make grid or rotate OBJ to Z-up.")
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("make-grid", help="Generate an n×n grid OBJ with complexity in filename.")
    g.add_argument("--out", required=True, help="Output path OR directory OR stem.")
    g.add_argument("--n", type=int, default=9, help="Vertices per side (default 9).")
    g.add_argument("--edge", type=float, default=1.0, help="Half-extent in meters in blender-compat mode; full width in legacy mode.")
    g.add_argument("--z", type=float, default=0.0, help="Z height (default 0.0).")
    g.add_argument("--thickness-mm", type=float, default=0.0,
                   help="Target total z-span in millimeters (legacy mode only; blender-compat ignores this).")
    g.add_argument("--no-blender-compat", action="store_true",
                   help="Disable blender-style topology/scale/winding/uv and use legacy generator.")
    g.add_argument("--object-name", default=None,
                   help="Write an 'o <name>' line before faces.")
    g.add_argument("--mtl", dest="mtllib", default=None,
                   help="Write an 'mtllib <file>' line before faces.")

    r = sub.add_parser("rotate", help="Rotate an OBJ to Z-up and compress thickness.")
    r.add_argument("--src", required=True, help="Source OBJ.")
    r.add_argument("--dst", default=None, help="Destination OBJ (auto-named if omitted).")
    r.add_argument("--thickness-mm", type=float, default=1.0,
                   help="Target total z-span in millimeters (default 1.0).")

    s = sub.add_parser("stats", help="Print n/verts/faces/complex for an OBJ.")
    s.add_argument("--src", required=True, help="Source OBJ to analyze.")

    c = sub.add_parser("compare", help="Compare two OBJs for vertex/face equality (ignoring winding).")
    c.add_argument("--a", required=True, help="First OBJ.")
    c.add_argument("--b", required=True, help="Second OBJ.")
    c.add_argument("--tol", type=float, default=1e-6, help="Position tolerance (default 1e-6).")
    c.add_argument("--ignore-winding", action="store_true", help="Ignore vertex winding when comparing.")

    return ap.parse_args()

def main():
    args = parse_args()
    if args.cmd == "make-grid":
        write_grid_obj(args.out, n=args.n, edge_len=args.edge, z=args.z, thickness_mm=args.thickness_mm, blender_compat=(not args.no_blender_compat), object_name=args.object_name, mtllib=args.mtllib)
    elif args.cmd == "rotate":
        rotate_to_zup(args.src, dst=args.dst, thickness_mm=args.thickness_mm)
    elif args.cmd == "stats":
        obj_stats(args.src)
    elif args.cmd == "compare":
        # run compare; function prints a summary and returns boolean
        _ = _compare(args.a, args.b, tol=args.tol, ignore_winding=args.ignore_winding)
    else:
        print("No subcommand given. Examples:")
        print("  python cloth_tools.py make-grid --out assets/cloth/ --n 9 --edge 1.0")
        print("  python cloth_tools.py rotate --src assets/cloth/your.obj --thickness-mm 1.0")
        print("  python cloth_tools.py stats --src assets/cloth/your.obj")
        print("  python cloth_tools.py compare --a cloth_z_up.obj --b assets/cloth/mj_square_n5_v25_f32_complex0.obj")

if __name__ == "__main__":
    main()
