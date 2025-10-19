# cloth_tools.py
import os, sys, math, argparse
import numpy as np

# --------------------- I/O ---------------------
def read_obj(path):
    V, F = [], []
    with open(path, "r") as f:
        for ln in f:
            if ln.startswith("v "):
                _, x, y, z, *rest = ln.strip().split()
                V.append([float(x), float(y), float(z)])
            elif ln.startswith("f "):
                F.append(ln.strip())
    return (np.array(V, float), F)

def write_obj(path, V, F, header="# generated obj\n"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        for v in V:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
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
def write_grid_obj(path, n=9, edge_len=1.0, z=0.0, thickness_mm=0.0):
    """
    Generate n×n grid (single triangulated sheet). If thickness_mm>0,
    apply a tiny checkerboard z-offset so z-span ≈ thickness_mm.
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
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    V = np.array(verts, float)
    write_obj(out_path, V, faces, header="# mj_match_square.obj\n")
    print(f"Wrote {out_path} (verts={verts_count}, faces={faces_count}, complexity={comp}, thickness≈{2*hz:.6f} m)")
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
    """Rotate any OBJ to Z-up and compress thickness to ~thickness_mm (single surface)."""
    V, F = read_obj(src)
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
    write_obj(out_path, V2, F, header="# z-up flat export\n")

    size_est = np.linalg.norm(V2.max(0) - V2.min(0))
    print(f"Wrote: {out_path} (verts={nverts}, faces={nfaces}, size≈{size_est:.3f} m, complexity={comp}, thickness≈{float(thickness_mm)*1e-3:.6f} m)")
    return out_path

# --------------------- CLI ---------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Cloth mesh tools: make grid or rotate OBJ to Z-up.")
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("make-grid", help="Generate an n×n grid OBJ with complexity in filename.")
    g.add_argument("--out", required=True, help="Output path OR directory OR stem.")
    g.add_argument("--n", type=int, default=9, help="Vertices per side (default 9).")
    g.add_argument("--edge", type=float, default=1.0, help="Edge length in meters (default 1.0).")
    g.add_argument("--z", type=float, default=0.0, help="Z height (default 0.0).")
    g.add_argument("--thickness-mm", type=float, default=0.0,
                   help="Target total z-span in millimeters (default 0: perfectly flat).")

    r = sub.add_parser("rotate", help="Rotate an OBJ to Z-up and compress thickness.")
    r.add_argument("--src", required=True, help="Source OBJ.")
    r.add_argument("--dst", default=None, help="Destination OBJ (auto-named if omitted).")
    r.add_argument("--thickness-mm", type=float, default=1.0,
                   help="Target total z-span in millimeters (default 1.0).")

    return ap.parse_args()

def main():
    args = parse_args()
    if args.cmd == "make-grid":
        write_grid_obj(args.out, n=args.n, edge_len=args.edge, z=args.z, thickness_mm=args.thickness_mm)
    elif args.cmd == "rotate":
        rotate_to_zup(args.src, dst=args.dst, thickness_mm=args.thickness_mm)
    else:
        print("No subcommand given. Examples:")
        print("  python cloth_tools.py make-grid --out assets/cloth/ --n 9 --edge 1.0 --thickness-mm 0.5")
        print("  python cloth_tools.py rotate --src assets/cloth/your.obj --thickness-mm 1.0")

if __name__ == "__main__":
    main()
