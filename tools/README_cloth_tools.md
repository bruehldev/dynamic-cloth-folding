# Cloth Tools & DR Batch — Quick Guide

Two entry points:

- **`cloth_tools.py`** — single-shape generators & utilities.
- **`cloth_dr_batch.py`** — batch generator for Domain Randomization (DR) with one central `CONFIG` block.

---

## 1) `cloth_tools.py`

### Core generators

**Grid (square sheet)**

```bash
python tools/cloth_tools.py make-grid --out assets/cloth/   --n 9 --edge 1.0   [--diagonal A|B|checker|row-alt|col-alt]   [--shear-x Sx --shear-y Sy --rot-deg R --scale-x Sx --scale-y Sy]   [--edge-ruffle-amp AMP --edge-ruffle-freq FREQ]   [--jitter-mm MM]   [--uv-scale US VS --uv-offset UOFF VOFF]   [--uv-rot-deg DEG --uv-mirror-u --uv-mirror-v --uv-noise EPS --uv-wrap]   [--uv-match-like ref.obj]   [--object-name NAME --mtl file.mtl]
```

- Coords: **[-edge, +edge]** in X/Y (Z=0).
- Uniform triangulation (`A` by default), **-Z normals**, per-vertex **vt** + single **vn**.

**Poncho (square w/ circular head hole)**

```bash
python tools/cloth_tools.py make-poncho --out assets/cloth/   --n 33 --edge 1.0 --hole-radius 0.25   [--object-name NAME --mtl file.mtl]
```

Removes faces fully inside the hole and **reindexes** vertices/UVs.

**Skirt (polar annulus)**

```bash
python tools/cloth_tools.py make-skirt --out assets/cloth/   --na 64 --nr 16 --r-inner 0.10 --r-outer 1.00 --flare-pow 1.2   [--jitter-mm 2] [--uv-tile 1 2]   [--object-name NAME --mtl file.mtl]
```

_(If enabled in your file)_ **Scarf / Cape**

```bash
# Scarf (rectangle)
python tools/cloth_tools.py make-scarf --out assets/cloth/   --nx 41 --ny 21 --half-w 1.2 --half-h 0.6 --diagonal A   [same UV flags as grid] [--object-name NAME --mtl file.mtl]

# Cape (square with corner cutouts & irregular hem)
python tools/cloth_tools.py make-cape --out assets/cloth/   --n 33 --half 1.1 --diagonal checker   --cut-tl 0.2 --cut-tr 0.15 --cut-bl 0.0 --cut-br 0.0   --hem-delete-prob 0.08   [same UV flags as grid] [--object-name NAME --mtl file.mtl]
```

### Utilities

```bash
# Rotate OBJ to Z-up & compress thickness
python tools/cloth_tools.py rotate --src path/to.obj --thickness-mm 1.0 [--dst out.obj]

# Mesh stats (n/verts/faces/complex)
python tools/cloth_tools.py stats --src path/to.obj

# Compare two OBJs
#   strict (ordered triples & winding)
python tools/cloth_tools.py compare --a A.obj --b B.obj
#   geometric/topological equivalence (ignoring winding)
python tools/cloth_tools.py compare --a A.obj --b B.obj --ignore-winding
```

---

## 2) `cloth_dr_batch.py`

Batch driver that centralizes **all DR ranges** in a single `CONFIG` dictionary at the top.

### Usage

```bash
# Generate N of each enabled type into per-type folders
python tools/cloth_dr_batch.py --out assets/cloth/dr --count 5 --seed 7

# Optional experiment/run subfolder
python tools/cloth_dr_batch.py --out assets/cloth/dr --count 5 --seed 7 --prefix expA
# → outputs: assets/cloth/dr/expA/<type>/<type>_<timestamp>_<index>.obj
```

### Behavior

- Types: **grid, poncho, skirt** (+ optional **scarf, cape** if present).
- Output per type: `.../<type>/`.
- Filename: `<type>_<timestamp>_<index>.obj` (object name matches file basename).
- If `CONFIG["mtllib"]` is set, each OBJ includes `mtllib` + default `usemtl None`.

### Centralized ranges

Edit `CONFIG` to control everything:

- `type_weights` (which types to include / relative proportion)
- **Grid**: `n`, `edge`, diagonal pattern, shear/rot/scale, ruffle, jitter, UV scale/offset (and extended UV transforms if enabled)
- **Poncho**: `n`, `edge`, `hole_radius`
- **Skirt**: `na`, `nr`, radii, `flare_pow`, jitter, UV tiling
- _(Optional)_ **Scarf/Cape**: dimensions, cuts, irregular hem, UV DR

---

## Tips

- For Blender-like UV ranges, use `--uv-match-like path/to/cloth_z_up.obj`, then apply your own `--uv-scale/--uv-offset/--uv-rot-deg`.
- Keep `--diagonal A` for stability unless you’re randomizing triangulation by design.
- Use `compare --ignore-winding` to confirm “same mesh” in practice; strict compare is for byte-level diffs.
- All generators emit **-Z normals** and consistent triangulation to avoid artifacts.
