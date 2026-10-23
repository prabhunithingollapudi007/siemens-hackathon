"""Make a binary mask from a CoW label file and suggest a DEMO start/goal (the two ends of the
longest centerline path). For a real task, choose start/goal anatomically in 3D Slicer instead."""
import argparse, numpy as np, networkx as nx, nibabel as nib
import cta_to_trajectory as m

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--label", required=True)
p.add_argument("--out-mask", default="vessels_binary.nii.gz")
a = p.parse_args()

lab = nib.load(a.label)
data = np.asarray(lab.dataobj)
print("label values present:", np.unique(data))
nib.save(nib.Nifti1Image((data > 0).astype(np.uint8), lab.affine), a.out_mask)

vol = m.load_volume(a.image)
mask = m.load_volume(a.out_mask).data > 0
VG = m.VascularGraph(mask, vol)
comps = sorted(VG.components(), key=len, reverse=True)
print(f"{len(comps)} skeleton components; largest has {len(comps[0])} voxels")
sub = VG.G.subgraph(comps[0])
ends = [n for n in sub if sub.degree(n) == 1]
d0 = nx.single_source_dijkstra_path_length(sub, ends[0], weight="length")
s = max(d0, key=d0.get)
d1 = nx.single_source_dijkstra_path_length(sub, s, weight="length")
g = max(d1, key=d1.get)
print(f"longest centerline path: {d1[g]:.1f} mm")
st, go = VG.xyz[s], VG.xyz[g]
print("\nRun:\npython cta_to_trajectory.py --volume", a.image, "--mask", a.out_mask,
      "--start %.1f %.1f %.1f --goal %.1f %.1f %.1f" % (*st, *go),
      "--device-diameter-mm 0.89 --algorithm device-aware --output nav_out")