# Procedural demonstration data

This small artificial scene exercises the released algorithm. It does not
support any published numerical table and does not contain measured minerals.

| File / key | Shape | Meaning |
|---|---|---|
| demo_input.npz / cube | 24 x 24 x 32 | Spatial rows, columns and bands; dimensionless values in [0,1] |
| demo_reference.npz / endmembers | 32 x 3 | Analytic component spectra |
| demo_reference.npz / abundances | 3 x 24 x 24 | Nonnegative fractions summing to one at each pixel |
| demo_reference.npz / names | 3 | demo_material_1, demo_material_2, demo_material_3; no mineral identity |

The band coordinate spans [0,1] and has no physical wavelength calibration.
Spectra are smooth analytic curves. Gaussian-filtered random fields followed by
softmax generate abundances; three 3 x 3 pure-component regions are inserted.
Linear mixing is followed by Gaussian noise with standard deviation 0.001 and
clipping to [0,1]. The generator uses NumPy seed 20260914. All numerical arrays
are float32. There are no missing values. The generator is the complete recipe.

Input and reference files are separated so that reference arrays need not be
provided to the attack. `run_demo.py` only opens the optional reference file
after attack generation. The original unperturbed reference remains the
evaluation target for both clean and attacked outputs.

Code and these project-generated demonstration arrays are distributed under
the repository MIT license. No external spectral library was used. NumPy NPZ
files can be read with `np.load(path, allow_pickle=False)`.
