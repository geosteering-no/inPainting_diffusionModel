import numpy as np
import re
import os
from skimage.transform import resize
import imageio.v2 as imageio

#1.loading grdecl files and extracting porosity values and dimensions
def load_grdecl(filename):
    with open(filename, "r") as f:
        text = f.read()

    dims_match = re.search(r"SPECGRID\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if not dims_match:
        dims_match = re.search(r"GRID\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if not dims_match:
        raise ValueError("Could not find NX NY NZ in the .grdecl file")

    nx, ny, nz = map(int, dims_match.groups())

    poro_match = re.search(r"PORO\s+(.*?)\s*/", text, re.S)
    if not poro_match:
        raise ValueError("PORO block not found.")

    poro_values = np.array(poro_match.group(1).split(), dtype=float)

    if poro_values.size != nx * ny * nz:
        raise ValueError(
            f"PORO size mismatch: expected {nx*ny*nz}, got {poro_values.size}"
        )

    return poro_values, (nx, ny, nz)



#2.generated overlapping patches from each XZ slice

def generate_xz_patches(
    poro_values,
    dims,
    patch_size=(64, 256),   # height, width
    stride=(32, 128),       # vertical, horizontal stride
    npy_folder="/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/no_normalization_data/XZ_numpy_patches",
    png_folder="/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/no_normalization_data/XZ_png_patches",
    use_npz=False
):
    os.makedirs(npy_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)

    nx, ny, nz = dims
    poro_3d = poro_values.reshape(nz, ny, nx)  # (Z, Y, X)

    saved_files = []
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    patch_counter = 0

    for y in range(ny):
        xz_slice = poro_3d[:, y, :]  # (Z, X)

        for top in range(0, xz_slice.shape[0] - patch_h + 1, stride_h):
            for left in range(0, xz_slice.shape[1] - patch_w + 1, stride_w):

                patch = xz_slice[top:top+patch_h, left:left+patch_w]

                patch_resized = resize(
                    patch,
                    patch_size,
                    mode="reflect",
                    anti_aliasing=True
                )

                #saving files as npy or npz
                if use_npz:
                    npz_path = os.path.join(npy_folder, f"patch_{patch_counter}.npz")
                    np.savez_compressed(npz_path, xz=patch_resized)
                    np_file_path = npz_path
                else:
                    npy_path = os.path.join(npy_folder, f"patch_{patch_counter}.npy")
                    np.save(npy_path, patch_resized)
                    np_file_path = npy_path

                
                #saving png as grayscake
                
                png_path = os.path.join(png_folder, f"patch_{patch_counter}.png")

                # patch_uint8 = (
                #     255
                #     * (patch_resized - patch_resized.min())
                #     / (np.ptp(patch_resized) + 1e-8)
                # ).astype(np.uint8)

                # imageio.imwrite(png_path, patch_uint8)

                saved_files.append((np_file_path, png_path))
                patch_counter += 1

    print(f"Generated {patch_counter} XZ patches (each as NPY/PNG)")
    return saved_files


poro, dims = load_grdecl(
    "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/"
    "guided_diffusion_mnist/guided_diffusion/Geology/"
    "Dataset_grdecl/NOFAULT_MODEL_R1.grdecl"
)

generate_xz_patches(
    poro,
    dims,
    patch_size=(64, 256),
    stride=(32, 128),
    npy_folder="/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/no_normalization_data/XZ_numpy_patches",
    png_folder="/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/Geology/no_normalization_data/XZ_png_patches",
    use_npz=False
)
