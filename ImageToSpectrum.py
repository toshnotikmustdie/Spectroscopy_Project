"""
Spectrophotometer Analysis Pipeline
====================================
Single-beam setup: capture reference and sample in two separate sessions.

Usage:
  # Fully interactive (prompts you to capture both cuvettes live):
  python spectrophotometer_analysis.py

  # Pass pre-saved images:
  python spectrophotometer_analysis.py --reference ref*.jpg --sample sample*.jpg

  # Non-interactive ROI (fixed row fractions):
  python spectrophotometer_analysis.py --no-interactive
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from pathlib import Path


# ─────────────────────────────────────────────
# WAVELENGTH CALIBRATION
# Map leftmost pixel column → WL_MIN nm
#     rightmost pixel column → WL_MAX nm
# ─────────────────────────────────────────────
WL_MIN = 400   # nm
WL_MAX = 700   # nm


def capture_frames(label: str, n_frames=5, camera_index=0, save_dir="captures"):
    """Live capture: press SPACE to grab a frame, Q to finish early."""
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    paths = []
    print(f"  [{label}] Press SPACE to capture, Q to finish early. ({n_frames} frames needed)")
    while len(paths) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        cv2.putText(overlay, f"{label}  {len(paths)}/{n_frames}  SPACE=capture  Q=done",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f"Capture — {label}", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            path = os.path.join(save_dir, f"{label.replace(' ', '_')}_{len(paths):03d}.png")
            cv2.imwrite(path, frame)
            paths.append(path)
            print(f"    Saved {path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if not paths:
        sys.exit(f"Error: no frames captured for {label}.")
    return paths


def load_and_average_images(image_paths: list) -> np.ndarray:
    """Load images, convert to greyscale float, return per-pixel average."""
    frames = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  Warning: could not read '{p}', skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(gray)
    if not frames:
        sys.exit("Error: no images could be loaded.")
    print(f"  Loaded {len(frames)} frame(s), averaging...")
    return np.mean(np.stack(frames, axis=0), axis=0)


class ROISelector:
    """Click-and-drag to select a horizontal strip. Full image width is used."""

    def __init__(self, image: np.ndarray, title: str):
        self.image = image
        self.title = title
        self.start_y = None
        self.end_y = None
        self.drawing = False

    def _mouse_cb(self, event, x, y, flags, param):
        h, w = self.image.shape[:2]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_y = y
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_y = y
            disp = self._make_display()
            y0, y1 = sorted([self.start_y, y])
            cv2.rectangle(disp, (0, y0), (w - 1, y1), (0, 255, 0), 2)
            cv2.imshow(self.title, disp)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_y = y
            self.drawing = False

    def _make_display(self):
        norm = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def select(self) -> tuple:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self._mouse_cb)
        print(f"\n  → Drag vertically to select: {self.title}")
        print("    ENTER to confirm, R to redo.")
        while True:
            disp = self._make_display()
            if self.start_y is not None and self.end_y is not None:
                h, w = self.image.shape[:2]
                y0, y1 = sorted([self.start_y, self.end_y])
                cv2.rectangle(disp, (0, y0), (w - 1, y1), (0, 255, 0), 2)
            cv2.imshow(self.title, disp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and self.start_y is not None:
                break
            elif key == ord('r'):
                self.start_y = None
                self.end_y = None
        cv2.destroyWindow(self.title)
        y0, y1 = sorted([self.start_y, self.end_y])
        return int(y0), int(y1)


def interactive_roi_selection_single(averaged: np.ndarray) -> tuple:
    """
    Single-beam: one image contains only ONE spectrum band.
    Select the spectrum ROI and a dark region (no second beam needed).
    """
    print("\n=== ROI Selection (single-beam) ===")
    print("  Select the spectrum strip, then a dark region below it.")
    spectrum_roi = ROISelector(averaged, "Spectrum band").select()
    dark_roi     = ROISelector(averaged, "Dark region (no light)").select()
    return spectrum_roi, dark_roi


def manual_roi_selection_single(averaged: np.ndarray,
                                frac_spectrum=(0.35, 0.55),
                                frac_dark=(0.75, 0.90)) -> tuple:
    h = averaged.shape[0]
    spectrum_roi = (int(h * frac_spectrum[0]), int(h * frac_spectrum[1]))
    dark_roi     = (int(h * frac_dark[0]),     int(h * frac_dark[1]))
    print(f"  Spectrum ROI rows {spectrum_roi[0]}–{spectrum_roi[1]}")
    print(f"  Dark ROI     rows {dark_roi[0]}–{dark_roi[1]}")
    return spectrum_roi, dark_roi


def extract_roi_intensity(averaged: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """Sum pixel intensities along rows within the ROI → 1-D profile."""
    return averaged[y0:y1, :].sum(axis=0)


def pixel_to_wavelength(n_pixels: int) -> np.ndarray:
    return np.linspace(WL_MIN, WL_MAX, n_pixels)


def compute_absorbance(I: np.ndarray,
                       I0: np.ndarray,
                       dark: np.ndarray) -> np.ndarray:
    """Beer-Lambert: A = log10((I0 - dark) / (I - dark))"""
    eps = 1e-6
    I_corr  = np.maximum(I  - dark, eps)
    I0_corr = np.maximum(I0 - dark, eps)
    return np.log10(I0_corr / I_corr)


def plot_results(wavelengths, I, I0, dark_ref, dark_sample, absorbance,
                 save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Spectrophotometer Analysis (single-beam)", fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(wavelengths, I0,         color='steelblue',  lw=1.5, label='I₀ (reference/blank)')
    ax1.plot(wavelengths, I,          color='darkorange',  lw=1.5, label='I (sample)')
    ax1.plot(wavelengths, dark_ref,   color='gray',        lw=1.0, ls='--', label='Dark (reference session)')
    ax1.plot(wavelengths, dark_sample,color='lightgray',   lw=1.0, ls=':',  label='Dark (sample session)')
    ax1.set_ylabel('Summed intensity (counts)')
    ax1.legend(framealpha=0.8, fontsize=9)
    ax1.set_title('Raw intensity profiles')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(wavelengths, absorbance, color='firebrick', lw=1.8)
    ax2.axhline(0, color='black', lw=0.8, ls='--')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Absorbance (A.U.)')
    ax2.set_title('Absorption spectrum  [A = log₁₀(I₀/I)]')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {save_path}")
    plt.show()


def save_csv(wavelengths, I, I0, dark_ref, dark_sample, absorbance, path):
    data = np.column_stack([wavelengths, I0, I, dark_ref, dark_sample, absorbance])
    header = "wavelength_nm,I0_reference,I_sample,dark_reference,dark_sample,absorbance"
    np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.4f')
    print(f"  Data saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global WL_MIN, WL_MAX

    parser = argparse.ArgumentParser(
        description="Single-beam spectrophotometer: capture reference and sample separately."
    )
    parser.add_argument('--reference', nargs='+', default=None,
        help='Images of blank/water cuvette (I₀). Omit to capture live.')
    parser.add_argument('--sample', nargs='+', default=None,
        help='Images of analyte cuvette (I). Omit to capture live.')
    parser.add_argument('--camera', type=int, default=0,
        help='Camera index (default: 0).')
    parser.add_argument('--frames', type=int, default=5,
        help='Frames to capture per session (default: 5).')
    parser.add_argument('--no-interactive', action='store_true',
        help='Use automatic ROI row fractions instead of GUI selection.')
    parser.add_argument('--wl-min', type=float, default=WL_MIN,
        help=f'Wavelength at left pixel edge (default {WL_MIN} nm)')
    parser.add_argument('--wl-max', type=float, default=WL_MAX,
        help=f'Wavelength at right pixel edge (default {WL_MAX} nm)')
    parser.add_argument('--output-dir', type=str, default='.',
        help='Directory for output files.')
    args = parser.parse_args()

    WL_MIN = args.wl_min
    WL_MAX = args.wl_max
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load/capture reference (blank cuvette) ───────────────
    print("\n[1] Reference — blank/water cuvette (I₀)")
    print("    Put the BLANK cuvette in position, then press Enter to continue...")
    input("    [Enter] ")
    ref_paths = args.reference or capture_frames("reference", args.frames, args.camera,
                                                  os.path.join(args.output_dir, "reference_frames"))
    averaged_ref = load_and_average_images(ref_paths)

    # ── Step 2: Load/capture sample (analyte cuvette) ─────────────────
    print("\n[2] Sample — analyte cuvette (I)")
    print("    Swap to the SAMPLE cuvette, then press Enter to continue...")
    input("    [Enter] ")
    sample_paths = args.sample or capture_frames("sample", args.frames, args.camera,
                                                  os.path.join(args.output_dir, "sample_frames"))
    averaged_sample = load_and_average_images(sample_paths)

    # ── Step 3: ROI selection ─────────────────────────────────────────
    print("\n[3] ROI selection")
    print("  Select ROIs on the REFERENCE image first (they will be reused for sample).")
    if args.no_interactive:
        spectrum_roi, dark_roi = manual_roi_selection_single(averaged_ref)
    else:
        spectrum_roi, dark_roi = interactive_roi_selection_single(averaged_ref)

    # ── Step 4: Extract intensities ───────────────────────────────────
    print("\n[4] Extracting intensity profiles...")
    I0         = extract_roi_intensity(averaged_ref,    *spectrum_roi)
    dark_ref   = extract_roi_intensity(averaged_ref,    *dark_roi)
    I          = extract_roi_intensity(averaged_sample, *spectrum_roi)
    dark_sample= extract_roi_intensity(averaged_sample, *dark_roi)

    # Use average of both dark measurements for the correction
    dark_avg = (dark_ref + dark_sample) / 2.0

    n_pixels    = averaged_ref.shape[1]
    wavelengths = pixel_to_wavelength(n_pixels)

    # ── Step 5: Compute absorbance ────────────────────────────────────
    print("\n[5] Computing absorbance...")
    absorbance = compute_absorbance(I, I0, dark_avg)
    peak_idx   = np.argmax(absorbance)
    print(f"  Peak absorbance: {absorbance[peak_idx]:.4f} A.U. "
          f"at {wavelengths[peak_idx]:.1f} nm")

    # ── Step 6: Save & plot ───────────────────────────────────────────
    print("\n[6] Saving results...")
    base = Path(args.output_dir)
    save_csv(wavelengths, I, I0, dark_ref, dark_sample, absorbance,
             str(base / 'spectrum_data.csv'))

    print("\n[7] Plotting...")
    plot_results(wavelengths, I, I0, dark_ref, dark_sample, absorbance,
                 save_path=str(base / 'spectrum_plot.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()