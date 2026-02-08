"""
Auto Decision Cloud Mask Viewer
--------------------------------
Displays MSG channels 1, 2 (radiances) plus 7, 9, 10 (brightness temperatures)
and automatically derives per-channel thresholds using the Bayesian GMM decision
boundary logic from histogram_testing/bayesian_gaussian_gmm.py. A single button
runs the fitting process for the current zone/box and overlays a combined cloud
mask on top of the selected visible band. Useful when you want data-driven
thresholds instead of hand tuning.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button, CheckButtons, RadioButtons
from matplotlib.colors import ListedColormap
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm
from scipy.signal import argrelextrema

BASE_DIR = 'data'

CHANNEL_CONFIGS = [
    {
        'id': 1,
        'name': 'Ch1 0.6μm (VIS)',
        'short': 'Ch1',
        'type': 'radiance',
        'unit': 'Radiance',
        'cmap': 'Greys_r',
        'percentiles': (1, 99),
        'comparison': '>=',
        'policy': 'first',
    },
    {
        'id': 2,
        'name': 'Ch2 0.8μm (NIR)',
        'short': 'Ch2',
        'type': 'radiance',
        'unit': 'Radiance',
        'cmap': 'bone_r',
        'percentiles': (1, 99),
        'comparison': '>=',
        'policy': 'first',
    },
    {
        'id': 7,
        'name': 'Ch7 8.7μm (BT)',
        'short': 'Ch7',
        'type': 'bt',
        'unit': 'Brightness Temp [K]',
        'cmap': 'inferno',
        'range': (190, 320),
        'comparison': '<=',
        'policy': 'last',
        'wavelength': 8.7,
    },
    {
        'id': 9,
        'name': 'Ch9 10.8μm (BT)',
        'short': 'Ch9',
        'type': 'bt',
        'unit': 'Brightness Temp [K]',
        'cmap': 'plasma',
        'range': (200, 320),
        'comparison': '<=',
        'policy': 'last',
        'wavelength': 10.79,
    },
    {
        'id': 10,
        'name': 'Ch10 11.94μm (BT)',
        'short': 'Ch10',
        'type': 'bt',
        'unit': 'Brightness Temp [K]',
        'cmap': 'viridis',
        'range': (200, 320),
        'comparison': '<=',
        'policy': 'last',
        'wavelength': 11.94,
    },
]

CONFIG_LOOKUP = {cfg['id']: cfg for cfg in CHANNEL_CONFIGS}


class AutoDecisionCloudMaskViewer:
    def __init__(self):
        self.current_zone = 12
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100
        self.overlay_channel = 1
        self.sigma_factor = 0.5

        self.images: dict[int, dict[str, np.ndarray | int | str]] = {}
        self.colorbars = []
        self.hist_buttons = []
        self.channel_display_data: dict[int, dict[str, object]] = {}
        self.per_channel_masks: dict[int, np.ndarray | None] = {}
        self.thresholds = {cfg['id']: None for cfg in CHANNEL_CONFIGS}
        self.mask_enabled = {cfg['id']: True for cfg in CHANNEL_CONFIGS}
        self.loading_ax = None

        self.mask_cmap = ListedColormap([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.75],
        ])

        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(
            3,
            3,
            left=0.04,
            right=0.99,
            top=0.96,
            bottom=0.25,
            hspace=0.32,
            wspace=0.18,
        )

        self.axes_map = {}
        axis_order = {
            1: (0, 0),
            2: (0, 1),
            7: (0, 2),
            9: (1, 0),
            10: (1, 1),
        }
        for channel_id, loc in axis_order.items():
            ax = self.fig.add_subplot(gs[loc])
            ax.axis('off')
            self.axes_map[channel_id] = ax

        self.mask_ax = self.fig.add_subplot(gs[1, 2])
        self.mask_ax.axis('off')
        self.overlay_ax = self.fig.add_subplot(gs[2, :])
        self.overlay_ax.axis('off')

        # Controls ---------------------------------------------------------
        ax_zone = plt.axes([0.10, 0.18, 0.80, 0.025])
        self.zone_slider = Slider(ax_zone, 'Zone', 0, 24, valinit=self.current_zone, valstep=1)
        self.zone_slider.on_changed(self.update_zone)

        ax_x = plt.axes([0.10, 0.12, 0.12, 0.032])
        self.x_box = TextBox(ax_x, 'X Center', initial=str(self.x_center))
        self.x_box.on_submit(self.update_x)

        ax_y = plt.axes([0.26, 0.12, 0.12, 0.032])
        self.y_box = TextBox(ax_y, 'Y Center', initial=str(self.y_center))
        self.y_box.on_submit(self.update_y)

        ax_box = plt.axes([0.42, 0.12, 0.12, 0.032])
        self.box_box = TextBox(ax_box, 'Box Size', initial=str(self.box_size))
        self.box_box.on_submit(self.update_box_size)

        ax_sigma = plt.axes([0.58, 0.12, 0.12, 0.032])
        self.sigma_box = TextBox(ax_sigma, 'σ Factor', initial=str(self.sigma_factor))
        self.sigma_box.on_submit(self.update_sigma_factor)

        ax_reset = plt.axes([0.74, 0.12, 0.16, 0.032])
        self.reset_button = Button(ax_reset, 'Reset View')
        self.reset_button.on_clicked(self.reset_view)

        ax_generate = plt.axes([0.10, 0.06, 0.32, 0.045])
        self.generate_button = Button(ax_generate, 'Generate Thresholds', color='#0077b6', hovercolor='#005082')
        self.generate_button.label.set_color('white')
        self.generate_button.on_clicked(self.generate_thresholds)

        ax_mask_check = plt.axes([0.45, 0.04, 0.16, 0.10])
        labels = [cfg['short'] for cfg in CHANNEL_CONFIGS]
        actives = [self.mask_enabled[cfg['id']] for cfg in CHANNEL_CONFIGS]
        self.label_to_channel = {cfg['short']: cfg['id'] for cfg in CHANNEL_CONFIGS}
        self.mask_checks = CheckButtons(ax_mask_check, labels, actives)
        self.mask_checks.on_clicked(self.toggle_mask_channel)
        ax_mask_check.set_title('Mask Inputs', fontsize=9)

        ax_overlay_radio = plt.axes([0.64, 0.04, 0.14, 0.10])
        self.overlay_radio = RadioButtons(ax_overlay_radio, ('Channel 1', 'Channel 2'), active=0)
        self.overlay_radio.on_clicked(self.set_overlay_channel)
        ax_overlay_radio.set_title('Overlay Base', fontsize=9)

        ax_status = plt.axes([0.80, 0.04, 0.18, 0.10])
        ax_status.axis('off')
        self.status_text = ax_status.text(0.0, 0.7, 'Status: Awaiting generation', fontsize=9, wrap=True)

        self.load_zone(self.current_zone)
        self.update_display()

    # ------------------------------------------------------------------
    def load_zone(self, zone: int) -> None:
        print(f"\nLoading zone {zone:02d}...")
        self.images = {}
        for cfg in CHANNEL_CONFIGS:
            channel = cfg['id']
            img_path = f"{BASE_DIR}/channel_{channel}/msg_c{channel:02d}_z{zone:02d}.img"
            try:
                self.images[channel] = self.load_image(img_path)
                meta = self.images[channel]
                print(f"  Ch{channel}: ✓ ({meta['nx']}x{meta['ny']})")
            except Exception as exc:  # noqa: BLE001
                print(f"  Ch{channel}: ✗ ({exc})")
                self.images[channel] = None
        loaded = len([img for img in self.images.values() if img is not None])
        print(f"Loaded {loaded}/{len(CHANNEL_CONFIGS)} channels")
        self.reset_thresholds()

    def reset_thresholds(self) -> None:
        self.thresholds = {cfg['id']: None for cfg in CHANNEL_CONFIGS}
        self.status_text.set_text('Status: Generate thresholds for current view')

    def load_image(self, filepath: str) -> dict[str, np.ndarray | int | str]:
        with open(filepath, 'r', encoding='utf-8') as f:
            title = f.readline().strip()
            nx, ny = [int(x) for x in f.readline().split()]
            imgdata = []
            for line in f:
                line = line.strip()
                if line:
                    imgdata.extend([float(x) for x in line.split()])
            data = np.array(imgdata).reshape((ny, nx))
        return {'data': data, 'nx': nx, 'ny': ny, 'title': title}

    def brightness_temperature(self, radiance: np.ndarray, wavelength: float) -> np.ndarray:
        radiance = np.clip(radiance, 1e-10, None)
        H = 6.63e-34
        C = 3.00e8
        K = 1.38e-23
        R1 = H * C / K
        R2 = 2 * H * C**2
        w = wavelength * 1.0e-6
        temp = R1 / w / np.log(1.0 + R2 / (w**5 * radiance * 1e6))
        return temp

    def get_zoom_region(self, data: np.ndarray) -> np.ndarray:
        ny, nx = data.shape
        half_box = self.box_size // 2
        x_min = max(0, self.x_center - half_box)
        x_max = min(nx, self.x_center + half_box)
        y_min = max(0, self.y_center - half_box)
        y_max = min(ny, self.y_center + half_box)
        return data[y_min:y_max, x_min:x_max]

    def compute_scale(self, data: np.ndarray, cfg: dict) -> tuple[float | None, float | None]:
        if not data.size:
            return None, None
        if cfg.get('range') is not None:
            return tuple(float(v) for v in cfg['range'])
        percentiles = cfg.get('percentiles')
        if percentiles is None:
            return None, None
        vmin = np.percentile(data, percentiles[0])
        vmax = np.percentile(data, percentiles[1])
        if vmin == vmax:
            vmax = vmin + 1e-6
        return float(vmin), float(vmax)

    def fit_bgm(self, data: np.ndarray) -> BayesianGaussianMixture:
        arr = np.asarray(data).reshape(-1, 1)
        bgm = BayesianGaussianMixture(
            n_components=4,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1e-2,
            mean_precision_prior=0.8,
            max_iter=1000,
            n_init=5,
            init_params='kmeans',
            random_state=42,
        )
        bgm.fit(arr)
        return bgm

    def compute_decision_boundary(self, x_grid: np.ndarray, total_pdf: np.ndarray,
                                  means: np.ndarray, stds: np.ndarray,
                                  policy: str, max_idx: np.ndarray,
                                  min_idx: np.ndarray) -> float | None:
        if min_idx.size == 0:
            return None
        maxima_points = np.sort(x_grid[max_idx]) if max_idx.size else np.array([])
        minima_points = np.sort(x_grid[min_idx])
        if policy == 'last':
            maxima_points = maxima_points[::-1]
            minima_points = minima_points[::-1]
        first_min = minima_points[0] if minima_points.size else None
        if first_min is None:
            return None
        second_min = minima_points[1] if minima_points.size > 1 else None
        second_max = maxima_points[1] if maxima_points.size > 1 else None

        sorted_indices = np.argsort(means)
        if len(sorted_indices) < 2:
            sigma_second = None
        else:
            if policy == 'first':
                second_idx = sorted_indices[1]
            else:
                second_idx = sorted_indices[-2]
            sigma_second = stds[second_idx]

        use_second_min = False
        if (sigma_second is not None) and (second_max is not None) and (sigma_second > 0):
            distance = abs(first_min - second_max)
            if distance <= self.sigma_factor * sigma_second:
                use_second_min = second_min is not None

        boundary = second_min if use_second_min else first_min
        midpoint = 0.5 * (np.min(means) + np.max(means))
        if policy == 'first':
            boundary = min(boundary, midpoint)
        else:
            boundary = max(boundary, midpoint)
        return boundary

    def compute_threshold_for_channel(self, channel_id: int) -> float | None:
        data_info = self.channel_display_data.get(channel_id)
        if data_info is None:
            return None
        data = np.array(data_info['data']).astype(float)
        data = data[np.isfinite(data)]
        if data.size < 20:
            return None
        bgm = self.fit_bgm(data)
        x_grid = np.linspace(np.min(data), np.max(data), 1200)
        weights = bgm.weights_
        means = bgm.means_.ravel()
        covs = np.array([
            bgm.covariances_[k].reshape(-1).item() if bgm.covariances_.ndim == 3 else bgm.covariances_[k]
            for k in range(bgm.n_components)
        ])
        stds = np.sqrt(covs)
        comp_pdfs = []
        for k in range(bgm.n_components):
            pdf_k = weights[k] * norm.pdf(x_grid, loc=means[k], scale=stds[k])
            comp_pdfs.append(pdf_k)
        comp_pdfs = np.vstack(comp_pdfs)
        total_pdf = comp_pdfs.sum(axis=0)
        max_idx = argrelextrema(total_pdf, np.greater)[0]
        min_idx = argrelextrema(total_pdf, np.less)[0]
        cfg = CONFIG_LOOKUP[channel_id]
        boundary = self.compute_decision_boundary(x_grid, total_pdf, means, stds,
                                                  cfg['policy'], max_idx, min_idx)
        if boundary is None:
            boundary = float(np.median(data))
        return float(boundary)

    def update_display(self) -> None:
        for cbar in self.colorbars:
            try:
                cbar.remove()
            except (KeyError, ValueError):
                pass
        self.colorbars = []
        for btn in self.hist_buttons:
            try:
                btn.ax.remove()
            except Exception:  # noqa: BLE001
                pass
        self.hist_buttons = []
        self.channel_display_data = {}
        self.per_channel_masks = {}

        for cfg in CHANNEL_CONFIGS:
            ax = self.axes_map[cfg['id']]
            ax.clear()
            ax.axis('off')
            image = self.images.get(cfg['id'])
            if image is None:
                ax.text(0.5, 0.5, f"{cfg['short']}\nNot Found", ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                continue

            raw_data = image['data']  # type: ignore[index]
            if cfg['type'] == 'bt':
                wavelength = cfg.get('wavelength')
                raw_data = self.brightness_temperature(raw_data, wavelength)  # type: ignore[arg-type]
            zoom_data = self.get_zoom_region(raw_data)
            vmin, vmax = self.compute_scale(zoom_data, cfg)
            im = ax.imshow(zoom_data, origin='lower', cmap=cfg.get('cmap', 'gray'),
                           vmin=vmin, vmax=vmax)
            threshold = self.thresholds.get(cfg['id'])
            threshold_text = f"Threshold: {threshold:.2f}" if threshold is not None else 'Threshold: --'
            ax.set_title(f"{cfg['name']}\n{threshold_text}", fontsize=8, pad=2)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
            cbar.set_label(cfg['unit'], fontsize=6)
            self.colorbars.append(cbar)

            bbox = ax.get_position()
            button_ax = plt.axes([
                bbox.x0 + bbox.width * 0.15,
                bbox.y0 - 0.040,
                bbox.width * 0.7,
                0.020,
            ])
            hist_btn = Button(button_ax, 'Histogram', color='#87CEEB', hovercolor='#4682B4')
            hist_btn.label.set_fontsize(8)
            hist_btn.on_clicked(lambda _event, ch=cfg['id']: self.show_histogram(ch))
            self.hist_buttons.append(hist_btn)

            self.channel_display_data[cfg['id']] = {
                'data': zoom_data,
                'label': cfg['name'],
                'unit': cfg['unit'],
                'type': cfg['type'],
            }

            if threshold is not None:
                comparator = cfg['comparison']
                if comparator == '>=':
                    mask = zoom_data >= threshold
                else:
                    mask = zoom_data <= threshold
                self.per_channel_masks[cfg['id']] = mask
            else:
                self.per_channel_masks[cfg['id']] = None

        combined_mask = self.compute_combined_mask()
        self.display_mask(combined_mask)
        self.display_overlay(combined_mask)
        self.fig.suptitle(
            f'Auto Decision Cloud Mask — Zone {self.current_zone:02d} | '
            f'Center ({self.x_center}, {self.y_center}) px | '
            f'Box {self.box_size}px',
            fontsize=11,
            fontweight='bold',
        )
        self.fig.canvas.draw_idle()

    def compute_combined_mask(self) -> np.ndarray | None:
        active_masks = [
            mask for ch, mask in self.per_channel_masks.items()
            if self.mask_enabled.get(ch) and mask is not None
        ]
        if not active_masks:
            return None
        combined = np.ones_like(active_masks[0], dtype=bool)
        for mask in active_masks:
            combined = np.logical_and(combined, mask)
        return combined

    def display_mask(self, combined_mask: np.ndarray | None) -> None:
        self.mask_ax.clear()
        self.mask_ax.axis('off')
        if combined_mask is None:
            self.mask_ax.text(0.5, 0.5, 'Mask pending thresholds', ha='center', va='center',
                              transform=self.mask_ax.transAxes, fontsize=10)
            return
        mask_int = combined_mask.astype(int)
        im = self.mask_ax.imshow(mask_int, origin='lower', cmap=self.mask_cmap,
                                 vmin=0, vmax=1)
        self.mask_ax.set_title('Combined Cloud Mask', fontsize=9, pad=2)
        cbar = plt.colorbar(im, ax=self.mask_ax, fraction=0.046, pad=0.02, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Clear', 'Cloud'])
        self.colorbars.append(cbar)

    def display_overlay(self, combined_mask: np.ndarray | None) -> None:
        self.overlay_ax.clear()
        self.overlay_ax.axis('off')
        base_data_info = self.channel_display_data.get(self.overlay_channel)
        if base_data_info is None:
            self.overlay_ax.text(0.5, 0.5, 'Base channel not loaded', ha='center', va='center',
                                 transform=self.overlay_ax.transAxes, fontsize=10)
            return
        base_data = base_data_info['data']  # type: ignore[assignment]
        base_cfg = CONFIG_LOOKUP[self.overlay_channel]
        vmin, vmax = self.compute_scale(base_data, base_cfg)
        self.overlay_ax.imshow(base_data, origin='lower', cmap=base_cfg.get('cmap', 'gray'),
                               vmin=vmin, vmax=vmax)
        if combined_mask is not None:
            overlay_rgba = np.zeros((*combined_mask.shape, 4))
            overlay_rgba[combined_mask] = [1, 0, 0, 0.45]
            self.overlay_ax.imshow(overlay_rgba, origin='lower')
            mask_pct = 100 * combined_mask.mean()
            subtitle = f'Cloud-covered pixels: {mask_pct:.1f}%'
        else:
            subtitle = 'Generate thresholds to view mask'
        self.overlay_ax.set_title(
            f'Overlay on {base_cfg["name"]}\n{subtitle}',
            fontsize=10,
            pad=4,
        )

    def show_loading(self, message: str) -> None:
        if self.loading_ax is None:
            self.loading_ax = self.fig.add_axes([0.25, 0.40, 0.5, 0.2])
        self.loading_ax.clear()
        self.loading_ax.set_facecolor('#f0f0f0')
        self.loading_ax.axis('off')
        self.loading_ax.text(0.5, 0.65, 'Generating thresholds...', ha='center', va='center',
                              fontsize=14, fontweight='bold')
        self.loading_ax.text(0.5, 0.35, message, ha='center', va='center', fontsize=11)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def hide_loading(self) -> None:
        if self.loading_ax is not None:
            self.loading_ax.remove()
            self.loading_ax = None
            self.fig.canvas.draw_idle()

    # --------------------------- Actions --------------------------------
    def generate_thresholds(self, _event) -> None:
        if not self.channel_display_data:
            self.status_text.set_text('Status: No data loaded')
            return
        self.show_loading('Starting...')
        updated = {}
        for cfg in CHANNEL_CONFIGS:
            message = f'Processing {cfg["short"]}'
            self.show_loading(message)
            threshold = self.compute_threshold_for_channel(cfg['id'])
            if threshold is not None:
                updated[cfg['id']] = threshold
        self.hide_loading()
        if updated:
            self.thresholds.update(updated)
            self.status_text.set_text('Status: Thresholds generated')
        else:
            self.status_text.set_text('Status: Could not derive thresholds')
        self.update_display()

    def update_zone(self, _val: float) -> None:
        new_zone = int(self.zone_slider.val)
        if new_zone != self.current_zone:
            self.current_zone = new_zone
            self.load_zone(self.current_zone)
            self.update_display()

    def update_x(self, text: str) -> None:
        try:
            self.x_center = int(text)
            self.reset_thresholds()
            self.update_display()
        except ValueError:
            print(f"Invalid X center: {text}")

    def update_y(self, text: str) -> None:
        try:
            self.y_center = int(text)
            self.reset_thresholds()
            self.update_display()
        except ValueError:
            print(f"Invalid Y center: {text}")

    def update_box_size(self, text: str) -> None:
        try:
            self.box_size = max(10, int(text))
            self.reset_thresholds()
            self.update_display()
        except ValueError:
            print(f"Invalid box size: {text}")

    def update_sigma_factor(self, text: str) -> None:
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
            self.sigma_factor = value
            self.status_text.set_text(f'Status: σ factor set to {value:.2f}. Regenerate thresholds.')
        except ValueError:
            print(f"Invalid sigma factor: {text}")
            self.sigma_box.set_val(str(self.sigma_factor))

    def reset_view(self, _event) -> None:
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100
        self.x_box.set_val(str(self.x_center))
        self.y_box.set_val(str(self.y_center))
        self.box_box.set_val(str(self.box_size))
        self.reset_thresholds()
        self.update_display()

    def toggle_mask_channel(self, label: str) -> None:
        channel = self.label_to_channel[label]
        self.mask_enabled[channel] = not self.mask_enabled[channel]
        self.update_display()

    def set_overlay_channel(self, label: str) -> None:
        self.overlay_channel = 1 if '1' in label else 2
        self.update_display()

    def show_histogram(self, channel: int) -> None:
        data_info = self.channel_display_data.get(channel)
        if data_info is None:
            print(f"No data available for channel {channel}")
            return
        data = data_info['data']  # type: ignore[assignment]
        label = data_info['label']  # type: ignore[assignment]
        unit = data_info['unit']  # type: ignore[assignment]

        hist_fig, hist_ax = plt.subplots(figsize=(8, 5))
        hist_fig.canvas.manager.set_window_title(f'Channel {channel} Histogram')
        data_flat = np.ravel(data)
        hist_ax.hist(data_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.75)
        hist_ax.set_xlabel(unit)
        hist_ax.set_ylabel('Frequency')
        hist_ax.set_title(
            f"{label} — Zone {self.current_zone:02d}\n"
            f"Center ({self.x_center}, {self.y_center}) | Box {self.box_size}px",
            fontsize=12,
        )
        hist_ax.grid(True, alpha=0.3)
        stats_text = (
            f"Mean: {data_flat.mean():.2f}\n"
            f"Std: {data_flat.std():.2f}\n"
            f"Min: {data_flat.min():.2f}\n"
            f"Max: {data_flat.max():.2f}\n"
            f"Pixels: {len(data_flat)}"
        )
        hist_ax.text(
            0.98,
            0.97,
            stats_text,
            transform=hist_ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
        plt.tight_layout()
        plt.show()


def main() -> None:
    print('Starting Auto Decision Cloud Mask Viewer')
    viewer = AutoDecisionCloudMaskViewer()
    plt.show()


if __name__ == '__main__':
    main()
