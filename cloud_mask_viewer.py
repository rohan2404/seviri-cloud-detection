"""
Custom Cloud Mask Viewer
------------------------
Visualise MSG channels 1, 2 (radiances) and 7, 9, 10 (brightness temperatures)
side-by-side, apply hard thresholds per channel, and generate a combined cloud
mask that you can view directly or overlaid on a selected visible image.

Controls:
* Zone slider (z00-z24), X/Y/box inputs to select the region of interest
* Threshold text boxes for each channel (>= for VIS/NIR radiances, <= for IR BTs)
* Checkboxes to include/exclude channels from the combined mask
* Radio buttons to choose whether the overlay uses channel 1 or 2 as the base
* Histogram buttons beneath each channel panel to inspect pixel distributions
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.widgets import Slider, TextBox, Button, CheckButtons, RadioButtons
from matplotlib.colors import ListedColormap

BASE_DIR = 'data'
PROJECT_ROOT = Path(__file__).resolve().parent

CHANNEL_CONFIGS = [
    {
        'id': 1,
        'name': 'Ch1 0.6μm (VIS)',
        'short': 'Ch1',
        'type': 'radiance',
        'unit': 'Radiance',
        'cmap': 'Greys_r',
        'percentiles': (1, 99),
        'threshold': 65.0,
        'comparison': '>=',
    },
    {
        'id': 2,
        'name': 'Ch2 0.8μm (NIR)',
        'short': 'Ch2',
        'type': 'radiance',
        'unit': 'Radiance',
        'cmap': 'bone_r',
        'percentiles': (1, 99),
        'threshold': 70.0,
        'comparison': '>=',
    },
    {
        'id': 7,
        'name': 'Ch7 8.7μm (BT)',
        'short': 'Ch7',
        'type': 'bt',
        'unit': 'Brightness Temp [K]',
        'cmap': 'inferno',
        'range': (190, 320),
        'threshold': 240.0,
        'comparison': '<=',
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
        'threshold': 240.0,
        'comparison': '<=',
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
        'threshold': 240.0,
        'comparison': '<=',
        'wavelength': 11.94,
    },
]

CONFIG_LOOKUP = {cfg['id']: cfg for cfg in CHANNEL_CONFIGS}


class CloudMaskViewer:
    def __init__(self):
        self.current_zone = 12
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100
        self.overlay_channel = 1  # Base visible channel for overlay

        self.images: dict[int, dict[str, np.ndarray | int | str]] = {}
        self.colorbars = []
        self.hist_buttons = []
        self.channel_display_data: dict[int, dict[str, object]] = {}
        self.per_channel_masks: dict[int, np.ndarray] = {}
        self.thresholds = {cfg['id']: cfg['threshold'] for cfg in CHANNEL_CONFIGS}
        self.mask_enabled = {
            1: False,
            2: False,
            7: True,
            9: True,
            10: True,
        }

        self.mask_cmap = ListedColormap([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.8],
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

        # Control widgets --------------------------------------------------
        ax_zone = plt.axes([0.12, 0.20, 0.76, 0.02])
        self.zone_slider = Slider(ax_zone, 'Zone', 0, 24, valinit=self.current_zone, valstep=1)
        self.zone_slider.on_changed(self.update_zone)

        ax_x = plt.axes([0.12, 0.16, 0.1, 0.028])
        self.x_box = TextBox(ax_x, 'X Center', initial=str(self.x_center))
        self.x_box.on_submit(self.update_x)

        ax_y = plt.axes([0.32, 0.16, 0.1, 0.028])
        self.y_box = TextBox(ax_y, 'Y Center', initial=str(self.y_center))
        self.y_box.on_submit(self.update_y)

        ax_box = plt.axes([0.52, 0.16, 0.1, 0.028])
        self.box_box = TextBox(ax_box, 'Box Size', initial=str(self.box_size))
        self.box_box.on_submit(self.update_box_size)

        ax_reset = plt.axes([0.72, 0.16, 0.16, 0.028])
        self.reset_button = Button(ax_reset, 'Reset View')
        self.reset_button.on_clicked(self.reset_view)

        # Threshold text boxes for each channel
        self.threshold_boxes: dict[int, TextBox] = {}
        left_start = 0.05
        box_width = 0.16
        for idx, cfg in enumerate(CHANNEL_CONFIGS):
            left = left_start + idx * (box_width + 0.01)
            ax_thr = plt.axes([left, 0.11, box_width, 0.032])
            label = f"{cfg['short']} Thr ({cfg['comparison']})"
            box = TextBox(ax_thr, label, initial=str(self.thresholds[cfg['id']]))
            box.on_submit(lambda text, ch=cfg['id']: self.set_threshold(ch, text))
            self.threshold_boxes[cfg['id']] = box

        # Mask inclusion checkboxes
        ax_mask_check = plt.axes([0.05, 0.02, 0.16, 0.08])
        labels = [cfg['short'] for cfg in CHANNEL_CONFIGS]
        actives = [self.mask_enabled[cfg['id']] for cfg in CHANNEL_CONFIGS]
        self.label_to_channel = {cfg['short']: cfg['id'] for cfg in CHANNEL_CONFIGS}
        self.mask_checks = CheckButtons(ax_mask_check, labels, actives)
        self.mask_checks.on_clicked(self.toggle_mask_channel)
        ax_mask_check.set_title('Mask Inputs', fontsize=9)

        # Overlay radio buttons (choose VIS base)
        ax_overlay_radio = plt.axes([0.25, 0.02, 0.12, 0.08])
        self.overlay_radio = RadioButtons(ax_overlay_radio, ('Channel 1', 'Channel 2'),
                                          active=0)
        self.overlay_radio.on_clicked(self.set_overlay_channel)
        ax_overlay_radio.set_title('Overlay Base', fontsize=9)

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
        H = 6.63e-34  # Planck constant
        C = 3.00e8    # Speed of light
        K = 1.38e-23  # Boltzmann constant
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
            ax.set_title(cfg['name'], fontsize=8, pad=2)
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

            threshold = self.thresholds[cfg['id']]
            comparator = cfg.get('comparison', '>=')
            if comparator == '>=':
                mask = zoom_data >= threshold
            else:
                mask = zoom_data <= threshold
            self.per_channel_masks[cfg['id']] = mask

        combined_mask = self.compute_combined_mask()
        self.display_mask(combined_mask)
        self.display_overlay(combined_mask)
        self.fig.suptitle(
            f'Cloud Mask Viewer — Zone {self.current_zone:02d} | '
            f'Center ({self.x_center}, {self.y_center}) px | '
            f'Box {self.box_size}px',
            fontsize=11,
            fontweight='bold',
        )
        self.fig.canvas.draw_idle()

    def compute_combined_mask(self) -> np.ndarray | None:
        active_masks = [
            self.per_channel_masks[ch]
            for ch in self.per_channel_masks
            if self.mask_enabled.get(ch) and self.per_channel_masks[ch].size
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
            self.mask_ax.text(0.5, 0.5, 'Cloud mask disabled', ha='center', va='center',
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
            subtitle = 'Cloud mask disabled'
        self.overlay_ax.set_title(
            f'Overlay on {base_cfg["name"]}\n{subtitle}',
            fontsize=10,
            pad=4,
        )

    # --------------------------- Callbacks -----------------------------
    def update_zone(self, _val: float) -> None:
        new_zone = int(self.zone_slider.val)
        if new_zone != self.current_zone:
            self.current_zone = new_zone
            self.load_zone(self.current_zone)
            self.update_display()

    def update_x(self, text: str) -> None:
        try:
            self.x_center = int(text)
            self.update_display()
        except ValueError:
            print(f"Invalid X center: {text}")

    def update_y(self, text: str) -> None:
        try:
            self.y_center = int(text)
            self.update_display()
        except ValueError:
            print(f"Invalid Y center: {text}")

    def update_box_size(self, text: str) -> None:
        try:
            self.box_size = max(10, int(text))
            self.update_display()
        except ValueError:
            print(f"Invalid box size: {text}")

    def reset_view(self, _event) -> None:
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100
        self.x_box.set_val(str(self.x_center))
        self.y_box.set_val(str(self.y_center))
        self.box_box.set_val(str(self.box_size))
        self.update_display()

    def set_threshold(self, channel: int, text: str) -> None:
        try:
            value = float(text)
            self.thresholds[channel] = value
            self.update_display()
        except ValueError:
            print(f"Invalid threshold for channel {channel}: {text}")
            self.threshold_boxes[channel].set_val(str(self.thresholds[channel]))

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
        save_ax = hist_fig.add_axes([0.02, 0.90, 0.18, 0.07])
        save_btn = Button(save_ax, 'Save CSV', color='#90EE90', hovercolor='#2E8B57')
        save_btn.label.set_fontsize(8)

        def _save(_event, ch=channel, arr=data_flat.copy()):
            self.save_histogram_data(ch, arr)

        save_btn.on_clicked(_save)
        plt.tight_layout()
        plt.show()

    def save_histogram_data(self, channel: int, data: np.ndarray) -> None:
        filename = (
            f"histogram_ch{channel}_z{self.current_zone:02d}_"
            f"x{self.x_center}_y{self.y_center}_box{self.box_size}.csv"
        )
        filepath = PROJECT_ROOT / filename
        header = (
            f"channel,{channel}\n"
            f"zone,{self.current_zone:02d}\n"
            f"x_center,{self.x_center}\n"
            f"y_center,{self.y_center}\n"
            f"box_size,{self.box_size}\n"
            "values"
        )
        np.savetxt(filepath, data, delimiter=',', header=header, comments='')
        print(f"Histogram data saved to {filepath}")


def main() -> None:
    print('Starting Cloud Mask Viewer (Channels 1, 2, 7, 9, 10)')
    viewer = CloudMaskViewer()
    plt.show()


if __name__ == '__main__':
    main()
