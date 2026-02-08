"""
Radiance Multi-Channel Viewer
--------------------------------
Displays the 11 MSG channels as radiance (or reflectance for the visible band) using
channel-appropriate colour tables. The viewer lets you:

* Choose the time/zone (z00-z24) via slider
* Specify the X/Y pixel centre and the box size you want to examine
* Inspect per-channel histograms for the zoomed window

Usage mirrors the other viewers that ship with this project.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

BASE_DIR = 'data'

# Per-channel styling metadata
CHANNEL_STYLES = {
    1: dict(name='Ch1 0.6μm (VIS)', cmap='Greys_r', percentiles=(1, 99), unit='Radiance'),
    2: dict(name='Ch2 0.8μm (NIR)', cmap='bone_r', percentiles=(1, 99), unit='Radiance'),
    3: dict(name='Ch3 1.6μm (NIR)', cmap='gist_gray', percentiles=(1, 99), unit='Radiance'),
    4: dict(name='Ch4 3.9μm (SWIR)', cmap='magma', percentiles=(2, 98), unit='Radiance'),
    5: dict(name='Ch5 6.2μm (WV)', cmap='YlGnBu_r', percentiles=(2, 98), unit='Radiance'),
    6: dict(name='Ch6 7.3μm (WV)', cmap='YlGnBu', percentiles=(2, 98), unit='Radiance'),
    7: dict(name='Ch7 8.7μm (IR)', cmap='inferno', percentiles=(2, 98), unit='Radiance'),
    8: dict(name='Ch8 9.7μm (O3)', cmap='PuRd', percentiles=(2, 98), unit='Radiance'),
    9: dict(name='Ch9 10.8μm (IR)', cmap='plasma', percentiles=(2, 98), unit='Radiance'),
    10: dict(name='Ch10 11.94μm (IR)', cmap='viridis', percentiles=(2, 98), unit='Radiance'),
    11: dict(name='Ch11 13.4μm (IR)', cmap='cividis', percentiles=(2, 98), unit='Radiance'),
}


class RadianceMultiChannelViewer:
    def __init__(self):
        self.current_zone = 12
        self.x_center = 300
        self.y_center = 800
        self.box_size = 100

        self.images: dict[int, dict[str, np.ndarray | int | str]] = {}
        self.colorbars = []
        self.hist_buttons = []
        self.current_display_data: dict[int, dict[str, np.ndarray | str | bool]] = {}

        self.fig = plt.figure(figsize=(13, 9))
        gs = self.fig.add_gridspec(4, 4, left=0.02, right=0.99, top=0.99,
                                   bottom=0.14, hspace=0.50, wspace=0.15)

        self.axes = []
        for i in range(11):
            row = i // 4
            col = i % 4
            ax = self.fig.add_subplot(gs[row, col])
            ax.axis('off')
            self.axes.append(ax)
            self.hist_buttons.append(None)

        ax_zone = plt.axes([0.12, 0.09, 0.76, 0.018])
        self.zone_slider = Slider(ax_zone, 'Zone', 0, 24,
                                  valinit=self.current_zone, valstep=1)
        self.zone_slider.on_changed(self.update_zone)

        ax_x = plt.axes([0.12, 0.045, 0.1, 0.022])
        self.x_box = TextBox(ax_x, 'X Center', initial=str(self.x_center))
        self.x_box.on_submit(self.update_x)

        ax_y = plt.axes([0.32, 0.045, 0.1, 0.022])
        self.y_box = TextBox(ax_y, 'Y Center', initial=str(self.y_center))
        self.y_box.on_submit(self.update_y)

        ax_box = plt.axes([0.52, 0.045, 0.1, 0.022])
        self.box_box = TextBox(ax_box, 'Box Size', initial=str(self.box_size))
        self.box_box.on_submit(self.update_box_size)

        ax_reset = plt.axes([0.72, 0.045, 0.16, 0.022])
        self.reset_button = Button(ax_reset, 'Reset View')
        self.reset_button.on_clicked(self.reset_view)

        self.load_zone(self.current_zone)
        self.update_display()

    def load_zone(self, zone: int) -> None:
        print(f"\nLoading zone {zone:02d}...")
        self.images = {}
        for channel in range(1, 12):
            img_path = f"{BASE_DIR}/channel_{channel}/msg_c{channel:02d}_z{zone:02d}.img"
            try:
                self.images[channel] = self.load_image(img_path)
                meta = self.images[channel]
                print(f"  Ch{channel}: ✓ ({meta['nx']}x{meta['ny']})")
            except Exception as exc:  # noqa: BLE001
                print(f"  Ch{channel}: ✗ ({exc})")
                self.images[channel] = None
        loaded_count = len([v for v in self.images.values() if v is not None])
        print(f"Loaded {loaded_count}/11 channels")

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

    def get_zoom_region(self, data: np.ndarray) -> np.ndarray:
        ny, nx = data.shape
        half_box = self.box_size // 2
        x_min = max(0, self.x_center - half_box)
        x_max = min(nx, self.x_center + half_box)
        y_min = max(0, self.y_center - half_box)
        y_max = min(ny, self.y_center + half_box)
        return data[y_min:y_max, x_min:x_max]

    def compute_scale(self, data: np.ndarray, channel: int) -> tuple[float | None, float | None]:
        style = CHANNEL_STYLES.get(channel, {})
        if not data.size:
            return None, None
        percentiles = style.get('percentiles')
        if percentiles is None:
            return style.get('vmin'), style.get('vmax')
        low, high = percentiles
        vmin = np.percentile(data, low)
        vmax = np.percentile(data, high)
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
        self.current_display_data = {}

        for i, channel in enumerate(range(1, 12)):
            ax = self.axes[i]
            ax.clear()
            ax.axis('off')

            image = self.images.get(channel)
            if image is None:
                ax.text(0.5, 0.5, f'Ch{channel}\nNot Found', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                continue

            data = image['data']  # type: ignore[index]
            zoom_data = self.get_zoom_region(data)
            style = CHANNEL_STYLES.get(channel, {})
            cmap = style.get('cmap', 'gray')
            vmin, vmax = self.compute_scale(zoom_data, channel)
            label = style.get('name', f'Channel {channel}')
            unit = style.get('unit', 'Radiance')

            self.current_display_data[channel] = {
                'data': zoom_data,
                'label': label,
                'unit': unit,
            }

            im = ax.imshow(zoom_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{label}\nZone {self.current_zone:02d}", fontsize=7, pad=1)
            cbar = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
            cbar.set_label(unit, fontsize=6)
            self.colorbars.append(cbar)

            bbox = ax.get_position()
            button_ax = plt.axes([bbox.x0 + bbox.width * 0.15,
                                   bbox.y0 - 0.045,
                                   bbox.width * 0.7,
                                   0.020])
            hist_btn = Button(button_ax, 'Histogram', color='#87CEEB', hovercolor='#4682B4')
            hist_btn.label.set_fontsize(9)
            hist_btn.on_clicked(lambda _event, ch=channel: self.show_histogram(ch))
            self.hist_buttons[i] = hist_btn

        self.fig.canvas.draw_idle()

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
            print(f"Invalid X: {text}")

    def update_y(self, text: str) -> None:
        try:
            self.y_center = int(text)
            self.update_display()
        except ValueError:
            print(f"Invalid Y: {text}")

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

    def show_histogram(self, channel: int) -> None:
        if channel not in self.current_display_data:
            print(f"No data available for channel {channel}")
            return
        data_info = self.current_display_data[channel]
        data = data_info['data']
        unit = data_info['unit']
        label = data_info['label']

        hist_fig, hist_ax = plt.subplots(figsize=(8, 5))
        hist_fig.canvas.manager.set_window_title(f'Channel {channel} Histogram')
        data_flat = np.ravel(data)
        hist_ax.hist(data_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.75)
        hist_ax.set_xlabel(unit)
        hist_ax.set_ylabel('Frequency')
        hist_ax.set_title(f"{label} — Zone {self.current_zone:02d}\n"
                          f"Center ({self.x_center}, {self.y_center}) | Box {self.box_size}px",
                          fontsize=12)
        hist_ax.grid(True, alpha=0.3)
        stats_text = (f"Mean: {data_flat.mean():.2f}\n"
                      f"Std: {data_flat.std():.2f}\n"
                      f"Min: {data_flat.min():.2f}\n"
                      f"Max: {data_flat.max():.2f}\n"
                      f"Pixels: {len(data_flat)}")
        hist_ax.text(0.98, 0.97, stats_text, transform=hist_ax.transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.show()


def main() -> None:
    print("All-Channel Radiance Viewer")
    print("Controls: Zone slider, X/Y/Box inputs, per-channel histograms.")
    viewer = RadianceMultiChannelViewer()
    plt.show()


if __name__ == '__main__':
    main()
