from typing import Optional
import numpy as np
import cv2 as cv
from renderers.renderer import Renderer
import os


class ImageRenderer(Renderer):
    image_path: str = ""
    original_image: Optional[np.ndarray] = None  # RGB
    visualized_image: Optional[np.ndarray] = None  # RGB

    def __init__(
        self,
        image_path: str,
        *,
        show_window: bool = True,
        window_name: str = "Animal Vision",
        save_to: Optional[str] = None,  # e.g., "out.png"
        wait_key: int = 0,  # 0 = wait indefinitely
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.show_window = show_window
        self.window_name = window_name
        self.save_to = save_to
        self.wait_key = wait_key
        self._window_created = False

    # ---------- Input (loader) ----------
    def get_image(self) -> Optional[np.ndarray]:
        """Load once, cache in RGB."""
        if self.original_image is not None:
            return self.original_image
        if not os.path.exists(self.image_path):
            print(f"[Image] Not found: {self.image_path}")
            return None
        bgr = cv.imread(self.image_path, cv.IMREAD_UNCHANGED)
        if bgr is None:
            print(f"[Image] Failed to read: {self.image_path}")
            return None
        # Normalize channels (handle gray, BGR, BGRA)
        if bgr.ndim == 2:
            rgb = cv.cvtColor(bgr, cv.COLOR_GRAY2RGB)
        elif bgr.shape[2] == 3:
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        elif bgr.shape[2] == 4:
            rgb = cv.cvtColor(bgr, cv.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"Unsupported channel shape: {bgr.shape}")
        self.original_image = rgb
        return self.original_image

    # ---------- Output (render) ----------
    def open(self) -> None:
        if self.show_window and not self._window_created:
            cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
            self._window_created = True

    def render(self, frame: np.ndarray) -> None:
        """Render one frame (RGB in → show/save)."""
        self.visualized_image = frame
        if self.save_to:
            # Write as sRGB PNG/JPEG
            bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            ok = cv.imwrite(self.save_to, bgr)
            if not ok:
                print(f"[Image] Failed to save: {self.save_to}")

        if self.show_window:
            bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            cv.imshow(self.window_name, bgr)
            # waitKey returns immediately if wait_key==1; blocks if 0
            key = cv.waitKey(self.wait_key) & 0xFF
            # Close on 'q' if non-blocking mode
            if self.wait_key != 0 and key == ord("q"):
                self.close()

    def close(self) -> None:
        if self._window_created:
            cv.destroyWindow(self.window_name)
            self._window_created = False

    # ---------- Convenience for your current call-site ----------
    def send_image(self, image: np.ndarray) -> None:
        """Keep for compatibility: just delegates to render()."""
        self.render(image)

    # ---------- Methods of split rendering ----------
    def _draw_label(self, img: np.ndarray, text: str, org: tuple[int, int]) -> None:
        """
        Draw a solid label box with text on an RGB image.

        Args:
            img: RGB uint8/float image (modified in place).
            text: Label text.
            org: Bottom-left corner (x, y) of the text baseline.
        """
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_color = (255, 255, 255)  # white (RGB)
        box_color = (0, 0, 0)  # black background
        pad = 6

        (tw, th), baseline = cv.getTextSize(text, font, font_scale, thickness)
        x, y = org
        x0 = max(x - pad, 0)
        y0 = max(y - th - baseline - pad, 0)
        x1 = min(x + tw + pad, img.shape[1] - 1)
        y1 = min(y + baseline + pad, img.shape[0] - 1)

        cv.rectangle(img, (x0, y0), (x1, y1), box_color, thickness=-1)
        cv.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv.LINE_AA)

    def render_split_compare(
        self,
        original: np.ndarray,
        modified: np.ndarray,
        *,
        left_label: str = "Original",
        right_label: str = "Transformed",
        draw_seam: bool = True,
    ) -> None:
        """
        Show/save a half-and-half comparison: left half from `original`, right half from `modified`.

        - Resizes `modified` to match `original` if needed.
        - Draws labels on top-left and top-right corners.
        - Uses existing window/save behavior via `self.render`.

        Args:
            original: RGB image (HxWx3).
            modified: RGB image (HxWx3), will be resized to match `original` if sizes differ.
            left_label: Label for the left (original) half.
            right_label: Label for the right (modified) half.
            draw_seam: If True, draw a thin vertical seam at the split.
        """
        assert isinstance(original, np.ndarray) and original.ndim == 3 and original.shape[2] == 3, (
            "original must be an HxWx3 RGB image"
        )
        assert isinstance(modified, np.ndarray) and modified.ndim == 3 and modified.shape[2] == 3, (
            "modified must be an HxWx3 RGB image"
        )

        H, W, _ = original.shape
        if modified.shape[:2] != (H, W):
            # OpenCV expects BGR but we're only resizing, so channel order doesn't matter here
            modified_rs = cv.resize(modified, (W, H), interpolation=cv.INTER_AREA)
        else:
            modified_rs = modified

        # Compose half-and-half
        out = original.copy()
        mid = W // 2
        out[:, mid:, :] = modified_rs[:, mid:, :]

        # Optional seam
        if draw_seam:
            # Draw a 1px white line at the split
            out[:, mid : mid + 1, :] = 255

        # Labels
        self._draw_label(out, left_label, org=(10, 24))  # top-left area
        right_text_size, _ = cv.getTextSize(right_label, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        rt_w = right_text_size[0]
        self._draw_label(out, right_label, org=(max(W - rt_w - 10, 10), 24))

        # Show/save using existing pipeline
        self.render(out)
