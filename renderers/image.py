# renderers/image.py
from typing import Optional
import numpy as np
import cv2 as cv
from renderers.renderer import Renderer
import os


class Image(Renderer):
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
