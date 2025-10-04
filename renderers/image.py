from typing import Optional
import numpy as np
import cv2 as cv
from renderers.renderer import Renderer
import os


class ImageRenderer(Renderer):
    """
    Simple image I/O helper that:
      • Loads an image from disk exactly once (cached as RGB).
      • Renders (shows) an image in an OpenCV window (optional).
      • Saves the currently visualized image to disk (optional).
      • Provides a split-screen comparison helper (left: original, right: modified).

    Notes
    -----
    - All public methods expect/provide RGB images with shape (H, W, 3), dtype=uint8.
    - Internally converts only at I/O boundaries (disk read/write and imshow).
    - The class does not mutate input arrays except where explicitly documented
      (e.g., drawing labels on a composed split image before rendering).
    """

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
        """
        Initialize the renderer with an input path and optional display/save settings.

        Parameters
        ----------
        image_path : str
            Path to the image file to load (read on first `get_image()` call).
        show_window : bool, optional
            If True, calls to `render()`/`render_split_compare()` will open/show an
            OpenCV window. Defaults to True.
        window_name : str, optional
            Name of the OpenCV preview window. Ignored if `show_window` is False.
        save_to : Optional[str], optional
            If provided, every call to `render()` (including via split rendering)
            will save the current frame to this path (PNG/JPEG inferred by extension).
        wait_key : int, optional
            Milliseconds to wait in `cv.waitKey()`. Use 0 to block indefinitely,
            or a small positive value (e.g., 1–30) for non-blocking UI. Defaults to 0.
        """
        super().__init__()
        self.image_path = image_path
        self.show_window = show_window
        self.window_name = window_name
        self.save_to = save_to
        self.wait_key = wait_key
        self._window_created = False

    # ---------- Input (loader) ----------
    def get_image(self) -> Optional[np.ndarray]:
        """
        Load the source image from disk once and cache it as RGB.

        Returns
        -------
        Optional[np.ndarray]
            The RGB image (H, W, 3, uint8) if successfully loaded; otherwise None.

        Behavior
        --------
        - If the image was previously loaded, returns the cached copy.
        - Supports grayscale, BGR, and BGRA inputs; converts to RGB.
        - Prints a short message and returns None if the file is missing or unreadable.
        """
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
        """
        Prepare the preview window if `show_window` is enabled.

        Notes
        -----
        - Safe to call multiple times; the window is created only once.
        - No image is displayed until `render()` or `render_split_compare()` is called.
        """
        if self.show_window and not self._window_created:
            cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
            self._window_created = True

    def render(self, frame: np.ndarray) -> None:
        """
        Render (show and/or save) a single RGB frame.

        Parameters
        ----------
        frame : np.ndarray
            RGB image with shape (H, W, 3). Should be dtype=uint8 for correct saving.

        Behavior
        --------
        - If `save_to` is set, writes the frame to disk (PNG/JPEG by extension).
        - If `show_window` is True, displays the frame in the named window.
        - Uses `wait_key` to control UI blocking (0 blocks; >0 is non-blocking).
        - In non-blocking mode (`wait_key` > 0), pressing 'q' closes the window.
        - Stores the last rendered frame in `visualized_image`.
        """
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
        """
        Close the preview window if it exists.

        Notes
        -----
        - Safe to call multiple times.
        - Does not clear cached images; only tears down the UI window.
        """
        if self._window_created:
            cv.destroyWindow(self.window_name)
            self._window_created = False

    # ---------- Convenience for your current call-site ----------
    def send_image(self, image: np.ndarray) -> None:
        """
        Backward-compatible alias for `render()`.

        Parameters
        ----------
        image : np.ndarray
            RGB image with shape (H, W, 3). Forwarded to `render(image)`.
        """
        self.render(image)

    # ---------- Methods of split rendering ----------
    def _draw_label(self, img: np.ndarray, text: str, org: tuple[int, int]) -> None:
        """
        Draw a semi-transparent label box with outlined text on an RGB image (in place).

        Parameters
        ----------
        img : np.ndarray
            Target RGB image (HxWx3), modified in place.
        text : str
            Label text to draw.
        org : tuple[int, int]
            Bottom-left (x, y) coordinates of the text baseline.

        Notes
        -----
        - Chooses a fixed font scale suitable for typical HD/UHD frames.
        - Keeps the label within image bounds and adds padding for readability.
        """
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_color = (255, 255, 255)  # white
        outline_color = (0, 0, 0)  # black
        pad = 6

        (tw, th), baseline = cv.getTextSize(text, font, font_scale, thickness)
        x, y = org
        x0 = max(x - pad, 0)
        y0 = max(y - th - baseline - pad, 0)
        x1 = min(x + tw + pad, img.shape[1] - 1)
        y1 = min(y + baseline + pad, img.shape[0] - 1)

        # Transparent box
        overlay = img.copy()
        cv.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
        cv.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # Text with outline
        cv.putText(img, text, (x, y), font, font_scale, outline_color, thickness + 2, cv.LINE_AA)
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
        Compose and render a half-and-half comparison frame.

        The left half is taken from `original` and the right half from `modified`.
        Labels are drawn in the top-left and top-right, and a thin vertical seam can
        be drawn at the split for clarity. The composed image is sent through the
        standard `render()` pipeline (show/save).

        Parameters
        ----------
        original : np.ndarray
            RGB image (HxWx3) for the left half.
        modified : np.ndarray
            RGB image (HxWx3) for the right half. Will be resized to match `original`
            if spatial dimensions differ.
        left_label : str, optional
            Text label for the left half. Defaults to "Original".
        right_label : str, optional
            Text label for the right half. Defaults to "Transformed".
        draw_seam : bool, optional
            If True, draw a 1-pixel white vertical seam at the center split. Defaults to True.

        Raises
        ------
        AssertionError
            If either input is not an HxWx3 array.

        Notes
        -----
        - The composed image is created by copying `original` and replacing the right half
          with the (possibly resized) `modified`. Inputs are not mutated.
        - After composition and label drawing, the result is passed to `render()`.
        """
        assert isinstance(original, np.ndarray) and original.ndim == 3 and original.shape[2] == 3, (
            "original must be an HxWx3 RGB image"
        )
        assert isinstance(modified, np.ndarray) and modified.ndim == 3 and modified.shape[2] == 3, (
            "modified must be an HxWx3 RGB image"
        )

        H, W, _ = original.shape
        if modified.shape[:2] != (H, W):
            # OpenCV expects BGR but we're only resizing; channel order is irrelevant here
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
