from typing import Optional
import numpy as np
import cv2 as cv
from renderers.renderer import Renderer


class WebcamRenderer(Renderer):
    def __init__(
        self,
        *,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        write_path: Optional[str] = None,  # save processed output
        window_name: str = "Webcam Preview",
        mirror_preview: bool = True,  # flip for human-friendly view
        autofocus: bool = True,  # best-effort; driver-dependent
        auto_exposure: bool = True,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.write_path = write_path
        self.window_name = window_name
        self.mirror_preview = mirror_preview
        self.autofocus = autofocus
        self.auto_exposure = auto_exposure

        self.cap: Optional[cv.VideoCapture] = None
        self.writer: Optional[cv.VideoWriter] = None
        self._writer_size: Optional[tuple[int, int]] = None
        self._window_created = False

    # ---------- Lifecycle ----------
    def open(self) -> None:
        self.cap = cv.VideoCapture(self.index, cv.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam index {self.index}")

        # Try to set properties (best-effort)
        if self.width:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps:
            self.cap.set(cv.CAP_PROP_FPS, float(self.fps))

        # Optional autofocus/auto-exposure (driver & OS dependent)
        if self.autofocus is not None:
            try:
                self.cap.set(cv.CAP_PROP_AUTOFOCUS, 1.0 if self.autofocus else 0.0)
            except Exception:
                pass
        if self.auto_exposure is not None:
            try:
                # OpenCV uses odd AE encodings; 1 = auto, 0.25 = manual on some backends
                self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1.0 if self.auto_exposure else 0.25)
            except Exception:
                pass

        if self.window_name and not self._window_created:
            cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
            self._window_created = True

        # If camera reports a valid FPS, prefer that for writer
        src_fps = self.cap.get(cv.CAP_PROP_FPS)
        if src_fps and src_fps > 0:
            self.fps = int(round(src_fps))

    # ---------- Input ----------
    def get_image(self) -> Optional[np.ndarray]:
        """Grab the next RGB frame from the webcam, or None if failure."""
        if not self.cap:
            return None
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        return rgb

    # ---------- Output ----------
    def _ensure_writer(self, frame: np.ndarray) -> None:
        if not self.write_path or self.writer is not None:
            return
        h, w = frame.shape[:2]
        self._writer_size = (w, h)
        fourcc = cv.VideoWriter_fourcc(*"mp4v")  #  type: ignore[attr-defined]
        self.writer = cv.VideoWriter(self.write_path, fourcc, float(self.fps or 30), (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video for writing: {self.write_path}")

    def render(self, frame: np.ndarray) -> None:
        """Write and/or preview one RGB frame."""
        # Create writer lazily with the first frame
        self._ensure_writer(frame)

        if self.writer:
            bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            self.writer.write(bgr)

        if self._window_created:
            preview = frame
            if self.mirror_preview:
                preview = np.ascontiguousarray(frame[:, ::-1, :])  # fast horizontal flip
            bgr = cv.cvtColor(preview, cv.COLOR_RGB2BGR)
            cv.imshow(self.window_name, bgr)
            if (cv.waitKey(1) & 0xFF) == ord("q"):
                self.close()

    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None
        if self._window_created:
            cv.destroyWindow(self.window_name)
            self._window_created = False

    # ---------- Methods of split rendering ----------
    def _draw_label(self, img: np.ndarray, text: str, org: tuple[int, int]) -> None:
        """
        Draw a solid label box with text on an RGB frame (in-place).
        Args:
            img: HxWx3 RGB frame (uint8 or float).
            text: Label text.
            org: Bottom-left (x, y) of the text baseline.
        """
        font = cv.FONT_HERSHEY_SIMPLEX
        h = img.shape[0]
        font_scale = max(0.45, min(1.2, h / 900.0))  # scale roughly with height
        thickness = 1
        text_color = (255, 255, 255)  # RGB
        box_color = (0, 0, 0)
        pad = 6

        (tw, th), baseline = cv.getTextSize(text, font, font_scale, thickness)
        x, y = org
        x0 = max(x - pad, 0)
        y0 = max(y - th - baseline - pad, 0)
        x1 = min(x + tw + pad, img.shape[1] - 1)
        y1 = min(y + baseline + pad, img.shape[0] - 1)

        cv.rectangle(img, (x0, y0), (x1, y1), box_color, thickness=-1)
        cv.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv.LINE_AA)

    def make_split_frame(
        self,
        original: np.ndarray,
        modified: np.ndarray,
        *,
        left_label: str = "Original",
        right_label: str = "Transformed",
        draw_seam: bool = True,
    ) -> np.ndarray:
        """
        Create a half-and-half comparison frame (left: original, right: modified).
        Resizes `modified` to match `original` if needed; draws labels and an optional seam.
        Returns an RGB frame ready for `render()`.
        """
        assert isinstance(original, np.ndarray) and original.ndim == 3 and original.shape[2] == 3, "original must be HxWx3 RGB"
        assert isinstance(modified, np.ndarray) and modified.ndim == 3 and modified.shape[2] == 3, "modified must be HxWx3 RGB"

        H, W, _ = original.shape
        if modified.shape[:2] != (H, W):
            modified_rs = cv.resize(modified, (W, H), interpolation=cv.INTER_AREA)
        else:
            modified_rs = modified

        out = original.copy()
        mid = W // 2
        out[:, mid:, :] = modified_rs[:, mid:, :]

        if draw_seam:
            out[:, mid : mid + 1, :] = 255  # 1px white seam

        # Labels
        self._draw_label(out, left_label, org=(10, 24))
        (tw, _), _ = cv.getTextSize(right_label, cv.FONT_HERSHEY_SIMPLEX, max(0.45, min(1.2, H / 900.0)), 1)
        self._draw_label(out, right_label, org=(max(W - tw - 10, 10), 24))

        return out

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
        Compose a labeled split frame and write/preview it via `render()`.
        Respects `mirror_preview` for the on-screen view only (saved video is non-mirrored).
        """
        frame = self.make_split_frame(
            original,
            modified,
            left_label=left_label,
            right_label=right_label,
            draw_seam=draw_seam,
        )
        self.render(frame)
