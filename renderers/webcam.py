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
                self.cap.set(
                    cv.CAP_PROP_AUTO_EXPOSURE, 1.0 if self.auto_exposure else 0.25
                )
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
        self.writer = cv.VideoWriter(
            self.write_path, fourcc, float(self.fps or 30), (w, h)
        )
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
                preview = np.ascontiguousarray(
                    frame[:, ::-1, :]
                )  # fast horizontal flip
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
