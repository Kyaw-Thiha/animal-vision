from typing import Optional
import numpy as np
import cv2
from renderers.renderer import Renderer


class VideoRenderer(Renderer):
    def __init__(
        self,
        *,
        read_path: Optional[str] = None,  # video to read
        write_path: Optional[str] = None,  # video to write
        fps: Optional[int] = None,  # fallback if source has no FPS
        window_name: str = "Video Analysis",  # optional preview
    ):
        self.read_path = read_path
        self.write_path = write_path
        self.fps = fps or 30
        self.window_name = window_name

        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self._writer_size: Optional[tuple[int, int]] = None
        self._window_created = False

    # ---------- Input ----------
    def open(self) -> None:
        if self.read_path:
            self.cap = cv2.VideoCapture(self.read_path)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Failed to open video for reading: {self.read_path}"
                )
            # If source FPS is available, use it for writing unless explicitly set
            src_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if src_fps and src_fps > 0:
                self.fps = int(round(src_fps))
        if self.window_name and not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_created = True

    def get_image(self) -> Optional[np.ndarray]:
        """Return next RGB frame from the input video, or None when exhausted."""
        if not self.cap:
            return None
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    # ---------- Output ----------
    def _ensure_writer(self, frame: np.ndarray) -> None:
        if not self.write_path:
            return
        if self.writer is not None:
            return
        h, w = frame.shape[:2]
        self._writer_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  #  type: ignore[attr-defined]
        self.writer = cv2.VideoWriter(self.write_path, fourcc, float(self.fps), (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video for writing: {self.write_path}")

    def render(self, frame: np.ndarray) -> None:
        """Write and/or preview one RGB frame."""
        # lazy-create writer on first frame (size known now)
        self._ensure_writer(frame)

        if self.writer:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(bgr)

        if self._window_created:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, bgr)
            # non-blocking; press 'q' to close preview
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                self.close()

    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
