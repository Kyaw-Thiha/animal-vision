from typing import Optional
import numpy as np
import cv2
from renderers.renderer import Renderer


class VideoRenderer(Renderer):
    """
    Video I/O helper that reads frames from a file, renders processed frames to a window,
    and (optionally) writes them to a new video file. Frames are exchanged in RGB.

    Usage:
        vr = VideoRenderer(read_path="in.mp4", write_path="out.mp4", window_name="Preview")
        vr.open()
        while True:
            frame = vr.get_image()
            if frame is None:
                break
            # out = your_processing(frame)  # keep RGB
            out = frame
            vr.render(out)  # previews and saves if write_path is set
        vr.close()

    Split comparison:
        split = vr.make_split_frame(original, modified, left_label="Input", right_label="Output")
        vr.render(split)
        # or directly:
        vr.render_split_compare(original, modified)
    """

    def __init__(
        self,
        *,
        read_path: Optional[str] = None,
        write_path: Optional[str] = None,
        fps: Optional[int] = None,
        window_name: str = "Video Analysis",
    ):
        """
        Initialize the renderer.

        Args:
            read_path: Optional path to an input video. If provided, frames can be read via get_image().
            write_path: Optional path to an output video. If provided, frames passed to render() will be saved.
            fps: Target FPS for the output file when the source has no FPS metadata. Defaults to 30.
            window_name: Name of the (optional) OpenCV preview window.
        """
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
        """
        Open the input and (optionally) create a preview window.

        - If `read_path` is set, opens a cv2.VideoCapture for reading.
        - If the source FPS is available and valid, it overrides the FPS fallback for writing.
        - If `window_name` is non-empty, creates a resizable preview window.
        """
        if self.read_path:
            self.cap = cv2.VideoCapture(self.read_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video for reading: {self.read_path}")

            # Prefer source FPS for writing if available
            src_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if src_fps and src_fps > 0:
                # Keep as float internally but store rounded int for consistency in UI
                self.fps = int(round(float(src_fps)))

        if self.window_name and not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_created = True

    def get_image(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the input video.

        Returns:
            The next frame as an HxWx3 RGB uint8 array, or None when the stream ends
            or when no input is open.
        """
        if not self.cap:
            return None
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    # ---------- Output ----------
    def _ensure_writer(self, frame: np.ndarray) -> None:
        """
        Lazily create the output video writer on the first frame (when size is known).

        Args:
            frame: First frame to be written; used to infer output size.
        """
        if not self.write_path:
            return
        if self.writer is not None:
            return

        h, w = frame.shape[:2]
        self._writer_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self.writer = cv2.VideoWriter(self.write_path, fourcc, float(self.fps), (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video for writing: {self.write_path}")

    def render(self, frame: np.ndarray) -> None:
        """
        Preview and/or save one RGB frame.

        Behavior:
            - If `write_path` was provided, the frame is encoded and appended to the output video.
            - If a preview window exists, the frame is shown non-blockingly.
              Press 'q' to close the preview and release resources.

        Args:
            frame: HxWx3 RGB uint8 array (if float, it will be interpreted incorrectly by the writer).
        """
        # Create writer on first frame (size established now)
        self._ensure_writer(frame)

        if self.writer:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(bgr)

        if self._window_created:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, bgr)
            # Non-blocking UI; press 'q' to close preview
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                self.close()

    def close(self) -> None:
        """
        Release all resources: input capture, output writer, and preview window (if any).
        Safe to call multiple times.
        """
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

    # ---------- Split rendering helpers ----------
    def _draw_label(self, img: np.ndarray, text: str, org: tuple[int, int]) -> None:
        """
        Draw a semi-transparent label box with outlined text on an RGB frame (in-place).

        Args:
            img: HxWx3 RGB frame (uint8 or float).
            text: Label text.
            org: Bottom-left (x, y) of the text baseline.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        h = img.shape[0]
        font_scale = max(0.5, min(1.2, h / 900.0))  # scale with video height
        thickness = 2
        text_color = (255, 255, 255)  # white text
        outline_color = (0, 0, 0)  # black outline
        pad = 8

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = org

        # Keep inside frame bounds
        if x + tw + pad > img.shape[1]:
            x = img.shape[1] - tw - pad
        if y - th - baseline - pad < 0:
            y = th + baseline + pad

        x0, y0 = max(x - pad, 0), max(y - th - baseline - pad, 0)
        x1, y1 = min(x + tw + pad, img.shape[1] - 1), min(y + baseline + pad, img.shape[0] - 1)

        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # Outlined text for readability
        cv2.putText(img, text, (x, y), font, font_scale, outline_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

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
        Construct a left/right split comparison frame (RGB).

        - Resizes `modified` to match `original` if needed.
        - Left half is `original`, right half is `modified`.
        - Optionally draws a 1px white seam at the vertical split.
        - Adds corner labels.

        Args:
            original: HxWx3 RGB image (left side).
            modified: HxWx3 RGB image (right side).
            left_label: Label to show on the left.
            right_label: Label to show on the right.
            draw_seam: Whether to draw a vertical seam at the center.

        Returns:
            HxWx3 RGB image suitable to pass into `render()`.
        """
        assert isinstance(original, np.ndarray) and original.ndim == 3 and original.shape[2] == 3, "original must be HxWx3 RGB"
        assert isinstance(modified, np.ndarray) and modified.ndim == 3 and modified.shape[2] == 3, "modified must be HxWx3 RGB"

        H, W, _ = original.shape
        if modified.shape[:2] != (H, W):
            modified_rs = cv2.resize(modified, (W, H), interpolation=cv2.INTER_AREA)
        else:
            modified_rs = modified

        out = original.copy()
        mid = W // 2
        out[:, mid:, :] = modified_rs[:, mid:, :]

        if draw_seam:
            out[:, mid : mid + 1, :] = 255  # white seam

        # Labels
        self._draw_label(out, left_label, org=(10, 24))
        (tw, _), _ = cv2.getTextSize(right_label, cv2.FONT_HERSHEY_SIMPLEX, max(0.45, min(1.2, H / 900.0)), 1)
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
        Compose a labeled split frame and send it to `render()`.

        Args:
            original: HxWx3 RGB image for the left side.
            modified: HxWx3 RGB image for the right side.
            left_label: Label text for the left.
            right_label: Label text for the right.
            draw_seam: Whether to draw the 1px seam at center.
        """
        frame = self.make_split_frame(
            original,
            modified,
            left_label=left_label,
            right_label=right_label,
            draw_seam=draw_seam,
        )
        self.render(frame)
