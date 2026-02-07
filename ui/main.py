from __future__ import annotations

import os
import sys
from pathlib import Path  # ✅ Path garanti
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup,
    QProgressBar, QMessageBox, QLineEdit, QGroupBox, QCheckBox
)

# ✅ CRITICAL: run.bat bozulsa bile engine import çalışsın
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """
    Dev: repo root (ui/main.py -> parents[1])
    EXE: sys._MEIPASS (onefile) or internal layout (onedir)
    """
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return REPO_ROOT


def work_root() -> Path:
    """
    Dev: repo root
    EXE: folder where the exe is located
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return REPO_ROOT


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _valid_exe(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def resolve_ffmpeg_path() -> str:
    """
    Dev: repo/ffmpeg/ffmpeg.exe (or ffmpeg in PATH)
    EXE: dist yanında ffmpeg/ffmpeg.exe vb.
    """
    # EXE side
    if is_frozen():
        exe_dir = work_root()
        candidates = [
            exe_dir / "ffmpeg" / "ffmpeg.exe",
            exe_dir / "ffmpeg.exe",
            exe_dir / "_internal" / "ffmpeg" / "ffmpeg.exe",
            exe_dir / "_internal" / "ffmpeg.exe",
        ]
        for c in candidates:
            if _valid_exe(c):
                return str(c)

    if is_frozen() and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
        candidates = [
            base / "ffmpeg" / "ffmpeg.exe",
            base / "ffmpeg.exe",
            base / "_internal" / "ffmpeg" / "ffmpeg.exe",
            base / "_internal" / "ffmpeg.exe",
        ]
        for c in candidates:
            if _valid_exe(c):
                return str(c)

    # Dev side
    local_ffmpeg = app_root() / "ffmpeg" / "ffmpeg.exe"
    if _valid_exe(local_ffmpeg):
        return str(local_ffmpeg)

    return "ffmpeg"


def ensure_ffmpeg_env() -> str:
    """
    engine tarafı da aynı ffmpeg'i bulsun diye ENV'e yazıyoruz.
    """
    ffmpeg = resolve_ffmpeg_path()
    os.environ["WR_AVATAR_TALKER_FFMPEG"] = str(ffmpeg)
    return ffmpeg


class Worker(QThread):
    started_msg = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(
        self,
        image: str,
        audio: str,
        preset: str,
        max_seconds: int,
        reels: bool,
        out_path: str,
        *,
        size: int = 256,
        preprocess: str = "crop",
        output_mode: str = "reels",
        cpu_threads: int = 4,
        keep_framing: bool = True,
        device_mode: str = "auto",
        eye_contact: str = "locked",
    ):
        super().__init__()
        self.image = image
        self.audio = audio
        self.preset = preset
        self.max_seconds = max_seconds
        self.reels = reels
        self.out_path = out_path
        self.size = int(size)
        self.preprocess = str(preprocess)
        self.output_mode = str(output_mode)
        self.cpu_threads = int(cpu_threads)
        self.keep_framing = bool(keep_framing)
        self.device_mode = str(device_mode)
        self.eye_contact = str(eye_contact)

    def run(self):
        try:
            ensure_ffmpeg_env()

            # ✅ Direct import (repo root sys.path'e eklendi)
            from engine.run import generate

            def _cb(pct: int, msg: str) -> None:
                self.started_msg.emit(msg)
                self.progress.emit(int(pct))

            ensure_dir(Path(self.out_path).parent)

            out_file = generate(
                image=self.image,
                audio=self.audio,
                preset=self.preset,
                max_seconds=int(self.max_seconds),
                reels=bool(self.reels),
                out=self.out_path,
                size=int(self.size),
                preprocess=str(self.preprocess),
                output_mode=str(self.output_mode),
                cpu_threads=int(self.cpu_threads),
                keep_framing=bool(self.keep_framing),
                device_mode=str(self.device_mode),
                eye_contact=str(self.eye_contact),
                progress_cb=_cb,
            )

            self.progress.emit(100)
            self.finished_ok.emit(str(out_file))

        except Exception as e:
            self.progress.emit(0)
            self.finished_err.emit(str(e))


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WR Avatar Talker (Reels / CPU/GPU)")
        self.setMinimumWidth(560)

        self.image_path = ""
        self.audio_path = ""

        title = QLabel("WR Avatar Talker")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")

        subtitle = QLabel("Photo + MP3/WAV → MP4 • CPU/GPU • Offline")
        subtitle.setStyleSheet("color: #666;")

        # Device
        self.rb_dev_auto = QRadioButton("Auto (use GPU if available)")
        self.rb_dev_cpu = QRadioButton("Force CPU")
        self.rb_dev_gpu = QRadioButton("Force GPU (CUDA)")
        self.rb_dev_auto.setChecked(True)
        self.dev_group = QButtonGroup(self)
        self.dev_group.addButton(self.rb_dev_auto)
        self.dev_group.addButton(self.rb_dev_cpu)
        self.dev_group.addButton(self.rb_dev_gpu)

        dev_box = QGroupBox("Device")
        dev_layout = QVBoxLayout()
        dev_layout.addWidget(self.rb_dev_auto)
        dev_layout.addWidget(self.rb_dev_cpu)
        dev_layout.addWidget(self.rb_dev_gpu)
        dev_box.setLayout(dev_layout)

        # Image
        self.image_line = QLineEdit()
        btn_img = QPushButton("Select Image")
        btn_img.clicked.connect(self.pick_image)

        row_img = QHBoxLayout()
        row_img.addWidget(QLabel("Image:"))
        row_img.addWidget(self.image_line, 1)
        row_img.addWidget(btn_img)

        # Audio
        self.audio_line = QLineEdit()
        btn_aud = QPushButton("Select Audio")
        btn_aud.clicked.connect(self.pick_audio)

        row_aud = QHBoxLayout()
        row_aud.addWidget(QLabel("Audio:"))
        row_aud.addWidget(self.audio_line, 1)
        row_aud.addWidget(btn_aud)

        # Preset
        self.rb_calm = QRadioButton("Calm (minimal head motion)")
        self.rb_normal = QRadioButton("Normal (slightly more motion)")
        self.rb_ultra = QRadioButton("Ultra Speed (less motion)")
        self.rb_calm.setChecked(True)
        self.preset_group = QButtonGroup(self)
        self.preset_group.addButton(self.rb_calm)
        self.preset_group.addButton(self.rb_normal)
        self.preset_group.addButton(self.rb_ultra)

        preset_box = QGroupBox("Preset")
        preset_layout = QVBoxLayout()
        preset_layout.addWidget(self.rb_calm)
        preset_layout.addWidget(self.rb_normal)
        preset_layout.addWidget(self.rb_ultra)
        preset_box.setLayout(preset_layout)

        # Resolution
        self.rb_256 = QRadioButton("256 (Fast)")
        self.rb_512 = QRadioButton("512 (Quality)")
        self.rb_256.setChecked(True)
        self.size_group = QButtonGroup(self)
        self.size_group.addButton(self.rb_256)
        self.size_group.addButton(self.rb_512)

        size_box = QGroupBox("Resolution")
        size_layout = QVBoxLayout()
        size_layout.addWidget(self.rb_256)
        size_layout.addWidget(self.rb_512)
        size_box.setLayout(size_layout)

        # Preprocess
        self.rb_crop = QRadioButton("crop (Fast)")
        self.rb_full = QRadioButton("full (Quality)")
        self.rb_crop.setChecked(True)
        self.pre_group = QButtonGroup(self)
        self.pre_group.addButton(self.rb_crop)
        self.pre_group.addButton(self.rb_full)

        pre_box = QGroupBox("Preprocess")
        pre_layout = QVBoxLayout()
        pre_layout.addWidget(self.rb_crop)
        pre_layout.addWidget(self.rb_full)
        pre_box.setLayout(pre_layout)

        # Postprocess / output mode
        self.rb_out_reels = QRadioButton("Reels (1080×1920)")
        self.rb_out_720p = QRadioButton("720p (1280×720)")
        self.rb_out_none = QRadioButton("None (use SadTalker output)")
        self.rb_out_reels.setChecked(True)
        self.out_group = QButtonGroup(self)
        self.out_group.addButton(self.rb_out_reels)
        self.out_group.addButton(self.rb_out_720p)
        self.out_group.addButton(self.rb_out_none)

        outmode_box = QGroupBox("Postprocess")
        outmode_layout = QVBoxLayout()
        outmode_layout.addWidget(self.rb_out_reels)
        outmode_layout.addWidget(self.rb_out_720p)
        outmode_layout.addWidget(self.rb_out_none)
        outmode_box.setLayout(outmode_layout)

        # Eye contact
        self.rb_eye_locked = QRadioButton("Locked (YouTuber — look at camera)")
        self.rb_eye_locked_plus = QRadioButton("Locked+ (YouTuber Pro — clamp + micro motion)")
        self.rb_eye_natural = QRadioButton("Natural")
        self.rb_eye_expressive = QRadioButton("Expressive")
        self.rb_eye_locked.setChecked(True)

        self.eye_group = QButtonGroup(self)
        self.eye_group.addButton(self.rb_eye_locked)
        self.eye_group.addButton(self.rb_eye_locked_plus)
        self.eye_group.addButton(self.rb_eye_natural)
        self.eye_group.addButton(self.rb_eye_expressive)

        eye_box = QGroupBox("Eye Contact")
        eye_layout = QVBoxLayout()
        eye_layout.addWidget(self.rb_eye_locked)
        eye_layout.addWidget(self.rb_eye_locked_plus)
        eye_layout.addWidget(self.rb_eye_natural)
        eye_layout.addWidget(self.rb_eye_expressive)
        eye_box.setLayout(eye_layout)

        # Keep original framing (avoid zoomed-in crop output)
        self.cb_keep_framing = QCheckBox("Keep original framing (no zoom)")
        self.cb_keep_framing.setChecked(True)

        # CPU threads
        self.cpu_threads_line = QLineEdit("4")
        self.cpu_threads_line.setFixedWidth(80)
        row_cpu = QHBoxLayout()
        row_cpu.addWidget(QLabel("CPU threads:"))
        row_cpu.addWidget(self.cpu_threads_line)
        row_cpu.addStretch(1)

        # Max seconds
        self.max_line = QLineEdit("60")
        self.max_line.setFixedWidth(80)
        row_max = QHBoxLayout()
        row_max.addWidget(QLabel("Max seconds:"))
        row_max.addWidget(self.max_line)
        row_max.addStretch(1)

        # Output
        self.out_line = QLineEdit(str(REPO_ROOT / "output" / "reels.mp4"))
        btn_out = QPushButton("Choose Output")
        btn_out.clicked.connect(self.pick_output)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Output MP4:"))
        row_out.addWidget(self.out_line, 1)
        row_out.addWidget(btn_out)

        # Buttons
        self.btn_generate = QPushButton("Generate Video")
        self.btn_generate.clicked.connect(self.on_generate)

        self.btn_open = QPushButton("Open Output Folder")
        self.btn_open.clicked.connect(self.open_output_folder)
        self.btn_open.setEnabled(False)

        row_btn = QHBoxLayout()
        row_btn.addWidget(self.btn_generate, 2)
        row_btn.addWidget(self.btn_open, 1)

        # Progress
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.status = QLabel("")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addLayout(row_img)
        layout.addLayout(row_aud)
        layout.addWidget(dev_box)
        layout.addWidget(preset_box)
        layout.addWidget(size_box)
        layout.addWidget(pre_box)
        layout.addWidget(outmode_box)
        layout.addWidget(eye_box)
        layout.addWidget(self.cb_keep_framing)
        layout.addLayout(row_cpu)
        layout.addLayout(row_max)
        layout.addLayout(row_out)
        layout.addSpacing(6)
        layout.addLayout(row_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        self.setLayout(layout)

        self.worker: Optional[Worker] = None

    def pick_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if p:
            self.image_line.setText(p)

    def pick_audio(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", "Audio (*.wav *.mp3)")
        if p:
            self.audio_line.setText(p)

    def pick_output(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save Output MP4", str(REPO_ROOT / "output"), "MP4 (*.mp4)")
        if p:
            self.out_line.setText(p)

    def open_output_folder(self):
        out_path = Path(self.out_line.text().strip())
        folder = out_path.parent
        if folder.exists():
            os.startfile(str(folder))

    def on_generate(self):
        image = self.image_line.text().strip()
        audio = self.audio_line.text().strip()
        out_path = self.out_line.text().strip()

        if not image or not Path(image).exists():
            QMessageBox.critical(self, "Error", "Please select a valid image.")
            return
        if not audio or not Path(audio).exists():
            QMessageBox.critical(self, "Error", "Please select a valid audio.")
            return

        preset = "ultra" if getattr(self, "rb_ultra", None) and self.rb_ultra.isChecked() else ("calm" if self.rb_calm.isChecked() else "normal")

        # Resolution + preprocess
        size = 256 if self.rb_256.isChecked() else 512
        preprocess = "crop" if self.rb_crop.isChecked() else "full"

        # Output mode
        if self.rb_out_720p.isChecked():
            output_mode = "720p"
        elif self.rb_out_none.isChecked():
            output_mode = "none"
        else:
            output_mode = "reels"

        # Eye contact
        if getattr(self, "rb_eye_locked_plus", None) and self.rb_eye_locked_plus.isChecked():
            eye_contact = "locked_plus"
        elif self.rb_eye_expressive.isChecked():
            eye_contact = "expressive"
        elif self.rb_eye_natural.isChecked():
            eye_contact = "natural"
        else:
            eye_contact = "locked"

        try:
            max_seconds = int(self.max_line.text().strip())
        except Exception:
            QMessageBox.critical(self, "Error", "Max seconds must be a number.")
            return

        try:
            cpu_threads = int(self.cpu_threads_line.text().strip())
            if cpu_threads < 1:
                raise ValueError()
        except Exception:
            QMessageBox.critical(self, "Error", "CPU threads must be a positive number.")
            return

        self.btn_generate.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Starting…")

        # Device mode
        if self.rb_dev_cpu.isChecked():
            device_mode = "cpu"
        elif self.rb_dev_gpu.isChecked():
            device_mode = "gpu"
        else:
            device_mode = "auto"

        # reels=True for backward compat: engine uses output_mode anyway
        self.worker = Worker(
            image,
            audio,
            preset,
            max_seconds,
            reels=True,
            out_path=out_path,
            size=size,
            preprocess=preprocess,
            output_mode=output_mode,
            cpu_threads=cpu_threads,
            keep_framing=bool(self.cb_keep_framing.isChecked()),
            device_mode=device_mode,
            eye_contact=str(eye_contact),
        )
        self.worker.started_msg.connect(self.status.setText)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished_ok.connect(self.on_done_ok)
        self.worker.finished_err.connect(self.on_done_err)
        self.worker.start()

    def on_done_ok(self, out_file: str):
        self.btn_generate.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.status.setText("Done.")
        QMessageBox.information(self, "OK", f"Video created:\n{out_file}")

    def on_done_err(self, err: str):
        self.btn_generate.setEnabled(True)
        self.btn_open.setEnabled(False)
        self.status.setText("Failed.")
        QMessageBox.critical(self, "Error", err)


def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
