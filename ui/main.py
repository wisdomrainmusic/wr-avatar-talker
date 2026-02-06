import os
import sys
import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup,
    QProgressBar, QMessageBox, QLineEdit, QGroupBox
)


def repo_root() -> Path:
    # ui/main.py -> repo root
    return Path(__file__).resolve().parents[1]


def default_output_path() -> Path:
    return repo_root() / "output" / "reels.mp4"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class Worker(QThread):
    started_msg = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, image: str, audio: str, preset: str, max_seconds: int, reels: bool, out_path: str):
        super().__init__()
        self.image = image
        self.audio = audio
        self.preset = preset
        self.max_seconds = max_seconds
        self.reels = reels
        self.out_path = out_path

    def run(self):
        try:
            self.started_msg.emit("Generating video… (CPU)")
            self.progress.emit(5)

            engine = repo_root() / "engine" / "run.py"
            if not engine.exists():
                raise RuntimeError("engine/run.py not found.")

            ensure_dir(Path(self.out_path).parent)

            # Call engine with current python (or bundled exe python)
            cmd = [
                sys.executable, str(engine),
                "--image", self.image,
                "--audio", self.audio,
                "--preset", self.preset,
                "--max_seconds", str(self.max_seconds),
                "--out", self.out_path
            ]
            if self.reels:
                cmd.append("--reels")

            self.progress.emit(10)

            # Run and stream output (no console, but errors will be caught)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(repo_root())
            )

            # Fake progress updates while running (CPU inference has no stable %)
            pct = 10
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    pct = min(95, pct + 1)
                    self.progress.emit(pct)
                if proc.poll() is not None:
                    break

            rc = proc.returncode
            if rc != 0:
                raise RuntimeError("Engine failed. Check inputs and ffmpeg availability.")

            self.progress.emit(100)
            self.finished_ok.emit(self.out_path)

        except Exception as e:
            self.progress.emit(0)
            self.finished_err.emit(str(e))


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WR Avatar Talker (Reels / CPU)")
        self.setMinimumWidth(560)

        self.image_path = ""
        self.audio_path = ""

        # --- UI Elements ---
        title = QLabel("WR Avatar Talker")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")

        subtitle = QLabel("Photo + MP3/WAV → Reels MP4 (1080×1920) • CPU-only • Offline")
        subtitle.setStyleSheet("color: #666;")

        # Image
        self.image_line = QLineEdit()
        self.image_line.setReadOnly(True)
        btn_image = QPushButton("Select Image")
        btn_image.clicked.connect(self.select_image)

        row_img = QHBoxLayout()
        row_img.addWidget(self.image_line, 1)
        row_img.addWidget(btn_image)

        # Audio
        self.audio_line = QLineEdit()
        self.audio_line.setReadOnly(True)
        btn_audio = QPushButton("Select Audio")
        btn_audio.clicked.connect(self.select_audio)

        row_aud = QHBoxLayout()
        row_aud.addWidget(self.audio_line, 1)
        row_aud.addWidget(btn_audio)

        # Preset
        preset_group = QGroupBox("Preset")
        self.rb_calm = QRadioButton("Calm (minimal head motion)")
        self.rb_normal = QRadioButton("Normal (slightly more expressive)")
        self.rb_calm.setChecked(True)

        self.preset_buttons = QButtonGroup()
        self.preset_buttons.addButton(self.rb_calm)
        self.preset_buttons.addButton(self.rb_normal)

        preset_layout = QVBoxLayout()
        preset_layout.addWidget(self.rb_calm)
        preset_layout.addWidget(self.rb_normal)
        preset_group.setLayout(preset_layout)

        # Max seconds (fixed 60, but visible)
        self.max_seconds_line = QLineEdit("60")
        self.max_seconds_line.setMaximumWidth(80)

        row_max = QHBoxLayout()
        row_max.addWidget(QLabel("Max seconds:"))
        row_max.addWidget(self.max_seconds_line)
        row_max.addStretch(1)

        # Output
        self.out_line = QLineEdit(str(default_output_path()))
        btn_out = QPushButton("Choose Output")
        btn_out.clicked.connect(self.select_output)

        row_out = QHBoxLayout()
        row_out.addWidget(self.out_line, 1)
        row_out.addWidget(btn_out)

        # Buttons
        self.btn_generate = QPushButton("Generate Video")
        self.btn_generate.clicked.connect(self.generate)

        self.btn_open = QPushButton("Open Output Folder")
        self.btn_open.clicked.connect(self.open_output_folder)
        self.btn_open.setEnabled(False)

        row_btn = QHBoxLayout()
        row_btn.addWidget(self.btn_generate, 1)
        row_btn.addWidget(self.btn_open)

        # Progress
        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.status = QLabel("Ready.")
        self.status.setStyleSheet("color: #333;")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(10)

        layout.addWidget(QLabel("Image:"))
        layout.addLayout(row_img)
        layout.addSpacing(6)

        layout.addWidget(QLabel("Audio:"))
        layout.addLayout(row_aud)
        layout.addSpacing(10)

        layout.addWidget(preset_group)
        layout.addLayout(row_max)
        layout.addSpacing(6)

        layout.addWidget(QLabel("Output MP4:"))
        layout.addLayout(row_out)
        layout.addSpacing(10)

        layout.addLayout(row_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        self.setLayout(layout)

        self.worker = None

    def select_image(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if fn:
            self.image_path = fn
            self.image_line.setText(fn)

    def select_audio(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", "Audio (*.mp3 *.wav)")
        if fn:
            self.audio_path = fn
            self.audio_line.setText(fn)

    def select_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Output As", str(default_output_path()), "MP4 Video (*.mp4)")
        if fn:
            if not fn.lower().endswith(".mp4"):
                fn += ".mp4"
            self.out_line.setText(fn)

    def preset_value(self) -> str:
        return "calm" if self.rb_calm.isChecked() else "normal"

    def generate(self):
        if not self.image_path:
            QMessageBox.warning(self, "Missing Image", "Please select an image.")
            return
        if not self.audio_path:
            QMessageBox.warning(self, "Missing Audio", "Please select an audio file (mp3/wav).")
            return

        try:
            max_seconds = int(self.max_seconds_line.text().strip())
        except Exception:
            QMessageBox.warning(self, "Invalid Max Seconds", "Max seconds must be a number (e.g., 60).")
            return

        out_path = self.out_line.text().strip()
        if not out_path:
            QMessageBox.warning(self, "Missing Output", "Please choose an output mp4 path.")
            return

        self.btn_generate.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Starting…")

        self.worker = Worker(
            image=self.image_path,
            audio=self.audio_path,
            preset=self.preset_value(),
            max_seconds=max_seconds,
            reels=True,
            out_path=out_path
        )
        self.worker.started_msg.connect(self.status.setText)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.finished_err.connect(self.on_err)
        self.worker.start()

    def on_done(self, out_path: str):
        self.btn_generate.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.status.setText(f"Done: {out_path}")
        QMessageBox.information(self, "Success", f"Video generated:\n{out_path}")

    def on_err(self, msg: str):
        self.btn_generate.setEnabled(True)
        self.btn_open.setEnabled(False)
        self.status.setText("Failed.")
        QMessageBox.critical(self, "Error", msg)

    def open_output_folder(self):
        out_path = Path(self.out_line.text().strip() or str(default_output_path()))
        folder = out_path.parent
        ensure_dir(folder)
        os.startfile(str(folder))


def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
