"""The anonymizer with the Swiss German checkpoint. The only module that imports it."""
import random
import tempfile
from pathlib import Path

import numpy as np

# which of the model config's directories each checkpoint file comes from
SOURCES = {
    "model.pth": "swiss_dir",
    "config.json": "swiss_dir",
    "vocab.json": "swiss_dir",
    "dvae.pth": "base_dir",
    "mel_stats.pth": "base_dir",
}


def assemble_model_dir(model_cfg) -> Path:
    """Link the five checkpoint files into the one directory the anonymizer expects."""
    model_dir = Path(tempfile.mkdtemp(prefix="globi-model-"))
    for name, source in SOURCES.items():
        src = (Path(model_cfg[source]) / name).resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        (model_dir / name).symlink_to(src)
    return model_dir


def set_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


class Engine:
    def __init__(self, model_cfg, device: str = "auto", whisper_model: str = "small"):
        from anonymizer import Anonymizer

        self.anon = Anonymizer(
            model_dir=str(assemble_model_dir(model_cfg)),
            device=None if device == "auto" else device,
            mode="single",
            whisper_model=whisper_model,
        )
        # load XTTS now instead of on the first job; a wrong vocab.json would otherwise
        # fail silently, encoding the dialect code as plain text
        tokenizer = self.anon.session.model.tokenizer.tokenizer
        missing = [lang for lang in model_cfg.languages if tokenizer.token_to_id(f"[{lang}]") is None]
        if missing:
            raise RuntimeError(f"vocab.json has no token for {missing}; is swiss_dir the Swiss checkpoint?")

    def anonymize(self, target: bytes, reference: bytes, language: str,
                  text: str | None = None, seed: int | None = None) -> tuple[np.ndarray, int, str]:
        """Return (waveform, sample rate, transcript used)."""
        if seed is not None:
            # the "single" XTTS pass is deterministic; the seed matters for Whisper's
            # temperature fallback and for any sampling a future mode adds
            set_seed(seed)
        with tempfile.TemporaryDirectory() as tmp:
            target_path, reference_path = Path(tmp, "target.wav"), Path(tmp, "reference.wav")
            target_path.write_bytes(target)
            reference_path.write_bytes(reference)
            if not text:
                # Whisper knows no dialect codes; Swiss German comes out as Standard German,
                # which is also what the model was trained on
                text = self.anon.session.transcribe(str(target_path), "de")
            result = self.anon.anonymize(str(target_path), reference=str(reference_path),
                                         text=text, language=language)
        return result.wav, result.sample_rate, result.text
