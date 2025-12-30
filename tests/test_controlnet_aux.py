import os
import shutil

import numpy as np
import pytest
from PIL import Image

from custom_controlnet_aux import (CannyDetector, ContentShuffleDetector, HEDdetector,
                            LeresDetector, LineartAnimeDetector,
                            LineartDetector, MediapipeFaceDetector,
                            MidasDetector, MLSDdetector, NormalBaeDetector,
                            OpenposeDetector, PidiNetDetector, SamDetector,
                            ZoeDetector, TileDetector)

OUTPUT_DIR = "tests/outputs"


def _skip_if_missing_pretrained(fn):
    """Decorator: skip a test if local pretrained weights are not present."""
    def _wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as e:
            pytest.skip(str(e))
    return _wrapped


def _make_test_image(size=512) -> Image.Image:
    """Create a deterministic local image; avoids any network dependency."""
    x = np.linspace(0, 255, size, dtype=np.uint8)
    grad = np.tile(x, (size, 1))
    img = np.stack([grad, np.flipud(grad), np.roll(grad, size // 4, axis=1)], axis=2)
    return Image.fromarray(img, mode="RGB")

def output(name, img):
    img.save(os.path.join(OUTPUT_DIR, "{:s}.png".format(name)))

def common(name, processor, img):
    output(name, processor(img))
    output(name + "_pil_np", Image.fromarray(processor(img, output_type="np")))
    output(name + "_np_np", Image.fromarray(processor(np.array(img, dtype=np.uint8), output_type="np")))
    output(name + "_np_pil", processor(np.array(img, dtype=np.uint8), output_type="pil"))
    output(name + "_scaled", processor(img, detect_resolution=640, image_resolution=768))

def return_pil(name, processor, img):
    output(name + "_pil_false", Image.fromarray(processor(img, return_pil=False)))
    output(name + "_pil_true", processor(img, return_pil=True))

@pytest.fixture(scope="module")
def img():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    return _make_test_image(512)

def test_canny(img):
    canny = CannyDetector()
    common("canny", canny, img)
    output("canny_img", canny(img=img))

@_skip_if_missing_pretrained
def test_hed(img):
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    common("hed", hed, img)
    return_pil("hed", hed, img)
    output("hed_safe", hed(img, safe=True))
    output("hed_scribble", hed(img, scribble=True))

@_skip_if_missing_pretrained
def test_leres(img):
    leres = LeresDetector.from_pretrained("lllyasviel/Annotators")
    common("leres", leres, img)
    output("leres_boost", leres(img, boost=True))

@_skip_if_missing_pretrained
def test_lineart(img):
    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
    common("lineart", lineart, img)
    return_pil("lineart", lineart, img)
    output("lineart_coarse", lineart(img, coarse=True))

@_skip_if_missing_pretrained
def test_lineart_anime(img):
    lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    common("lineart_anime", lineart_anime, img)
    return_pil("lineart_anime", lineart_anime, img)

def test_mediapipe_face(img):
    mediapipe = MediapipeFaceDetector()
    common("mediapipe", mediapipe, img)
    output("mediapipe_image", mediapipe(image=img))

@_skip_if_missing_pretrained
def test_midas(img):
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
    common("midas", midas, img)
    output("midas_normal", midas(img, depth_and_normal=True)[1])

@_skip_if_missing_pretrained
def test_mlsd(img):
    mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
    common("mlsd", mlsd, img)
    return_pil("mlsd", mlsd, img)

@_skip_if_missing_pretrained
def test_normalbae(img):
    normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
    common("normal_bae", normal_bae, img)
    return_pil("normal_bae", normal_bae, img)

@_skip_if_missing_pretrained
def test_openpose(img):
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    common("openpose", openpose, img)
    return_pil("openpose", openpose, img)
    output("openpose_hand_and_face_false", openpose(img, hand_and_face=False))
    output("openpose_hand_and_face_true", openpose(img, hand_and_face=True))
    output("openpose_face", openpose(img, include_body=True, include_hand=False, include_face=True))
    output("openpose_faceonly", openpose(img, include_body=False, include_hand=False, include_face=True))
    output("openpose_full", openpose(img, include_body=True, include_hand=True, include_face=True))
    output("openpose_hand", openpose(img, include_body=True, include_hand=True, include_face=False))

@_skip_if_missing_pretrained
def test_pidi(img):
    pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    common("pidi", pidi, img)
    return_pil("pidi", pidi, img)
    output("pidi_safe", pidi(img, safe=True))
    output("pidi_scribble", pidi(img, scribble=True))

@_skip_if_missing_pretrained
def test_sam(img):
    sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
    common("sam", sam, img)
    output("sam_image", sam(image=img))

def test_shuffle(img):
    shuffle = ContentShuffleDetector()
    common("shuffle", shuffle, img)
    return_pil("shuffle", shuffle, img)

@_skip_if_missing_pretrained
def test_zoe(img):
    zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
    common("zoe", zoe, img)

def test_tile(img):
    tile = TileDetector()
    common("tile", tile, img)
    output("tile_img", tile(img))