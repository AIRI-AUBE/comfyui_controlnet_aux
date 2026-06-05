#!/usr/bin/env python
"""Download all runtime model assets into the local offline ckpts layout."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPTS_DIR = Path(os.environ.get("AUX_ANNOTATOR_CKPTS_PATH", REPO_ROOT / "ckpts"))

TRANSFORMER_REPO_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "*.bin",
    "*.py",
]


@dataclass(frozen=True)
class HfFileAsset:
    repo_id: str
    filename: str
    repo_type: str = "model"
    optional: bool = False
    failure_hint: str | None = None

    def target_path(self, ckpts_dir: Path) -> Path:
        return ckpts_dir.joinpath(*self.repo_id.split("/"), *self.filename.split("/"))


@dataclass(frozen=True)
class HfRepoAsset:
    repo_id: str
    repo_type: str = "model"
    optional: bool = False
    failure_hint: str | None = None

    def target_dir(self, ckpts_dir: Path) -> Path:
        return ckpts_dir.joinpath(*self.repo_id.split("/"))


@dataclass(frozen=True)
class UrlAsset:
    url: str
    relative_path: str
    sha256_prefix: str | None = None
    optional: bool = False
    failure_hint: str | None = None

    def target_path(self, ckpts_dir: Path) -> Path:
        return ckpts_dir.joinpath(*self.relative_path.split("/"))


HF_FILE_ASSETS: tuple[HfFileAsset, ...] = (
    HfFileAsset("bdsqlsz/qinglong_controlnet-lllite", "Annotators/UNet.pth"),
    HfFileAsset("bdsqlsz/qinglong_controlnet-lllite", "Annotators/7_model.pth"),
    HfFileAsset("TheMistoAI/MistoLine", "Anyline/MTEED.pth"),
    HfFileAsset("skytnt/anime-seg", "isnetis.ckpt"),
    HfFileAsset("LayerNorm/DensePose-TorchScript-with-hint-image", "densepose_r50_fpn_dl.torchscript"),
    HfFileAsset("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth"),
    HfFileAsset("depth-anything/Depth-Anything-V2-Base", "depth_anything_v2_vitb.pth"),
    HfFileAsset("depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth"),
    HfFileAsset(
        "depth-anything/Depth-Anything-V2-Giant",
        "depth_anything_v2_vitg.pth",
        optional=True,
        failure_hint="The Giant checkpoint appears unavailable or gated without authenticated access.",
    ),
    HfFileAsset("depth-anything/Depth-Anything-V2-Metric-VKITTI-Large", "depth_anything_v2_metric_vkitti_vitl.pth"),
    HfFileAsset("depth-anything/Depth-Anything-V2-Metric-Hypersim-Large", "depth_anything_v2_metric_hypersim_vitl.pth"),
    HfFileAsset("hr16/Diffusion-Edge", "diffusion_edge_indoor.pt"),
    HfFileAsset("hr16/Diffusion-Edge", "diffusion_edge_urban.pt"),
    HfFileAsset("hr16/Diffusion-Edge", "diffusion_edge_natrual.pt"),
    HfFileAsset("hr16/Diffusion-Edge", "dsine.pt"),
    HfFileAsset("yzd-v/DWPose", "yolox_l.onnx"),
    HfFileAsset("yzd-v/DWPose", "dw-ll_ucoco_384.onnx"),
    HfFileAsset("hr16/yolox-onnx", "yolox_l.torchscript.pt"),
    HfFileAsset("hr16/yolo-nas-fp16", "yolo_nas_l_fp16.onnx"),
    HfFileAsset("hr16/yolo-nas-fp16", "yolo_nas_m_fp16.onnx"),
    HfFileAsset("hr16/yolo-nas-fp16", "yolo_nas_s_fp16.onnx"),
    HfFileAsset("hr16/UnJIT-DWPose", "rtmpose-m_ap10k_256.onnx"),
    HfFileAsset("hr16/DWPose-TorchScript-BatchSize5", "dw-ll_ucoco_384_bs5.torchscript.pt"),
    HfFileAsset("hr16/DWPose-TorchScript-BatchSize5", "rtmpose-m_ap10k_256_bs5.torchscript.pt"),
    HfFileAsset("lllyasviel/Annotators", "ControlNetHED.pth"),
    HfFileAsset("lllyasviel/Annotators", "res101.pth"),
    HfFileAsset("lllyasviel/Annotators", "latest_net_G.pth"),
    HfFileAsset("lllyasviel/Annotators", "sk_model.pth"),
    HfFileAsset("lllyasviel/Annotators", "sk_model2.pth"),
    HfFileAsset("lllyasviel/Annotators", "netG.pth"),
    HfFileAsset("lllyasviel/Annotators", "erika.pth"),
    HfFileAsset("lllyasviel/Annotators", "body_pose_model.pth"),
    HfFileAsset("lllyasviel/Annotators", "hand_pose_model.pth"),
    HfFileAsset("lllyasviel/Annotators", "facenet.pth"),
    HfFileAsset("lllyasviel/Annotators", "table5_pidinet.pth"),
    HfFileAsset("lllyasviel/Annotators", "mlsd_large_512_fp32.pth"),
    HfFileAsset("lllyasviel/Annotators", "scannet.pt"),
    HfFileAsset("lllyasviel/Annotators", "upernet_global_small.pth"),
    HfFileAsset("hr16/ControlNet-HandRefiner-pruned", "graphormer_hand_state_dict.bin"),
    HfFileAsset("hr16/ControlNet-HandRefiner-pruned", "hrnetv2_w64_imagenet_pretrained.pth"),
    HfFileAsset("JUGGHM/Metric3D", "metric_depth_vit_small_800k.pth"),
    HfFileAsset("JUGGHM/Metric3D", "metric_depth_vit_large_800k.pth"),
    HfFileAsset("JUGGHM/Metric3D", "metric_depth_vit_giant2_800k.pth"),
    HfFileAsset("hr16/Unimatch", "gmflow-scale2-regrefine6-mixdata.pth"),
    HfFileAsset("hr16/Unimatch", "gmflow-scale2-mixdata.pth"),
    HfFileAsset("hr16/Unimatch", "gmflow-scale1-mixdata.pth"),
)

TRANSFORMER_REPO_ASSETS: tuple[HfRepoAsset, ...] = (
    HfRepoAsset("LiheYoung/depth-anything-large-hf"),
    HfRepoAsset("LiheYoung/depth-anything-base-hf"),
    HfRepoAsset("LiheYoung/depth-anything-small-hf"),
    HfRepoAsset("Intel/dpt-large"),
    HfRepoAsset("Intel/dpt-hybrid-midas"),
    HfRepoAsset("shi-labs/oneformer_demo", repo_type="dataset"),
    HfRepoAsset("shi-labs/oneformer_ade20k_swin_large"),
    HfRepoAsset("shi-labs/oneformer_coco_swin_large"),
    HfRepoAsset("facebook/sam-vit-base"),
    HfRepoAsset("facebook/sam-vit-large"),
    HfRepoAsset("facebook/sam-vit-huge"),
    HfRepoAsset("Intel/zoedepth-nyu-kitti"),
)

URL_ASSETS: tuple[UrlAsset, ...] = (
    UrlAsset(
        "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        "torch/mobilenet_v2-b0353104.pth",
        "b0353104",
    ),
    UrlAsset(
        "https://download.pytorch.org/models/vgg16-397923af.pth",
        "torch/vgg16-397923af.pth",
        "397923af",
    ),
    UrlAsset(
        "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        "torch/resnet101-cd907fc2.pth",
        "cd907fc2",
    ),
    UrlAsset(
        "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
        "torch/efficientnet_b7_lukemelas-dcc49843.pth",
        "c5b4e57e",
    ),
    UrlAsset(
        "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
        "torch/swin_b-68c6b09e.pth",
        "68c6b09e",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all comfyui_controlnet_aux runtime model assets into ckpts/.",
    )
    parser.add_argument(
        "--ckpts-dir",
        type=Path,
        default=DEFAULT_CKPTS_DIR,
        help="Destination checkpoint root. Defaults to AUX_ANNOTATOR_CKPTS_PATH or ./ckpts.",
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token, if required for gated repos.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision for all HF downloads.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan without downloading.")
    parser.add_argument("--list", action="store_true", help="Alias for --dry-run.")
    parser.add_argument("--skip-hf-files", action="store_true", help="Skip individual Hugging Face checkpoint files.")
    parser.add_argument("--skip-transformer-repos", action="store_true", help="Skip full local transformer repository downloads.")
    parser.add_argument("--skip-url-files", action="store_true", help="Skip direct URL files such as PyTorch backbones.")
    parser.add_argument(
        "--full-transformer-repos",
        action="store_true",
        help="Download complete transformer repos instead of only runtime file patterns.",
    )
    return parser.parse_args()


def require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it in this environment with:\n"
            "  python -m pip install huggingface_hub"
        ) from exc

    return hf_hub_download, snapshot_download


def call_hf_function(function, **kwargs):
    return function(**kwargs)


def sha256_prefix_matches(path: Path, prefix: str) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().startswith(prefix)


def should_download(target: Path, force: bool, sha256_prefix: str | None = None) -> bool:
    if force or not target.exists():
        return True
    if sha256_prefix and not sha256_prefix_matches(target, sha256_prefix):
        print(f"Hash mismatch, redownloading: {target}")
        return True
    print(f"Skip existing: {target}")
    return False


def download_hf_file(asset: HfFileAsset, ckpts_dir: Path, hf_hub_download, args: argparse.Namespace) -> None:
    target = asset.target_path(ckpts_dir)
    if not should_download(target, args.force):
        return

    print(f"Download HF file: {asset.repo_id}/{asset.filename}")
    target.parent.mkdir(parents=True, exist_ok=True)
    call_hf_function(
        hf_hub_download,
        repo_id=asset.repo_id,
        filename=asset.filename,
        repo_type=asset.repo_type,
        revision=args.revision,
        token=args.token,
        local_dir=ckpts_dir.joinpath(*asset.repo_id.split("/")),
        force_download=args.force,
    )


def download_hf_repo(asset: HfRepoAsset, ckpts_dir: Path, snapshot_download, args: argparse.Namespace) -> None:
    target = asset.target_dir(ckpts_dir)
    print(f"Download transformer repo: {asset.repo_id}")
    target.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        repo_id=asset.repo_id,
        repo_type=asset.repo_type,
        revision=args.revision,
        token=args.token,
        local_dir=target,
        force_download=args.force,
    )
    if not args.full_transformer_repos:
        kwargs["allow_patterns"] = TRANSFORMER_REPO_PATTERNS

    call_hf_function(snapshot_download, **kwargs)


def download_url(asset: UrlAsset, ckpts_dir: Path, force: bool) -> None:
    target = asset.target_path(ckpts_dir)
    if not should_download(target, force, asset.sha256_prefix):
        return

    print(f"Download URL file: {asset.url}")
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        request = urllib.request.Request(asset.url, headers={"User-Agent": "comfyui-controlnet-aux-downloader"})
        with urllib.request.urlopen(request) as response, temp_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(f"  {downloaded / (1024 * 1024):.1f} MiB / {total / (1024 * 1024):.1f} MiB ({percent:.1f}%)", end="\r")
        if total:
            print()
        if asset.sha256_prefix and not sha256_prefix_matches(temp_path, asset.sha256_prefix):
            raise RuntimeError(f"SHA256 prefix check failed for {asset.url}")
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def print_plan(ckpts_dir: Path, include_hf_files: bool, include_repos: bool, include_urls: bool) -> None:
    print(f"Destination: {ckpts_dir}")
    print("Scope: runtime preprocessor assets and local torch backbones used by this repository.")
    if include_hf_files:
        print("\nHugging Face files:")
        for asset in HF_FILE_ASSETS:
            optional_marker = " [optional]" if asset.optional else ""
            print(f"  {asset.repo_id}/{asset.filename}{optional_marker} -> {asset.target_path(ckpts_dir)}")
    if include_repos:
        print("\nTransformer repos:")
        for asset in TRANSFORMER_REPO_ASSETS:
            optional_marker = " [optional]" if asset.optional else ""
            print(f"  {asset.repo_id}{optional_marker} -> {asset.target_dir(ckpts_dir)}")
    if include_urls:
        print("\nDirect URL files:")
        for asset in URL_ASSETS:
            optional_marker = " [optional]" if asset.optional else ""
            print(f"  {asset.url}{optional_marker} -> {asset.target_path(ckpts_dir)}")


def record_failure(failures: list[tuple[str, bool, str]], label: str, optional: bool, error: Exception, hint: str | None) -> None:
    message = f"{type(error).__name__}: {error}"
    if hint:
        message = f"{message}\n    hint: {hint}"
    failures.append((label, optional, message))


def print_failure_summary(failures: list[tuple[str, bool, str]]) -> None:
    if not failures:
        return

    required_failures = [failure for failure in failures if not failure[1]]
    optional_failures = [failure for failure in failures if failure[1]]

    print("\nDownload summary:")
    if optional_failures:
        print("  Optional assets skipped:")
        for label, _, message in optional_failures:
            print(f"    - {label}")
            print(f"      {message}")
    if required_failures:
        print("  Required assets failed:")
        for label, _, message in required_failures:
            print(f"    - {label}")
            print(f"      {message}")


def main() -> int:
    args = parse_args()
    ckpts_dir = args.ckpts_dir.expanduser().resolve()
    dry_run = args.dry_run or args.list
    include_hf_files = not args.skip_hf_files
    include_repos = not args.skip_transformer_repos
    include_urls = not args.skip_url_files

    if dry_run:
        print_plan(ckpts_dir, include_hf_files, include_repos, include_urls)
        return 0

    ckpts_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, bool, str]] = []

    hf_hub_download = snapshot_download = None
    if include_hf_files or include_repos:
        hf_hub_download, snapshot_download = require_huggingface_hub()

    if include_hf_files:
        for asset in HF_FILE_ASSETS:
            label = f"{asset.repo_id}/{asset.filename}"
            try:
                download_hf_file(asset, ckpts_dir, hf_hub_download, args)
            except Exception as exc:
                if asset.optional:
                    print(f"Optional asset unavailable, skipping: {label}")
                else:
                    print(f"Required asset failed: {label}")
                record_failure(failures, label, asset.optional, exc, asset.failure_hint)

    if include_repos:
        for asset in TRANSFORMER_REPO_ASSETS:
            label = asset.repo_id
            try:
                download_hf_repo(asset, ckpts_dir, snapshot_download, args)
            except Exception as exc:
                if asset.optional:
                    print(f"Optional asset unavailable, skipping: {label}")
                else:
                    print(f"Required asset failed: {label}")
                record_failure(failures, label, asset.optional, exc, asset.failure_hint)

    if include_urls:
        for asset in URL_ASSETS:
            label = asset.url
            try:
                download_url(asset, ckpts_dir, args.force)
            except Exception as exc:
                if asset.optional:
                    print(f"Optional asset unavailable, skipping: {label}")
                else:
                    print(f"Required asset failed: {label}")
                record_failure(failures, label, asset.optional, exc, asset.failure_hint)

    print_failure_summary(failures)
    print(f"Done. Assets are under: {ckpts_dir}")
    return 1 if any(not optional for _, optional, _ in failures) else 0


if __name__ == "__main__":
    sys.exit(main())