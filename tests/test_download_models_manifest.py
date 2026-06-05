from download_models import HF_FILE_ASSETS


def test_anyline_mteed_asset_is_in_offline_manifest():
    assert any(
        asset.repo_id == "TheMistoAI/MistoLine" and asset.filename == "Anyline/MTEED.pth"
        for asset in HF_FILE_ASSETS
    )