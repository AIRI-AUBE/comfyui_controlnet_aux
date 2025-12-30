"""custom_controlnet_aux package.

This project is configured for offline/local-only model loading. Any attempt to
contact Hugging Face (huggingface.co / hf.co) is blocked.
"""

import os
from urllib.parse import urlparse


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _is_hf_url(url: str) -> bool:
	try:
		host = urlparse(str(url)).hostname or ""
	except Exception:
		return False
	host = host.lower()
	return host.endswith("huggingface.co") or host.endswith("hf.co")


def _block_huggingface_network() -> None:
	# Block requests -> huggingface hosts.
	try:
		import requests  # type: ignore

		_orig_request = requests.sessions.Session.request

		def _request(self, method, url, *args, **kwargs):
			if _is_hf_url(url):
				raise RuntimeError(
					"Network access to Hugging Face is disabled (offline/local-only mode). "
					f"Blocked request: {method} {url}"
				)
			return _orig_request(self, method, url, *args, **kwargs)

		requests.sessions.Session.request = _request
	except Exception:
		pass

	# Block httpx -> huggingface hosts (some environments use httpx).
	try:
		import httpx  # type: ignore

		_orig_client_request = httpx.Client.request
		_orig_async_request = httpx.AsyncClient.request

		def _client_request(self, method, url, *args, **kwargs):
			if _is_hf_url(str(url)):
				raise RuntimeError(
					"Network access to Hugging Face is disabled (offline/local-only mode). "
					f"Blocked request: {method} {url}"
				)
			return _orig_client_request(self, method, url, *args, **kwargs)

		async def _async_client_request(self, method, url, *args, **kwargs):
			if _is_hf_url(str(url)):
				raise RuntimeError(
					"Network access to Hugging Face is disabled (offline/local-only mode). "
					f"Blocked request: {method} {url}"
				)
			return await _orig_async_request(self, method, url, *args, **kwargs)

		httpx.Client.request = _client_request
		httpx.AsyncClient.request = _async_client_request
	except Exception:
		pass


_block_huggingface_network()