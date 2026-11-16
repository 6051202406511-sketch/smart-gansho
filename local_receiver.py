"""Token-protected, LAN-only photo receiver with a QR-code upload link."""

from __future__ import annotations

import io
import json
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit
from uuid import uuid4

from PIL import Image

from security import create_private_cache

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class _PhotoHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(20)
        return request, address


def local_ip_address() -> str:
    """Find the preferred outbound LAN address without sending application data."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 80))
        address = probe.getsockname()[0]
        if address and not address.startswith("127."):
            return address
    except OSError:
        pass
    finally:
        probe.close()

    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = item[4][0]
            if not address.startswith("127."):
                return address
    except OSError:
        pass
    raise OSError("このPCのローカルネットワークアドレスを取得できません。")


def _upload_page(token: str) -> bytes:
    document = f"""<!doctype html>
<html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>写真を送信</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:34rem;margin:12vh auto;padding:1.5rem;
background:#f5f7fa;color:#17212b;text-align:center}}
button{{font-size:1.25rem;font-weight:700;padding:1rem 1.4rem;border:0;border-radius:.7rem;
background:#167a4a;color:white;min-height:3.5rem}}
input{{position:absolute;width:1px;height:1px;opacity:0}}
#status{{margin-top:1.2rem;min-height:3rem}}
</style>
<main><h1>写真をPCへ送信</h1>
<p>PCアプリと同じWi-Fiに接続していることを確認してください。</p>
<input id="photo" type="file" accept="image/*" capture="environment">
<button id="send" type="button">写真を撮影して送信</button>
<p id="status" role="status">送信する写真はこのPCに一時保存されます。</p></main>
<script>
const input=document.getElementById('photo'), button=document.getElementById('send');
const status=document.getElementById('status');
button.addEventListener('click',()=>input.click());
input.addEventListener('change',async()=>{{
 if(!input.files.length)return;
 button.disabled=true; status.textContent='写真を送信しています…';
 try{{
   const response=await fetch('/{token}/upload',{{method:'POST',
     headers:{{'Content-Type':input.files[0].type||'application/octet-stream'}},
     body:input.files[0]}});
   const result=await response.json();
   if(!response.ok)throw new Error(result.error||'送信できませんでした');
   status.textContent='送信しました。PCの画面で確認してください。';
 }}catch(error){{status.textContent='送信できませんでした: '+error.message;}}
 finally{{button.disabled=false;input.value='';}}
}});
</script></html>"""
    return document.encode("utf-8")


class LocalPhotoReceiver:
    """Run a small upload server on the local network and store photos in cache."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        on_upload: Callable[[Path], None] | None = None,
        *,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else create_private_cache()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.on_upload = on_upload
        self.max_upload_bytes = max_upload_bytes
        self.token = secrets.token_urlsafe(24)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> str:
        if self._server:
            return self.url
        address = local_ip_address()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args: object) -> None:
                return

            def _respond(self, status: int, payload: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)

            def _json(self, status: int, value: dict[str, str]) -> None:
                self._respond(
                    status,
                    json.dumps(value, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )

            def do_GET(self) -> None:
                if urlsplit(self.path).path != f"/{receiver.token}":
                    self._respond(404, b"Not found", "text/plain; charset=utf-8")
                    return
                self._respond(
                    200,
                    _upload_page(receiver.token),
                    "text/html; charset=utf-8",
                )

            def do_POST(self) -> None:
                if urlsplit(self.path).path != f"/{receiver.token}/upload":
                    self._json(404, {"error": "送信先が見つかりません。"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._json(400, {"error": "送信サイズが不正です。"})
                    return
                if length <= 0 or length > receiver.max_upload_bytes:
                    self._json(413, {"error": "写真のサイズは20MB以下にしてください。"})
                    return
                try:
                    self.connection.settimeout(20)
                    content = self.rfile.read(length)
                    if len(content) != length:
                        raise ValueError("送信データが途中で終わっています。")
                    with Image.open(io.BytesIO(content)) as image:
                        if image.width * image.height > MAX_IMAGE_PIXELS:
                            raise ValueError("画像が大きすぎます（5000万画素まで）。")
                        image.verify()
                    with Image.open(io.BytesIO(content)) as image:
                        suffix = ALLOWED_FORMATS.get(image.format or "")
                    if not suffix:
                        raise ValueError("JPEG、PNG、WebPの写真を送信してください。")
                    destination = receiver.cache_dir / f"phone-{uuid4().hex}{suffix}"
                    destination.write_bytes(content)
                    try:
                        destination.chmod(0o600)
                    except OSError:
                        pass
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    self._json(400, {"error": f"画像を読み込めません: {error}"})
                    return
                self._json(201, {"message": "写真をPCへ送信しました。"})
                if receiver.on_upload:
                    threading.Thread(
                        target=receiver.on_upload,
                        args=(destination,),
                        daemon=True,
                        name="photo-upload-notify",
                    ).start()

        server = _PhotoHTTPServer(("0.0.0.0", 0), Handler)
        self._server = server
        self.url = f"http://{address}:{server.server_port}/{self.token}"
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.25},
            daemon=True,
            name="local-photo-receiver",
        )
        self._thread.start()
        return self.url

    def qr_image(self):
        if not self.running:
            raise RuntimeError("スマホ受信サーバーはまだ起動していません。")
        try:
            import qrcode
        except ImportError as error:
            raise RuntimeError("QRコード表示には qrcode[pil] が必要です。") from error
        return qrcode.make(self.url)

    def stop(self) -> None:
        server, self._server = self._server, None
        if not server:
            return
        server.shutdown()
        server.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        self.url = ""
