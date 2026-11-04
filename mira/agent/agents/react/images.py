_IMG_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "gif": "image/gif", "webp": "image/webp"}
def _rewrite_file_images_to_data_urls(messages):
    import base64 as _b64, os as _os
    out = []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            out.append(m); continue
        new_c = []
        for part in c:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = ((part.get("image_url") or {}).get("url") or "")
                path = None
                if url.startswith("file://"):
                    path = url[len("file://"):]
                elif url and not url.startswith(("http://", "https://", "data:")):
                    path = url
                if path and _os.path.exists(path):
                    try:
                        with open(path, "rb") as fh:
                            data = fh.read()
                        ext = _os.path.splitext(path)[1].lstrip(".").lower()
                        mime = _IMG_MIME.get(ext, "image/png")
                        b64 = _b64.b64encode(data).decode("ascii")
                        new_c.append({"type": "image_url",
                                      "image_url": {"url": f"data:{mime};base64,{b64}"}})
                        continue
                    except Exception as e:
                        print(f"[external-api] failed to inline image {path[:80]!r}: {e}")
            new_c.append(part)
        mm = dict(m); mm["content"] = new_c
        out.append(mm)
    return out
