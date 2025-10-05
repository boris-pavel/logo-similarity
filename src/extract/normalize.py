"""Utilities for normalizing logo imagery into a consistent format."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageChops

try:  # pragma: no cover - optional dependency
    import cairosvg  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cairosvg = None  # type: ignore[assignment]

_SVG_MIME_TYPES = {"image/svg+xml", "image/svg", "text/svg"}


def to_png_rgba(image_bytes: bytes, mime_hint: str | None) -> Image.Image:
    """Return a Pillow image in RGBA mode, rasterizing SVG input when possible."""
    if not image_bytes:
        raise ValueError("Empty image payload cannot be normalized")

    data = image_bytes
    mime = (mime_hint or "").lower()
    is_svg = mime in _SVG_MIME_TYPES or _looks_like_svg(image_bytes)

    if is_svg and cairosvg is not None:
        try:
            data = cairosvg.svg2png(bytestring=image_bytes)  # type: ignore[attr-defined]
        except Exception:
            data = image_bytes

    with Image.open(BytesIO(data)) as img:
        return img.convert("RGBA")


def trim_and_square(
    img: Image.Image, pad: int = 8, bg: tuple[int, int, int, int] = (0, 0, 0, 0)
) -> Image.Image:
    """Trim uniform borders and place the result on a square transparent canvas."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    bbox = _alpha_bbox(img) or _color_bbox(img)
    trimmed = img.crop(bbox) if bbox else img.copy()

    content_w, content_h = trimmed.size
    if content_w == 0 or content_h == 0:
        return trimmed

    pad = max(0, pad)
    target_side = max(content_w, content_h) + (pad * 2)
    canvas = Image.new("RGBA", (target_side, target_side), color=bg)
    offset_x = (target_side - content_w) // 2
    offset_y = (target_side - content_h) // 2
    canvas.paste(trimmed, (offset_x, offset_y), mask=trimmed)
    return canvas


def resize_logo(img: Image.Image, size: int = 256) -> Image.Image:
    """Return *img* resized to a square size-by-size image with antialiasing."""
    if size <= 0:
        raise ValueError("Size must be a positive integer")

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    resample_attr = getattr(Image, "Resampling", None)
    resample_filter = getattr(resample_attr, "LANCZOS", None) if resample_attr else None
    if resample_filter is None:
        resample_filter = getattr(Image, "LANCZOS", Image.BICUBIC)

    return img.resize((size, size), resample_filter)


def normalize_logo(image_bytes: bytes, mime_hint: str | None) -> Image.Image:
    """Full normalization pipeline returning a 256x256 RGBA Pillow image."""
    img = to_png_rgba(image_bytes, mime_hint)
    img = trim_and_square(img)
    return resize_logo(img, size=256)


def _looks_like_svg(image_bytes: bytes) -> bool:
    snippet = image_bytes[:1024].lstrip().lower()
    return snippet.startswith(b"<svg") or (
        snippet.startswith(b"<?xml") and b"<svg" in snippet
    )

def _alpha_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    if "A" not in img.getbands():
        return None
    alpha = img.getchannel("A")
    return alpha.getbbox()


def _color_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    try:
        bg_color = img.getpixel((0, 0))
    except Exception:
        return None
    background = Image.new(img.mode, img.size, bg_color)
    diff = ImageChops.difference(img, background)
    return diff.getbbox()
