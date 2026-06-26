"""Generate simple placeholder hallway images for vision_core smoke tests."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_images"


def _font(size: int = 18):
    for name in ("Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def hallway_base(title: str) -> Image.Image:
    img = Image.new("RGB", (640, 480), (210, 205, 195))
    draw = ImageDraw.Draw(img)
    # Floor
    draw.rectangle((0, 300, 640, 480), fill=(170, 165, 155))
    # Walls / vanishing perspective
    draw.polygon([(0, 0), (640, 0), (520, 300), (120, 300)], fill=(225, 220, 210))
    draw.line([(120, 300), (520, 300)], fill=(140, 135, 125), width=3)
    draw.text((16, 16), title, fill=(30, 30, 30), font=_font(20))
    return img, draw


def empty_hallway() -> Image.Image:
    img, draw = hallway_base("Empty hallway — clear path")
    draw.text((180, 360), "CLEAR CORRIDOR", fill=(40, 120, 40), font=_font(22))
    return img


def crowd_blocking() -> Image.Image:
    img, draw = hallway_base("Hallway — crowd blocking center")
    # Crowd blob in center
    draw.ellipse((250, 220, 390, 380), fill=(80, 80, 90))
    draw.ellipse((220, 250, 300, 360), fill=(90, 85, 95))
    draw.ellipse((340, 250, 420, 360), fill=(90, 85, 95))
    draw.text((150, 400), "CROWD BLOCKING PATH", fill=(160, 30, 30), font=_font(20))
    return img


def commotion_clear_right() -> Image.Image:
    img, draw = hallway_base("Commotion ahead — clear path on right")
    # Commotion / smoke ahead left-center
    draw.ellipse((180, 200, 340, 340), fill=(120, 110, 100))
    draw.ellipse((200, 180, 320, 260), fill=(150, 140, 130))
    draw.text((170, 150), "COMMOTION / SMOKE", fill=(120, 50, 20), font=_font(18))
    # Clear lane on right
    draw.rectangle((430, 310, 610, 470), fill=(190, 220, 190))
    draw.text((440, 330), "CLEAR ->", fill=(20, 100, 20), font=_font(22))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {
        "01_empty_hallway.png": empty_hallway,
        "02_crowd_blocking.png": crowd_blocking,
        "03_commotion_clear_right.png": commotion_clear_right,
    }
    for name, factory in cases.items():
        path = OUT / name
        factory().save(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
