#!/usr/bin/env python3
"""Draw the page's derived image assets from the hero.

    python website/make_preview.py    # -> assets/preview.png, assets/favicon-{16,32}.png

Run it when `assets/hero.png` changes, and commit the results. It is NOT part of
`build_website.py`, which is stdlib-only on purpose — this needs Pillow, and these are
assets, not build products.

Two things come out of it, and they travel differently. The **card image** cannot be
carried inside the page: `og:image` is fetched out of band by whoever renders the link, so
a data URI is not an option, and it is uploaded to the site root beside index.html (see
"Hosting" in website/README.md). The **tab icons** are inlined as data URIs like the hero,
so the one-file page keeps its icon when it is opened from disk or arrives by email.

What it does to the art: nothing. The hero is trimmed to its own alpha bounds, scaled to
fit, and centred on the page's paper — no crop through the drawing, no filter, no text
baked over it. 1200x630 is what the card consumers expect at 2:1-ish; the hero is 3:2, so
fitting it whole leaves paper either side, which is what the page does with it too.

The icons are the exception, and only to the *resampling*: see `_shrink()`.
"""

from pathlib import Path

from PIL import Image, ImageFilter

ASSETS = Path(__file__).resolve().parent / "assets"
SRC, OUT = ASSETS / "hero.png", ASSETS / "preview.png"
SIZE = (1200, 630)
PAPER = (247, 244, 234)          # --surface-0, the page's own paper
MARGIN = 0.86                    # of the shorter axis, so the art is not flush to the edge

# Icon size -> the MinFilter radius that draws it, tuned by eye against a magnified
# contact sheet of radii 3..13 at both sizes; see `_shrink()` for what the filter is for.
# Named rather than derived from the decimation factor: they are two numbers somebody
# looked at, and a formula fitted to two points would only dress that up as arithmetic.
# Lower reads washed out, higher smears the wing veins into a blob — 16 turns first.
ICONS = {16: 9, 32: 7}
ICON_PAD = 0.14                  # of the crop's longer side: the filter dilates the ink,
                                 # so tight padding clips the wings it just thickened
DENSITY = 0.02                   # of the densest column, for telling butterfly from trail


def _butterfly(art):
    """The butterfly's own bounds, without the dashed trail that leads up to it.

    `getbbox()` is what the card wants — the whole drawing — but an icon of the whole
    drawing at 16px is a grey smear with a speck at one end. The trail is a hairline of
    dashes and the butterfly is dense line work, so a column's total alpha separates them
    by more than an order of magnitude: walk out from the densest column while the
    density holds. Measured stable at 1%, 2% and 4% (x moves by two pixels), so the
    threshold is not doing delicate work. Derived rather than hardcoded because a fixed
    box goes stale silently the moment the hero is redrawn.

    Resizing the alpha channel to one row is how a column sum is spelled in PIL alone;
    `tobytes()` on an 'L' image is those values, and keeps numpy out of an asset script.
    """
    alpha = art.getchannel("A")
    cols = alpha.resize((art.width, 1), Image.BOX).tobytes()
    peak = max(range(len(cols)), key=cols.__getitem__)
    limit = cols[peak] * DENSITY
    x0 = x1 = peak
    while x0 > 0 and cols[x0 - 1] > limit:
        x0 -= 1
    while x1 < len(cols) - 1 and cols[x1 + 1] > limit:
        x1 += 1
    rows = alpha.crop((x0, 0, x1 + 1, art.height)).resize((1, art.height), Image.BOX).tobytes()
    limit = max(rows) * DENSITY
    ys = [i for i, v in enumerate(rows) if v > limit]
    return x0, ys[0], x1 + 1, ys[-1] + 1


def _shrink(square, n):
    """One icon, at the size a browser will actually paint it.

    The hero is hairline pencil work, and averaging it down loses it: measured on this
    art, a straight LANCZOS to 16px leaves 14 of 256 pixels carrying any ink at all and
    the darkest of them at 3.9:1 on the paper — a blank cream square in the tab. A
    darkest-pixel-wins pass first is the ordinary way to decimate line art, and takes the
    same crop to 149 inked pixels with the strokes at full strength. It changes nothing
    about hero.png; it is a resampling choice, and it applies only here.

    Each size is drawn from the full-resolution crop rather than from a bigger icon. A
    single 32 that the browser scales to 16 for a 1x display re-averages the ink and
    undoes all of the above, which is what `sizes=` on the link tags exists to prevent.
    """
    return square.filter(ImageFilter.MinFilter(ICONS[n])).resize((n, n), Image.LANCZOS)


def preview(art):
    """The link-preview card: the whole drawing, centred on the paper at 1200x630."""
    art = art.crop(art.getbbox() or (0, 0, *art.size))    # its own alpha bounds
    scale = min(SIZE[0] * MARGIN / art.width, SIZE[1] * MARGIN / art.height)
    art = art.resize((round(art.width * scale), round(art.height * scale)), Image.LANCZOS)
    card = Image.new("RGB", SIZE, PAPER)
    card.paste(art, ((SIZE[0] - art.width) // 2, (SIZE[1] - art.height) // 2), art)
    card.save(OUT, optimize=True)
    print(f"wrote {OUT.name} ({SIZE[0]}x{SIZE[1]}, {OUT.stat().st_size:,} bytes)")


def icons(art):
    """The tab icons: the butterfly alone, squared up on the paper, one file per size.

    Paper rather than transparency on purpose — the drawing is dark line work, and on a
    dark browser chrome a transparent icon is an invisible one.
    """
    box = _butterfly(art)
    cut = art.crop(box)
    side = round(max(cut.size) * (1 + ICON_PAD))
    square = Image.new("RGB", (side, side), PAPER)
    square.paste(cut, ((side - cut.width) // 2, (side - cut.height) // 2), cut)
    for n in ICONS:
        path = ASSETS / f"favicon-{n}.png"
        _shrink(square, n).save(path, optimize=True)
        print(f"wrote {path.name} ({n}x{n}, {path.stat().st_size:,} bytes)")


def main():
    art = Image.open(SRC).convert("RGBA")
    preview(art)
    icons(art)


if __name__ == "__main__":
    main()
