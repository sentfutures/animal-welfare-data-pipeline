#!/usr/bin/env python3
"""Build the handoff page: both corpora, one self-contained HTML file.

    python website/build_website.py --dad-run outputs/dad/runs/<run_id> \\
                                  --sdf-run outputs/sdf/runs/<run_id>
    # -> website/index.html

``--run`` still works as an alias for ``--dad-run``, which keeps the command printed in
the page's own "Running it yourself" block true. ``--sdf-run`` is optional: without it
the document corpus's column and section say so instead of showing figures.

The page is one self-contained HTML file: no external CSS, JS, fonts or images, so it
opens offline from the filesystem, survives an artifact host's CSP, and publishes to
GitHub Pages as-is. The generator is stdlib only and imports nothing from viewer/ or
shared/, so it builds in an environment where the pipeline's own dependencies are not
installed.

For a HOSTED build, name the URL it will be served from — that adds the link-preview tags
and copies the card image out beside the page, the one file that travels with it:

    python website/build_website.py --dad-run <run> --sdf-run <run> \\
                                  --site-url https://<host>/
    # -> website/index.html + website/preview.png

Every build is `noindex` either way; see "Hosting" in website/README.md.
"""

import base64
import shutil
import sys
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from website import common as C  # noqa: E402
from website import page  # noqa: E402

WEBSITE_DIR = Path(__file__).resolve().parent
CONTENT = [WEBSITE_DIR / "content_page.md", WEBSITE_DIR / "content_dad.md",
           WEBSITE_DIR / "content_sdf.md"]
HERO = WEBSITE_DIR / "assets" / "hero.png"
# assets/sf.png is deliberately not read: the maker's mark used to sit inside "A project by
# Sentient Futures", where it was a picture of a name printed 4px to its right. The footer's
# only marks now are the two that identify a destination the reader has not seen yet.
# The link preview's image — the hero on the page's own paper, drawn by make_preview.py.
# The ONE file that travels beside index.html, and only for a hosted build: a card renderer
# fetches og:image over the network, so this is the one picture the page cannot carry.
PREVIEW = WEBSITE_DIR / "assets" / "preview.png"
# The tab icons, one PNG per size because each is decimated for the size it names — see
# make_preview.py, which draws these and the card from the same hero. Inlined like the
# hero, so they are NOT among the files a deploy has to carry.
FAVICONS = [(px, WEBSITE_DIR / "assets" / f"favicon-{px}.png") for px in (16, 32)]


def data_uri(path, mime="image/png"):
    """An image as a data: URI, or "" if it is not there.

    Inlining is not an optimisation here, it is the format: the page has to be one file
    that opens offline, so the only picture it can carry is one encoded into it. The
    source art lives in website/assets/ and never ships next to the HTML.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return ""
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def main():
    args = C.cli_parser(__doc__).parse_args()
    if not args.dad_run:
        C.die("--dad-run is required")
    out_dir = Path(args.out_dir or WEBSITE_DIR)
    kwargs = page.load_inputs(args.content or CONTENT, dad_run=args.dad_run,
                              sdf_run=args.sdf_run)
    hero = data_uri(HERO)
    icons = [(px, data_uri(p)) for px, p in FAVICONS]
    site_url = args.site_url or ""
    # A hosted build gets the card image without being asked twice: the default names the
    # file this script copies out beside index.html, resolved against the page's own URL
    # (urljoin, so both ".../" and ".../index.html" land on ".../preview.png").
    preview_url = args.preview_url or (urljoin(site_url, PREVIEW.name) if site_url else "")
    html = page.build(example=args.example, sdf_example=args.sdf_example, illustration=hero,
                      icons=icons, site_url=site_url,
                      preview_url=preview_url, **kwargs)
    if site_url and not args.preview_url and PREVIEW.exists():
        shutil.copyfile(PREVIEW, out_dir / PREVIEW.name)
    audit = (kwargs.get("dad_inputs") or {}).get("audit") or {}
    C.write(out_dir / "index.html", html,
            label=f"{C.editorial_words(html):,} words of prose · "
                  f"hero={'inlined' if hero else 'NO'} · "
                  f"icons={sum(1 for _, u in icons if u)}/{len(icons)} · "
                  f"dad n={audit.get('n_prompts')} "
                  f"delivery={'yes' if audit.get('delivery') else 'NO'} "
                  f"showcase={'yes' if audit.get('showcase') else 'NO'} "
                  f"sdf={'yes' if kwargs.get('sdf_inputs') else 'NO'} "
                  f"preview={preview_url or 'no'}")


if __name__ == "__main__":
    main()
