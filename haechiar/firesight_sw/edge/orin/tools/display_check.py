#!/usr/bin/env python3
"""Step through window placement so we can see exactly where output is lost.

Each stage shows a differently coloured card with its own number. Note which
stage numbers you actually see, on which screen. Press any key to advance.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from display import list_displays  # noqa: E402


TITLE = "FireSight display check"


def card(number: int, label: str, colour, size=(1920, 1080)) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), colour, dtype=np.uint8)
    cv2.putText(image, f"STAGE {number}", (60, 200), cv2.FONT_HERSHEY_SIMPLEX,
                4.0, (255, 255, 255), 8, cv2.LINE_AA)
    for index, line in enumerate(label.split("\n")):
        cv2.putText(image, line, (60, 330 + index * 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(image, (10, 10), (width - 10, height - 10), (255, 255, 255), 6)
    return image


def hold(image, seconds: float = 6.0) -> None:
    """Pump the Cocoa event loop; a single waitKey is not always enough."""
    for _ in range(int(seconds * 100)):
        cv2.imshow(TITLE, image)
        if cv2.waitKey(10) & 0xFF in (27, ord("q"), 32):
            break


def main() -> int:
    displays = list_displays()
    print("displays:")
    for entry in displays:
        print("  index {} origin ({}, {}) size {}x{}{}".format(
            entry["index"], entry["x"], entry["y"],
            entry["width"], entry["height"], "  <- main" if entry["is_main"] else ""))
    second = next((d for d in displays if not d["is_main"]), None)

    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)

    print("\nSTAGE 1: small window on the MAIN screen (red)")
    cv2.resizeWindow(TITLE, 900, 500)
    cv2.moveWindow(TITLE, 80, 80)
    hold(card(1, "Small red window.\nShould be on the LAPTOP screen.", (40, 40, 200), (900, 500)))

    if second is None:
        print("no second display found; stopping")
        cv2.destroyAllWindows()
        return 0

    print(f"STAGE 2: same window moved to the GLASSES at ({second['x']}, {second['y']}) (green)")
    cv2.moveWindow(TITLE, second["x"] + 100, second["y"] + 100)
    hold(card(2, "Green window, still small.\nShould be in the GLASSES now.", (40, 160, 40), (900, 500)))

    print("STAGE 3: full 1920x1080 window on the glasses, NOT fullscreen (blue)")
    cv2.resizeWindow(TITLE, second["width"], second["height"])
    cv2.moveWindow(TITLE, second["x"], second["y"])
    hold(card(3, "Blue, fills the glasses screen.\nTitle bar may still show.", (200, 80, 40)))

    print("STAGE 4: fullscreen (orange)")
    cv2.setWindowProperty(TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    hold(card(4, "Orange, true fullscreen.\nNo title bar.", (30, 140, 230)))

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("\ndone - tell me which stage numbers you actually saw, and on which screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
