"""Boson 라이브 뷰 (구도 확인용, 저장 안 함). Q 또는 창 닫기로 종료.

실행:  py -3.12 live_view.py [스냅샷.png]
  스냅샷 경로를 주면 시작 2초 후 화면 1장을 저장한다.
"""
import sys
import time
from collections import deque

import cv2
from flirpy.camera.boson import Boson

from record_session import PORT, render, check_raw, GREEN, RED, GRAY, WHITE, YELLOW

WINDOW = "Boson live view (Q: quit)"


def main():
    snap_path = sys.argv[1] if len(sys.argv) > 1 else None
    cam = Boson(port=PORT)
    t0 = None
    stamps = deque(maxlen=20)  # 최근 프레임 시각 (카메라 여는 시간이 fps에 섞이지 않게)
    try:
        while True:
            frame = cam.grab()
            if frame is None:
                continue
            now = time.perf_counter()
            if t0 is None:
                t0 = now
            stamps.append(now)
            el = now - t0
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
            ok, desc = check_raw(frame)
            lines = [("LIVE (저장 안 함)", 52, GREEN if ok else RED),
                     (f"{fps:.1f} fps", 30, WHITE),
                     (f"min {int(frame.min())}  max {int(frame.max())}", 22, GRAY),
                     (f"mean {float(frame.mean()):.0f}", 22, GRAY),
                     ("RAW 16-bit OK" if ok else "RAW 아님: " + desc, 20, GREEN if ok else RED),
                     (None, 20, None),
                     ("Q : 종료", 26, GRAY),
                     ("녹화 전에 꼭 닫기", 22, YELLOW)]
            img = render(frame, lines)
            cv2.imshow(WINDOW, img)
            if snap_path and el >= 2.0:
                cv2.imwrite(snap_path, img)
                print(f"스냅샷 저장: {snap_path}  ({desc})", flush=True)
                snap_path = None
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
