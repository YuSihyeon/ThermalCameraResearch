# 거리 센서 어댑터 계약

`publish_depth.py`는 아직 결정되지 않은 거리 센서의 물리 SDK와 FireSight의
동일 호스트 TCP 거리 프로토콜 사이를 연결한다. 특정 센서 백엔드는 포함하지
않는다. Jetson Orin Nano에서 사용할 센서를 고른 뒤 해당 SDK 어댑터와 실제
카메라 등록 보정을 별도로 구현하고 검증해야 한다.

## 어댑터 API

`--adapter package.module:factory`로 팩토리를 지정한다. 팩토리와 반환 객체의
계약은 다음과 같다.

```python
def factory(calibration, options):
    return Adapter(...)

class Adapter(object):
    def open(self):
        # SDK와 장치를 연다.
        pass

    def read(self, timeout_s):
        # timeout_s 안에 아래 샘플 하나 또는 None을 반환한다.
        pass

    def close(self):
        # 부분 초기화나 예외 뒤에도 안전하게 호출할 수 있어야 한다.
        pass
```

`read()`는 제한 시간 안에 `None` 또는 다음 두 형식 중 정확히 하나를 반환해야
한다.

```python
{"timestamp_s": acquired_s, "depth_m": registered_h_by_w_float_array}
{"timestamp_s": acquired_s, "points_xyz_m": sensor_xyz_n_by_3_array}
```

- `depth_m`는 `--height`, `--width`로 정한 실시간 출력 `[H, W]`에 이미 등록된
  부동소수점 카메라 Z축 깊이(미터)다. 정수형 SDK 원시값은 어댑터에서 명시적으로
  단위와 좌표계를 변환해야 한다.
- `points_xyz_m`는 센서 좌표계의 XYZ 미터 점이다. 퍼블리셔가 보정의 센서→카메라
  `rotation`, `translation_m`, 카메라 행렬과 왜곡계수로 투영한다. 결과값은
  유클리드 거리가 아니라 카메라 Z축 깊이다.
- `timestamp_s`는 SDK 측정값을 실제로 취득한 순간 또는 SDK 호출이 반환된 즉시
  어댑터가 같은 호스트의 `time.perf_counter()`로 기록한다. 퍼블리셔는 빠진
  시각을 만들거나 전송 시각으로 다시 찍지 않는다. Windows Python 3.10 이상과
  Linux에서 이 값은 같은 호스트 프로세스 사이에 비교할 수 있다.
- 각 연결에서 시각은 엄격히 증가해야 한다. `read(None)` 형태의 무기한 대기는
  허용하지 않으며 `read(timeout_s)`는 전달받은 제한 시간을 지켜야 한다.

퍼블리셔는 한 번에 샘플 하나만 보유하고 소켓 연결과 전송에 제한 시간을 둔다.
연결 실패 시 제한된 횟수만 재시도한다. 전송이 실패한 샘플은 재연결 뒤 다시 보내지
않고 폐기하며, 어댑터에서 최신 측정값을 새로 읽는다. 연속 전송 실패도 제한된
횟수 뒤 종료한다. 새 연결에서 frame ID는 0부터 다시 시작한다. 모든 종료 및 실패
경로에서 소켓과 어댑터를 닫는다. 전송 provenance는 항상 `measured`다. 합성
데이터는 실제 센서 어댑터처럼 포장하지 말고 명시적인 replay 테스트에서만
`synthetic`으로 사용한다.

물리 SDK 어댑터의 `read(timeout_s)`는 SDK 내부 버퍼가 누적되지 않도록 호출 시점의
최신 측정값을 반환해야 한다. 전송이나 재연결 동안 쌓인 오래된 측정값을 차례로
재생해서는 안 된다.

## 실행

```bash
python3 publish_depth.py \
  --adapter my_sensor.fire_camera:factory \
  --adapter-option serial=ABC123 \
  --adapter-option mode=wide \
  --calibration calib/sensor_to_camera.json \
  --host 127.0.0.1 --port 38517 \
  --width 320 --height 240
```

`--width`와 `--height`의 기본값은 실시간 TEED 기본 출력과 같은 320×240이다.
XYZ 점은 보정 파일의 원래 영상 크기에서 이 출력 크기로 pixel-center intrinsics를
조정해 투영한다. `depth_m` 배열은 이미 이 출력 크기와 같은 카메라 등록 맵이어야
하며, 퍼블리셔는 시야각이나 미측정 hole을 훼손할 수 있는 암묵적 resize를 하지
않고 크기가 다르면 실패한다.

`--max-frames N`은 설치 점검이나 유한 시험에 사용한다. `0`은 계속 실행한다.
`--read-timeout-ms`, `--connect-timeout-ms`, `--send-timeout-ms`,
`--reconnect-delay-ms`, `--reconnect-attempts`,
`--max-consecutive-send-failures`로 실패 경계를 조정할 수 있다. 같은 호스트 시계
계약을 지키기 위해 `--host`는 `127.0.0.1` 또는 `localhost`만 허용한다.

어댑터 예시는 SDK가 반환한 원시 정수 깊이를 미터로 바꾸는 지점을 명시해야 한다.

```python
def read(self, timeout_s):
    raw_mm = self.device.wait_for_depth(timeout_ms=int(timeout_s * 1000.0))
    if raw_mm is None:
        return None
    acquired_s = time.perf_counter()  # SDK 반환 직후, 같은 호스트 시계
    registered_m = self.register_to_camera(raw_mm).astype(np.float32) * 0.001
    return {"timestamp_s": acquired_s, "depth_m": registered_m}
```

이 골격만으로 물리 정합 정확성이나 실측 센서 연결이 증명되지는 않는다. 센서 선택
후 공급사 SDK 설치, 동기화 특성 확인, 카메라 내·외부 보정, 실제 장치 프레임과
`calibration_id` 검증을 완료해야 장치 검증을 통과할 수 있다.
