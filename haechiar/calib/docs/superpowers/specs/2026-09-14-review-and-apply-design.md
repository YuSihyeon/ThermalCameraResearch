# Boson calibration review and application

The program owns the final consent interaction: input images or capture, solve, open a standalone report, ask yes/no/quit. A rejection returns to the input prompt and a fresh attempt directory. EOF and cancellation never imply yes. Capture closes before reviewing, and the application source opens only after explicit consent.

`workflow.calibrate_dataset` writes the existing YAML/debug data and an HTML review with native-image-derived overlays, per-view error, coverage counts, mean spatial reprojection residuals, numerical values and target status. Heatmap empty cells mean no measurements. The report is descriptive; numerical heuristics and provisional target status are visible to the human decision maker.

`wizard.review_and_apply` verifies report/calibration hashes, validates camera parameters, names the selected device or topics, opens the report and records the decision. It verifies the hashes again and passes a frozen in-memory calibration to the runtime. Each attempt and decision is retained. Structural invalidity prevents application; quality warnings require human judgment through the same explicit consent prompt.

`application.Rectifier` caches OpenCV undistortion maps for the native 320×256 image and preserves mono8/mono16 dtype. USB application displays original and corrected images. ROS application publishes corrected Image plus CameraInfo with matching acquisition stamp/frame_id, zero distortion and P=[K|0]. Topic identity checks prevent a feedback loop. Frame loss and errors close resources; only successfully shown/published frames count as applied.

Application changes host processing while the program runs. It does not write Boson firmware, EEPROM, or another program's camera settings. A USB index cannot prove camera identity; the approval prompt requires checking the same physical camera, lens, focus and image mode. Downstream ROS consumers subscribe to the corrected output topics. Each rerun of a saved result requires fresh approval.

The root launchers default to the wizard. Batch intrinsic/capture, self-check, extrinsic, project and verify-config remain available. Existing design documents are retained as historical records and linked from README. Validation covers decision transitions, report integrity and rendering, synthetic numerical correction, native dtype, resource cleanup and real DDS transport in Humble CI. Real Boson/target accuracy still requires physical equipment.

Reference APIs: [OpenCV rectification](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html), [ROS CameraInfo definition](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CameraInfo.msg).
