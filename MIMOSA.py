import streamlit as st
import cv2
import numpy as np
import pandas as pd
import math
import tempfile
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from streamlit_image_coordinates import streamlit_image_coordinates


# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="Mimosa Motion Analyzer",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Mimosa Motion Analyzer")

st.write(
    "AI API 없이, 색상 기반 영역(ROI) 분석만으로 "
    "미모사 잎의 움직임을 추적합니다. "
    "(잎 끝 위치 추적 + 초록색 면적비 + 움직임량, 3중 신호로 분석)"
)


# ============================================================
# 2. 영상 업로드
# ============================================================

uploaded_video = st.file_uploader(
    "미모사 영상을 업로드하세요.",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_video is None:
    st.info("먼저 미모사 영상을 업로드하세요.")
    st.stop()


# ============================================================
# 3. 업로드 영상을 임시 파일로 저장
# ============================================================

video_bytes = uploaded_video.getvalue()

input_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
input_video.write(video_bytes)
input_video.close()

video_path = input_video.name


# ============================================================
# 4. 영상 정보 확인
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("영상을 열 수 없습니다.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0 or math.isnan(fps):
    fps = 30.0

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ok, first_frame = cap.read()
cap.release()

if not ok:
    st.error("첫 번째 프레임을 읽을 수 없습니다.")
    st.stop()

duration = frame_count / fps


# ============================================================
# 5. 영상 정보 표시
# ============================================================

st.subheader("영상 정보")

col1, col2, col3, col4 = st.columns(4)
col1.metric("해상도", f"{width} × {height}")
col2.metric("FPS", f"{fps:.2f}")
col3.metric("프레임 수", f"{frame_count}")
col4.metric("영상 길이", f"{duration:.2f}초")

st.video(video_bytes)


# ============================================================
# 6. 1단계 — P1 설정 (거리 측정 기준점)
# ============================================================

st.subheader("1단계 — P1 설정 (거리 측정 기준점)")

st.write("아래 이미지를 **클릭**해서 P1 좌표를 설정하세요. (보통 잎이 붙어있는 줄기 쪽)")

if "p1_x" not in st.session_state:
    st.session_state.p1_x = width // 2
if "p1_y" not in st.session_state:
    st.session_state.p1_y = height // 2

p1_preview = first_frame.copy()
cv2.circle(p1_preview, (st.session_state.p1_x, st.session_state.p1_y), 10, (0, 0, 255), -1)
cv2.putText(
    p1_preview, "P1",
    (st.session_state.p1_x + 10, st.session_state.p1_y - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
)
p1_preview_rgb = cv2.cvtColor(p1_preview, cv2.COLOR_BGR2RGB)

p1_click = streamlit_image_coordinates(p1_preview_rgb, key="p1_picker")

if p1_click is not None:
    cx = max(0, min(width - 1, int(p1_click["x"])))
    cy = max(0, min(height - 1, int(p1_click["y"])))
    if (cx, cy) != (st.session_state.p1_x, st.session_state.p1_y):
        st.session_state.p1_x = cx
        st.session_state.p1_y = cy
        st.rerun()

st.write(f"현재 P1 좌표: **({st.session_state.p1_x}, {st.session_state.p1_y})**")

with st.expander("🔧 P1 좌표 직접 입력 / 미세 조정"):
    col1, col2 = st.columns(2)
    with col1:
        manual_x = st.number_input(
            "P1 X", min_value=0, max_value=width - 1,
            value=st.session_state.p1_x, step=1, key="manual_p1_x"
        )
    with col2:
        manual_y = st.number_input(
            "P1 Y", min_value=0, max_value=height - 1,
            value=st.session_state.p1_y, step=1, key="manual_p1_y"
        )
    if st.button("이 좌표로 P1 설정"):
        st.session_state.p1_x = int(manual_x)
        st.session_state.p1_y = int(manual_y)
        st.rerun()

BASE_POINT = (int(st.session_state.p1_x), int(st.session_state.p1_y))


# ============================================================
# 7. 2단계 — ROI(관심 영역) 설정 — 원하는 모양으로 자유롭게 클릭
# ============================================================

st.subheader("2단계 — 관심 영역(ROI) 설정")

st.write(
    """
    잎이 움직이는 범위를 **원하는 모양대로 점을 찍어서** 감싸주세요.
    점을 3개 이상 찍은 다음 **"다각형 완성"** 버튼을 누르면 영역이 확정됩니다.
    """
)

if "roi_points" not in st.session_state:
    st.session_state.roi_points = []

if "roi_closed" not in st.session_state:
    st.session_state.roi_closed = False

col_undo, col_reset, col_close, col_status = st.columns([1, 1, 1, 3])

with col_undo:
    if st.button("↩ 마지막 점 취소", disabled=st.session_state.roi_closed):
        if st.session_state.roi_points:
            st.session_state.roi_points.pop()
            st.rerun()

with col_reset:
    if st.button("↺ 전체 초기화"):
        st.session_state.roi_points = []
        st.session_state.roi_closed = False
        st.rerun()

with col_close:
    if st.button(
        "✅ 다각형 완성",
        disabled=(len(st.session_state.roi_points) < 3 or st.session_state.roi_closed)
    ):
        st.session_state.roi_closed = True
        st.rerun()

with col_status:
    if st.session_state.roi_closed:
        st.success(f"ROI 설정 완료 (점 {len(st.session_state.roi_points)}개)")
    else:
        st.info(f"현재 {len(st.session_state.roi_points)}개 점 찍음 — 이미지를 클릭해서 점을 추가하세요.")

roi_preview = p1_preview.copy()

pts = st.session_state.roi_points

for i, pt in enumerate(pts):
    cv2.circle(roi_preview, pt, 5, (255, 0, 0), -1)
    cv2.putText(roi_preview, str(i + 1), (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

if len(pts) >= 2:
    line_color = (0, 255, 255) if st.session_state.roi_closed else (0, 200, 255)
    for i in range(len(pts) - 1):
        cv2.line(roi_preview, pts[i], pts[i + 1], line_color, 2)
    if st.session_state.roi_closed:
        cv2.line(roi_preview, pts[-1], pts[0], line_color, 2)

roi_preview_rgb = cv2.cvtColor(roi_preview, cv2.COLOR_BGR2RGB)

roi_click = streamlit_image_coordinates(roi_preview_rgb, key="roi_picker")

if roi_click is not None and not st.session_state.roi_closed:
    cx = max(0, min(width - 1, int(roi_click["x"])))
    cy = max(0, min(height - 1, int(roi_click["y"])))

    new_point = (cx, cy)

    if not st.session_state.roi_points or st.session_state.roi_points[-1] != new_point:
        st.session_state.roi_points.append(new_point)
        st.rerun()


ROI = None
ROI_POLY_MASK = None

if st.session_state.roi_closed and len(st.session_state.roi_points) >= 3:

    pts_arr = np.array(st.session_state.roi_points, dtype=np.int32)

    x1, y1 = pts_arr[:, 0].min(), pts_arr[:, 1].min()
    x2, y2 = pts_arr[:, 0].max(), pts_arr[:, 1].max()

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))

    if x2 - x1 >= 5 and y2 - y1 >= 5:

        ROI = (x1, y1, x2, y2)

        # 다각형 마스크: ROI 크롭 좌표계 기준으로 생성
        local_pts = pts_arr - np.array([x1, y1])
        ROI_POLY_MASK = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(ROI_POLY_MASK, [local_pts], 255)


# ============================================================
# 8. 3단계 — 색상(HSV) 임계값 보정
# ============================================================

st.subheader("3단계 — 초록색 인식 범위 보정")

st.write("슬라이더를 움직이며 아래 미리보기에서 잎이 하얗게(=인식됨) 나오도록 맞추세요.")

col1, col2, col3 = st.columns(3)

with col1:
    h_range = st.slider("Hue (색상)", 0, 179, (30, 90))
with col2:
    s_range = st.slider("Saturation (채도)", 0, 255, (30, 255))
with col3:
    v_range = st.slider("Value (명도)", 0, 255, (30, 255))

HSV_LOWER = (h_range[0], s_range[0], v_range[0])
HSV_UPPER = (h_range[1], s_range[1], v_range[1])


def compute_green_mask(frame_bgr, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def find_farthest_point_in_mask(mask, ref_point_local):
    """
    mask(ROI 좌표계) 안의 가장 큰 초록 덩어리 윤곽선에서
    ref_point_local(ROI 좌표계)로부터 가장 먼 점을 찾는다.
    반환: (x, y) ROI 좌표계, 없으면 None
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 5:
        return None

    pts = largest.reshape(-1, 2)
    rx, ry = ref_point_local

    dists = np.hypot(pts[:, 0] - rx, pts[:, 1] - ry)
    idx = int(np.argmax(dists))

    return (float(pts[idx][0]), float(pts[idx][1]))


if ROI is not None:
    x1, y1, x2, y2 = ROI
    preview_crop = first_frame[y1:y2, x1:x2]
    preview_mask = compute_green_mask(preview_crop, HSV_LOWER, HSV_UPPER)

    if ROI_POLY_MASK is not None:
        preview_mask = cv2.bitwise_and(preview_mask, ROI_POLY_MASK)

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            cv2.cvtColor(preview_crop, cv2.COLOR_BGR2RGB),
            caption="ROI 원본 (사각형은 다각형을 감싸는 바운딩 박스)",
            use_container_width=True
        )
    with col2:
        st.image(preview_mask, caption="인식된 초록색 마스크 (다각형 안쪽만, 흰색 = 인식됨)", use_container_width=True)
else:
    st.warning("먼저 2단계에서 ROI를 지정하세요 (점 3개 이상 찍고 '다각형 완성').")


# ============================================================
# 9. 4단계 — 자극 시점 / 반응 감지 옵션
# ============================================================

st.subheader("4단계 — 자극 시점 및 반응 감지 옵션")

stimulus_input = st.text_input(
    "자극을 준 시점(초), 쉼표로 여러 개 입력 가능 (예: 60, 180)",
    value="",
    help="비워두면 자동 반응 감지 없이 그래프만 보여줍니다."
)

change_threshold_pct = st.slider(
    "반응 감지 민감도 (기준선 대비 변화율, %)",
    min_value=1, max_value=50, value=5
) / 100.0

stimulus_times = []
if stimulus_input.strip():
    for tok in stimulus_input.split(","):
        tok = tok.strip()
        if tok:
            try:
                stimulus_times.append(float(tok))
            except ValueError:
                pass


def detect_response_events(times, values, stimulus_time,
                            baseline_window_sec=30, change_threshold_ratio=0.05,
                            direction="auto"):
    """
    friend's detect_response_events를 일반화한 버전.
    direction: 'decrease' (감소만 반응으로 인정), 'increase' (증가만),
               'auto' (증가/감소 모두 절대 변화량 기준으로 인정)
    """
    times = np.asarray(times)
    values = np.asarray(values)

    baseline_mask = (times >= stimulus_time - baseline_window_sec) & (times < stimulus_time)
    if baseline_mask.sum() < 3:
        baseline_mask = times < stimulus_time
    baseline = np.mean(values[baseline_mask]) if baseline_mask.sum() > 0 else values[0]

    after_mask = times >= stimulus_time
    t_after, v_after = times[after_mask], values[after_mask]

    if len(v_after) == 0:
        return None

    threshold = abs(baseline) * change_threshold_ratio

    if direction == "decrease":
        crossed = np.where(v_after < baseline - threshold)[0]
        extreme_idx = int(np.argmin(v_after))
    elif direction == "increase":
        crossed = np.where(v_after > baseline + threshold)[0]
        extreme_idx = int(np.argmax(v_after))
    else:
        crossed = np.where(np.abs(v_after - baseline) >= threshold)[0]
        extreme_idx = int(np.argmax(np.abs(v_after - baseline)))

    onset_time = t_after[crossed[0]] - stimulus_time if len(crossed) > 0 else None

    peak_time = t_after[extreme_idx] - stimulus_time
    peak_value = v_after[extreme_idx]
    change_percent = ((peak_value - baseline) / baseline * 100) if baseline != 0 else None

    recovery_threshold = threshold * 0.5
    post_peak_mask = t_after >= t_after[extreme_idx]
    recovered = np.where(np.abs(v_after[post_peak_mask] - baseline) <= recovery_threshold)[0]
    recovery_time = (
        t_after[post_peak_mask][recovered[0]] - stimulus_time
        if len(recovered) > 0 else None
    )

    return {
        "baseline": baseline,
        "onset_latency_sec": onset_time,
        "peak_time_sec": peak_time,
        "change_percent": change_percent,
        "recovery_time_sec": recovery_time,
    }


# ============================================================
# 10. 분석 시작
# ============================================================

st.subheader("5단계 — 분석 시작")

analyze_button = st.button(
    "🌱 분석 시작",
    type="primary",
    disabled=(ROI is None)
)

if ROI is None:
    st.warning("분석을 시작하려면 먼저 2단계에서 ROI를 지정하세요.")


if analyze_button:

    progress_bar = st.progress(0)
    status = st.empty()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("영상을 열 수 없습니다.")
        st.stop()

    output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video.name, fourcc, fps, (width, height))

    x1, y1, x2, y2 = ROI
    p1_local = (BASE_POINT[0] - x1, BASE_POINT[1] - y1)

    records = []
    frame_index = 0
    prev_gray_roi = None

    last_tip_local = None

    while True:

        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_index / fps

        crop = frame[y1:y2, x1:x2]

        # ------------------------------------------------
        # 방법 A: 초록색 마스크 + 잎 끝(가장 먼 점) 탐색
        # ------------------------------------------------

        mask = compute_green_mask(crop, HSV_LOWER, HSV_UPPER)

        if ROI_POLY_MASK is not None:
            mask = cv2.bitwise_and(mask, ROI_POLY_MASK)

        green_pixels = cv2.countNonZero(mask)
        total_pixels = int(cv2.countNonZero(ROI_POLY_MASK)) if ROI_POLY_MASK is not None else (mask.shape[0] * mask.shape[1])
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else np.nan

        tip_local = find_farthest_point_in_mask(mask, p1_local)

        if tip_local is not None:
            last_tip_local = tip_local
        elif last_tip_local is not None:
            tip_local = last_tip_local  # 이번 프레임에 못 찾으면 마지막 위치 유지

        # ------------------------------------------------
        # 방법 B: 프레임 차분 기반 움직임량 (ROI 내부만)
        # ------------------------------------------------

        gray_roi = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_roi_blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        if prev_gray_roi is not None:
            diff = cv2.absdiff(prev_gray_roi, gray_roi_blur)
            _, diff_thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            motion_score = cv2.countNonZero(diff_thresh) / diff_thresh.size
        else:
            motion_score = 0.0

        prev_gray_roi = gray_roi_blur

        # ------------------------------------------------
        # 거리 계산 (전체 프레임 좌표 기준)
        # ------------------------------------------------

        if tip_local is not None:
            tip_x = tip_local[0] + x1
            tip_y = tip_local[1] + y1
            distance_px = math.hypot(tip_x - BASE_POINT[0], tip_y - BASE_POINT[1])
        else:
            tip_x, tip_y, distance_px = np.nan, np.nan, np.nan

        # ------------------------------------------------
        # 시각화
        # ------------------------------------------------

        cv2.polylines(
            frame,
            [np.array(st.session_state.roi_points, dtype=np.int32)],
            isClosed=True,
            color=(0, 255, 255),
            thickness=1
        )
        cv2.circle(frame, BASE_POINT, 8, (0, 0, 255), -1)
        cv2.putText(frame, "P1", (BASE_POINT[0] + 10, BASE_POINT[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if not math.isnan(tip_x):
            tip_int = (int(round(tip_x)), int(round(tip_y)))
            cv2.circle(frame, tip_int, 8, (0, 165, 255), -1)
            cv2.line(frame, BASE_POINT, tip_int, (255, 255, 0), 2)
            cv2.putText(frame, "TIP", (tip_int[0] + 10, tip_int[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.putText(frame, f"Time: {timestamp:.2f}s", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Green: {green_ratio*100:.1f}%", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)

        records.append({
            "frame": frame_index,
            "time_s": timestamp,
            "green_ratio": green_ratio,
            "motion_score": motion_score,
            "tip_x": tip_x,
            "tip_y": tip_y,
            "distance_px": distance_px,
        })

        frame_index += 1

        progress_bar.progress(min(frame_index / frame_count, 1.0))
        status.text(f"{frame_index} / {frame_count} 프레임 분석 중...")

    cap.release()
    writer.release()

    status.text(f"{frame_index}개 프레임 분석 완료")


    # ========================================================
    # 11. 데이터프레임 + 스무딩
    # ========================================================

    df = pd.DataFrame(records)

    df["distance_px"] = df["distance_px"].interpolate(limit_direction="both")
    df["tip_x"] = df["tip_x"].interpolate(limit_direction="both")
    df["tip_y"] = df["tip_y"].interpolate(limit_direction="both")

    def smooth(series, max_window=31, polyorder=3):
        n = len(series)
        window = min(max_window, n - (1 - n % 2))
        if window >= 5 and window > polyorder:
            if window % 2 == 0:
                window -= 1
            return savgol_filter(series, window_length=window, polyorder=polyorder)
        return series.values

    df["green_ratio_smooth"] = smooth(df["green_ratio"], max_window=51, polyorder=3)
    df["motion_score_smooth"] = smooth(df["motion_score"], max_window=31, polyorder=2)
    df["distance_smooth"] = smooth(df["distance_px"], max_window=31, polyorder=3)

    initial_distance = float(df["distance_smooth"].iloc[0])
    df["relative_distance_px"] = df["distance_smooth"] - initial_distance

    df["dt"] = df["time_s"].diff()
    df["distance_velocity_px_s"] = df["relative_distance_px"].diff() / df["dt"]
    df["distance_velocity_px_s"] = df["distance_velocity_px_s"].interpolate(limit_direction="both")


    # ========================================================
    # 12. 결과 표시
    # ========================================================

    st.success("분석이 완료되었습니다.")

    st.subheader("6단계 — 분석 결과")

    if stimulus_times:

        results_summary = []

        for stim_t in stimulus_times:

            res_green = detect_response_events(
                df["time_s"], df["green_ratio_smooth"], stim_t,
                change_threshold_ratio=change_threshold_pct, direction="decrease"
            )
            res_dist = detect_response_events(
                df["time_s"], df["distance_smooth"], stim_t,
                change_threshold_ratio=change_threshold_pct, direction="auto"
            )

            if res_green:
                results_summary.append({
                    "stimulus_time_s": stim_t,
                    "signal": "green_ratio",
                    **res_green
                })
            if res_dist:
                results_summary.append({
                    "stimulus_time_s": stim_t,
                    "signal": "distance_px",
                    **res_dist
                })

        if results_summary:
            st.write("**자극별 반응 이벤트 (초록 면적 / 거리 신호 비교)**")
            st.dataframe(pd.DataFrame(results_summary), use_container_width=True)

    else:
        st.info("자극 시점을 입력하지 않아 자동 반응 감지는 건너뛰었습니다. 아래 그래프로 직접 확인하세요.")


    max_distance_change = float(df["relative_distance_px"].abs().max())
    max_distance_velocity = float(df["distance_velocity_px_s"].abs().max())
    min_green_ratio = float(df["green_ratio_smooth"].min())
    max_green_ratio = float(df["green_ratio_smooth"].max())

    col1, col2, col3 = st.columns(3)
    col1.metric("최대 거리 변화", f"{max_distance_change:.1f}px")
    col2.metric("최대 이동 속도", f"{max_distance_velocity:.1f}px/s")
    col3.metric("초록 면적비 범위", f"{min_green_ratio*100:.1f}% ~ {max_green_ratio*100:.1f}%")


    # ========================================================
    # 13. 그래프 (3개 신호)
    # ========================================================

    st.subheader("움직임 그래프")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(df["time_s"], df["green_ratio"], alpha=0.3, color="gray", label="원본")
    axes[0].plot(df["time_s"], df["green_ratio_smooth"], color="green", linewidth=2, label="스무딩")
    axes[0].set_ylabel("Green Area Ratio")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["time_s"], df["motion_score"], alpha=0.3, color="gray", label="원본")
    axes[1].plot(df["time_s"], df["motion_score_smooth"], color="blue", linewidth=2, label="스무딩")
    axes[1].set_ylabel("Motion Score")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(df["time_s"], df["relative_distance_px"], color="orange", linewidth=2, label="P1-Tip 상대 거리")
    axes[2].set_ylabel("Relative Distance (px)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    for stim_t in stimulus_times:
        for ax in axes:
            ax.axvline(stim_t, color="red", linestyle="--", alpha=0.6)

    fig.suptitle("Mimosa Motion Analysis (ROI 기반, API 불필요)")
    fig.tight_layout()

    st.pyplot(fig)


    # ========================================================
    # 14. 데이터 테이블 + CSV
    # ========================================================

    st.subheader("프레임별 데이터")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📊 CSV 다운로드", data=csv_data,
        file_name="mimosa_analysis.csv", mime="text/csv"
    )


    # ========================================================
    # 15. 분석 영상
    # ========================================================

    st.subheader("추적 결과 영상")

    with open(output_video.name, "rb") as f:
        output_video_bytes = f.read()

    st.video(output_video_bytes)

    st.download_button(
        "🎥 분석 영상 다운로드", data=output_video_bytes,
        file_name="mimosa_tracking_result.mp4", mime="video/mp4"
    )
