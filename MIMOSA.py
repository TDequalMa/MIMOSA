import streamlit as st
import cv2
import numpy as np
import pandas as pd
import math
import tempfile
import os
import json
import matplotlib.pyplot as plt

from PIL import Image
from google import genai
from google.genai import types
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
    "Gemini로 첫 프레임에서 keypoint를 찾고, "
    "이후 프레임은 Optical Flow로 추적하여 "
    "미모사 잎 끝 움직임을 분석합니다."
)


# ============================================================
# 2. Gemini 연결
# ============================================================

if "GEMINI_API_KEY" not in st.secrets:
    st.error(
        "GEMINI_API_KEY가 설정되지 않았습니다. "
        "Streamlit Cloud → 앱 설정 → Secrets에 GEMINI_API_KEY를 추가하세요."
    )
    st.stop()

GEMINI_CLIENT = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

GEMINI_MODEL = st.selectbox(
    "Gemini 모델",
    options=["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-2.5-flash"],
    index=0,
    help="3.5-flash: 빠르고 저렴함(추천) / 3.1-pro-preview: 더 정밀하지만 느리고 비쌈 / "
         "2.5-flash: 구버전, 안 될 수도 있음. "
         "모델이 또 404가 나면 https://ai.google.dev/gemini-api/docs/models 에서 "
         "최신 모델명을 확인하세요."
)


# ============================================================
# 3. Gemini 기반 keypoint 탐지 함수
# ============================================================

def detect_points_with_gemini(image_bgr, hint_xy=None):
    """
    Gemini에게 이미지 속 미모사 잎의 keypoint들을 짚어달라고 요청한다.
    hint_xy = (x, y)가 주어지면, 그 근처의 점을 우선적으로
    다시 찾아달라고 프롬프트에 포함한다 (재탐지용).

    반환: [(x_px, y_px, label), ...], raw_response_text
    """

    h, w = image_bgr.shape[:2]

    pil_img = Image.fromarray(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    )

    if hint_xy is not None:
        hx, hy = hint_xy
        hint_text = (
            f"참고로 방금 전까지는 이 점이 대략 픽셀 좌표 "
            f"({hx:.0f}, {hy:.0f}) 근처에 있었습니다. "
            "가능하면 그 근처에서 같은 지점을 다시 찾아주세요."
        )
    else:
        hint_text = ""

    prompt = f"""이 이미지는 미모사(mimosa) 식물의 잎을 촬영한 사진입니다.
잎의 움직임을 추적하기 위한 keypoint를 최대 5개까지 찾아주세요
(잎 끝(tip), 중간 마디, 잎자루 등 잎을 따라 있는 특징점).
{hint_text}
각 점에 대해 label과 이미지 내 정규화 좌표(0~1000 범위의 y, x)를
아래 JSON 형식으로만 응답하세요. 다른 설명 없이 JSON만 반환하세요.

{{"points": [{{"label": "tip", "point": [y, x]}}, {{"label": "mid", "point": [y, x]}}]}}"""

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=[pil_img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        raw_text = response.text

    except Exception as e:
        return [], f"API 에러: {e}"

    try:
        data = json.loads(raw_text)
    except Exception:
        return [], raw_text

    points = []

    for item in data.get("points", []):
        try:
            label = item.get("label", "point")
            y_norm, x_norm = item["point"]
            x_px = float(x_norm) / 1000.0 * w
            y_px = float(y_norm) / 1000.0 * h
            points.append((x_px, y_px, label))
        except Exception:
            continue

    return points, raw_text


# ============================================================
# 4. 영상 업로드
# ============================================================

uploaded_video = st.file_uploader(
    "미모사 영상을 업로드하세요.",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_video is None:
    st.info("먼저 미모사 영상을 업로드하세요.")
    st.stop()


# ============================================================
# 5. 업로드 영상을 임시 파일로 저장
# ============================================================

video_bytes = uploaded_video.getvalue()

input_video = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
)

input_video.write(video_bytes)
input_video.close()

video_path = input_video.name


# ============================================================
# 6. 영상 정보 확인
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
# 7. 영상 정보 표시
# ============================================================

st.subheader("영상 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric("해상도", f"{width} × {height}")
col2.metric("FPS", f"{fps:.2f}")
col3.metric("프레임 수", f"{frame_count}")
col4.metric("영상 길이", f"{duration:.2f}초")

st.video(video_bytes)


# ============================================================
# 8. 첫 프레임 표시 + P1 클릭 설정
# ============================================================

st.subheader("1단계 — P1 설정")

st.write(
    """
    **P1은 분석 대상 잎의 운동을 측정하기 위한 기준점입니다.**

    아래 이미지를 **클릭**해서 P1 좌표를 설정하세요.
    """
)

if "p1_x" not in st.session_state:
    st.session_state.p1_x = width // 2

if "p1_y" not in st.session_state:
    st.session_state.p1_y = height // 2

p1_preview = first_frame.copy()

cv2.circle(
    p1_preview,
    (st.session_state.p1_x, st.session_state.p1_y),
    10,
    (0, 0, 255),
    -1
)

cv2.putText(
    p1_preview,
    "P1",
    (st.session_state.p1_x + 10, st.session_state.p1_y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

p1_preview_rgb = cv2.cvtColor(p1_preview, cv2.COLOR_BGR2RGB)

p1_click_coords = streamlit_image_coordinates(p1_preview_rgb, key="p1_picker")

if p1_click_coords is not None:

    clicked_x = int(p1_click_coords["x"])
    clicked_y = int(p1_click_coords["y"])

    clicked_x = max(0, min(width - 1, clicked_x))
    clicked_y = max(0, min(height - 1, clicked_y))

    if (clicked_x, clicked_y) != (st.session_state.p1_x, st.session_state.p1_y):
        st.session_state.p1_x = clicked_x
        st.session_state.p1_y = clicked_y
        st.rerun()

st.write(f"현재 P1 좌표: **({st.session_state.p1_x}, {st.session_state.p1_y})**")

with st.expander("🔧 P1 좌표 직접 입력 / 미세 조정"):

    col1, col2 = st.columns(2)

    with col1:
        manual_x = st.number_input(
            "P1 X 좌표",
            min_value=0,
            max_value=width - 1,
            value=st.session_state.p1_x,
            step=1,
            key="manual_p1_x"
        )

    with col2:
        manual_y = st.number_input(
            "P1 Y 좌표",
            min_value=0,
            max_value=height - 1,
            value=st.session_state.p1_y,
            step=1,
            key="manual_p1_y"
        )

    if st.button("이 좌표로 P1 설정"):
        st.session_state.p1_x = int(manual_x)
        st.session_state.p1_y = int(manual_y)
        st.rerun()

BASE_POINT = (int(st.session_state.p1_x), int(st.session_state.p1_y))


# ============================================================
# 9. 2단계 — 추적할 점 선택 (Gemini)
# ============================================================

st.subheader("2단계 — 추적할 점 선택 (Gemini)")

st.write(
    """
    Gemini가 첫 프레임에서 찾은 **모든 점**을 초록색으로 표시합니다.
    이 중에서 **추적하고 싶은 점을 클릭**하세요 (가장 가까운 점이 선택됩니다).
    """
)

if "detected_points" not in st.session_state:
    st.session_state.detected_points = None

if "selected_point" not in st.session_state:
    st.session_state.selected_point = None

if "gemini_debug_text" not in st.session_state:
    st.session_state.gemini_debug_text = None

detect_button = st.button("🔍 Gemini로 첫 프레임에서 점 감지하기")

if detect_button:

    with st.spinner("Gemini에게 물어보는 중..."):
        points, raw_text = detect_points_with_gemini(first_frame)

    st.session_state.detected_points = points
    st.session_state.gemini_debug_text = raw_text

    if points:
        farthest = max(
            points,
            key=lambda p: math.hypot(p[0] - BASE_POINT[0], p[1] - BASE_POINT[1])
        )
        st.session_state.selected_point = (farthest[0], farthest[1])
    else:
        st.session_state.selected_point = None
        st.warning("Gemini가 이 프레임에서 점을 하나도 찾지 못했습니다. 아래 디버그 응답을 확인해보세요.")

    st.rerun()


debug_mode = st.checkbox(
    "🔍 디버그 모드 (Gemini 원본 응답 표시)",
    value=True
)

if debug_mode and st.session_state.gemini_debug_text is not None:
    with st.expander("Gemini 원본 응답 (첫 프레임)"):
        st.code(st.session_state.gemini_debug_text, language="json")


if st.session_state.detected_points:

    points_preview = first_frame.copy()

    cv2.circle(points_preview, BASE_POINT, 10, (0, 0, 255), -1)
    cv2.putText(
        points_preview,
        "P1",
        (BASE_POINT[0] + 10, BASE_POINT[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    for (x, y, label) in st.session_state.detected_points:

        xi, yi = int(round(x)), int(round(y))

        is_selected = (
            st.session_state.selected_point is not None
            and abs(xi - int(round(st.session_state.selected_point[0]))) <= 1
            and abs(yi - int(round(st.session_state.selected_point[1]))) <= 1
        )

        color = (0, 165, 255) if is_selected else (0, 255, 0)
        radius = 11 if is_selected else 6

        cv2.circle(points_preview, (xi, yi), radius, color, -1)
        cv2.putText(
            points_preview,
            label,
            (xi + 8, yi - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    points_preview_rgb = cv2.cvtColor(points_preview, cv2.COLOR_BGR2RGB)

    point_click = streamlit_image_coordinates(points_preview_rgb, key="point_picker")

    if point_click is not None:

        cx = int(point_click["x"])
        cy = int(point_click["y"])

        nearest = min(
            st.session_state.detected_points,
            key=lambda p: math.hypot(p[0] - cx, p[1] - cy)
        )

        new_selected = (nearest[0], nearest[1])

        if (
            st.session_state.selected_point is None
            or abs(new_selected[0] - st.session_state.selected_point[0]) > 0.5
            or abs(new_selected[1] - st.session_state.selected_point[1]) > 0.5
        ):
            st.session_state.selected_point = new_selected
            st.rerun()

    if st.session_state.selected_point is not None:
        sx, sy = st.session_state.selected_point
        dist0 = math.hypot(sx - BASE_POINT[0], sy - BASE_POINT[1])
        st.success(f"✅ 선택된 점: ({sx:.0f}, {sy:.0f}) — P1과의 거리: **{dist0:.1f}px**")

else:
    st.info("먼저 위의 '🔍 Gemini로 첫 프레임에서 점 감지하기' 버튼을 눌러주세요.")


# ============================================================
# 10. 옵션
# ============================================================

st.subheader("옵션")

default_threshold = max(3, round(width * 0.02))

response_threshold_px = st.number_input(
    "반응 감지 임계값 (px)",
    min_value=1,
    value=default_threshold,
    step=1,
    help="P1과 추적점 사이 거리가 초기값보다 이만큼 변하면 '반응 시작'으로 판단합니다."
)

col1, col2 = st.columns(2)

with col1:
    st.info(
        "매 프레임마다 Gemini를 호출해서 점을 다시 찾고, "
        "이전 프레임의 선택점과 가장 가까운 점을 자동으로 골라 추적합니다."
    )

with col2:
    frame_skip = st.number_input(
        "프레임 스킵 (1 = 모든 프레임 분석)",
        min_value=1,
        value=1,
        step=1,
        help="1이면 모든 프레임을 Gemini로 분석합니다. "
             "값을 늘리면 그만큼 프레임을 건너뛰어 호출 수/시간을 줄일 수 있지만, "
             "시간 해상도가 낮아집니다."
    )


# ============================================================
# 11. 분석 시작
# ============================================================

st.subheader("3단계 — 영상 분석")

analyze_button = st.button(
    "🌱 분석 시작",
    type="primary",
    disabled=(st.session_state.selected_point is None)
)

if st.session_state.selected_point is None:
    st.warning("분석을 시작하려면 먼저 2단계에서 추적할 점을 선택하세요.")


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

    records = []
    frame_index = 0

    tracked_point = st.session_state.selected_point  # (x, y), 매 프레임 갱신됨
    gemini_call_count = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        timestamp = frame_index / fps

        tracking_method = None

        # ------------------------------------------------
        # 첫 프레임: 사용자가 2단계에서 선택한 점을 그대로 사용
        # ------------------------------------------------

        if frame_index == 0:

            sx, sy = st.session_state.selected_point
            tip_x, tip_y = float(sx), float(sy)
            tracking_ok = True
            tracking_method = "initial"

        # ------------------------------------------------
        # 프레임 스킵 설정으로 건너뛰는 프레임: 이전 위치 그대로 기록
        # (거리 변화 계산의 시간축은 유지하되, 이 프레임은 API를 호출하지 않음)
        # ------------------------------------------------

        elif frame_index % frame_skip != 0:

            tip_x, tip_y = tracked_point
            tracking_ok = True
            tracking_method = "skipped"

        else:

            # --------------------------------------------
            # 매 프레임 Gemini로 점 재탐지 →
            # 이전 프레임 추적점과 가장 가까운 점을 선택
            # --------------------------------------------

            all_points, raw_text = detect_points_with_gemini(
                frame, hint_xy=tracked_point
            )

            gemini_call_count += 1

            if all_points:

                nearest = min(
                    all_points,
                    key=lambda p: math.hypot(
                        p[0] - tracked_point[0], p[1] - tracked_point[1]
                    )
                )

                tip_x, tip_y = float(nearest[0]), float(nearest[1])
                tracking_ok = True
                tracking_method = "gemini"

            else:

                tip_x, tip_y = np.nan, np.nan
                tracking_ok = False
                tracking_method = "lost"
                # tracked_point는 그대로 유지 (다음 프레임에서 같은 위치 기준으로 재시도)

        if tracking_ok:
            tracked_point = (tip_x, tip_y)


        # ----------------------------------------------------
        # 거리 계산
        # ----------------------------------------------------

        distance_px = np.nan

        if tracking_ok:

            distance_px = math.hypot(
                tip_x - BASE_POINT[0],
                tip_y - BASE_POINT[1]
            )

            tip_int = (int(round(tip_x)), int(round(tip_y)))

            cv2.circle(frame, BASE_POINT, 8, (0, 0, 255), -1)

            marker_color = (255, 0, 255) if tracking_method == "skipped" else (0, 165, 255)

            cv2.circle(frame, tip_int, 8, marker_color, -1)
            cv2.line(frame, BASE_POINT, tip_int, (255, 255, 0), 2)

            cv2.putText(
                frame, "P1", (BASE_POINT[0] + 10, BASE_POINT[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

            label = "SKIP" if tracking_method == "skipped" else "TRACK"

            cv2.putText(
                frame, label, (tip_int[0] + 10, tip_int[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, marker_color, 2
            )

        cv2.putText(
            frame, f"Time: {timestamp:.2f}s", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        if not math.isnan(distance_px):
            cv2.putText(
                frame, f"Distance: {distance_px:.1f}px", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

        writer.write(frame)


        # ----------------------------------------------------
        # 데이터 저장
        # ----------------------------------------------------

        records.append({
            "frame": frame_index,
            "time_s": timestamp,
            "P1_x": BASE_POINT[0],
            "P1_y": BASE_POINT[1],
            "tip_x": tip_x if tracking_ok else np.nan,
            "tip_y": tip_y if tracking_ok else np.nan,
            "distance_px": distance_px,
            "tracking_method": tracking_method
        })

        frame_index += 1

        progress_bar.progress(min(frame_index / frame_count, 1.0))
        status.text(
            f"{frame_index} / {frame_count} 프레임 분석 중... "
            f"(Gemini 호출 {gemini_call_count}회)"
        )

    cap.release()
    writer.release()

    status.text(
        f"{frame_index}개 프레임 분석 완료 (Gemini 호출 {gemini_call_count}회 발생)"
    )


    # ========================================================
    # 12. 데이터프레임
    # ========================================================

    df = pd.DataFrame(records)

    df["tip_x"] = df["tip_x"].interpolate(limit_direction="both")
    df["tip_y"] = df["tip_y"].interpolate(limit_direction="both")
    df["distance_px"] = df["distance_px"].interpolate(limit_direction="both")

    if df["distance_px"].isna().all():
        st.error(
            "추적점을 한 번도 확보하지 못했습니다. "
            "위쪽 디버그 응답을 확인하거나, Gemini 재탐지 옵션을 켜보세요."
        )
        st.stop()


    # ========================================================
    # 13. 초기 거리 대비 상대 변화
    # ========================================================

    initial_distance = float(df["distance_px"].iloc[0])
    df["relative_distance_px"] = df["distance_px"] - initial_distance


    # ========================================================
    # 14. 거리 변화 속도 (px/s)
    # ========================================================

    df["dt"] = df["time_s"].diff()
    df["ddist"] = df["relative_distance_px"].diff()
    df["distance_velocity_px_s"] = df["ddist"] / df["dt"]

    velocity_outlier_limit = (
        max(50, float(df["distance_velocity_px_s"].abs().quantile(0.99)) * 5)
        if df["distance_velocity_px_s"].notna().any() else 1e9
    )

    df.loc[
        df["distance_velocity_px_s"].abs() > velocity_outlier_limit,
        "distance_velocity_px_s"
    ] = np.nan

    df["distance_velocity_px_s"] = df["distance_velocity_px_s"].interpolate(limit_direction="both")


    # ========================================================
    # 15. 반응 시작 시간
    # ========================================================

    response_candidates = df.loc[
        df["relative_distance_px"].abs() >= response_threshold_px,
        "time_s"
    ]

    response_time = float(response_candidates.iloc[0]) if len(response_candidates) > 0 else np.nan


    # ========================================================
    # 16. 주요 결과
    # ========================================================

    max_distance_change = float(df["relative_distance_px"].abs().max())
    max_distance_velocity = float(df["distance_velocity_px_s"].abs().max())
    max_index = int(df["relative_distance_px"].abs().idxmax())
    max_change_time = float(df.loc[max_index, "time_s"])


    # ========================================================
    # 17. 결과 표시
    # ========================================================

    st.success("분석이 완료되었습니다.")

    st.subheader("4단계 — 분석 결과")

    col1, col2, col3 = st.columns(3)

    col1.metric("최대 거리 변화", f"{max_distance_change:.1f}px")
    col2.metric("최대 이동 속도", f"{max_distance_velocity:.1f}px/s")

    if math.isnan(response_time):
        col3.metric("반응 시작", "검출되지 않음")
    else:
        col3.metric("반응 시작", f"{response_time:.2f}s")

    st.write(f"최대 변화 시점: **{max_change_time:.2f}초**")

    method_counts = df["tracking_method"].value_counts()
    st.caption(
        "추적 방식별 프레임 수: "
        + ", ".join(f"{k} {v}개" for k, v in method_counts.items())
    )


    # ========================================================
    # 18. 그래프
    # ========================================================

    st.subheader("움직임 그래프")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df["time_s"], df["relative_distance_px"], label="Relative Distance")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Relative Distance (px)")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(df["time_s"], df["distance_velocity_px_s"], alpha=0.65, label="Distance Change Rate")
    ax2.set_ylabel("Distance Change Rate (px/s)")

    if not math.isnan(response_time):
        ax1.axvline(response_time, linestyle="--", alpha=0.7)

    fig.suptitle("Mimosa Motion Analysis (Gemini + Optical Flow)")
    fig.tight_layout()

    st.pyplot(fig)


    # ========================================================
    # 19. 데이터 테이블
    # ========================================================

    st.subheader("프레임별 데이터")
    st.dataframe(df, use_container_width=True)


    # ========================================================
    # 20. CSV 다운로드
    # ========================================================

    csv_data = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "📊 CSV 다운로드",
        data=csv_data,
        file_name="mimosa_analysis.csv",
        mime="text/csv"
    )


    # ========================================================
    # 21. 분석 영상
    # ========================================================

    st.subheader("추적 결과 영상")

    with open(output_video.name, "rb") as f:
        output_video_bytes = f.read()

    st.video(output_video_bytes)

    st.download_button(
        "🎥 분석 영상 다운로드",
        data=output_video_bytes,
        file_name="mimosa_tracking_result.mp4",
        mime="video/mp4"
    )
