import streamlit as st
import cv2
import numpy as np
import pandas as pd
import math
import tempfile
import os
import matplotlib.pyplot as plt

from inference_sdk import InferenceHTTPClient
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
    "Roboflow Keypoint Detection을 이용하여 "
    "미모사의 잎 끝 움직임을 분석합니다."
)

MODEL_ID = "tip-detection-qejht/1"


# ============================================================
# 2. Roboflow 연결
# ============================================================

if "ROBOFLOW_API_KEY" not in st.secrets:
    st.error(
        "ROBOFLOW_API_KEY가 설정되지 않았습니다. "
        ".streamlit/secrets.toml에 API Key를 입력하세요."
    )
    st.stop()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=st.secrets["ROBOFLOW_API_KEY"]
)


# ============================================================
# 3. 영상 업로드
# ============================================================

uploaded_video = st.file_uploader(
    "미모사 영상을 업로드하세요.",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_video is None:
    st.info("먼저 미모사 영상을 업로드하세요.")
    st.stop()


# ============================================================
# 4. 업로드 영상을 임시 파일로 저장
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
# 5. 영상 정보 확인
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("영상을 열 수 없습니다.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0 or math.isnan(fps):
    fps = 30.0

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

ok, first_frame = cap.read()

cap.release()

if not ok:
    st.error("첫 번째 프레임을 읽을 수 없습니다.")
    st.stop()


duration = frame_count / fps


# ============================================================
# 6. 영상 정보 표시
# ============================================================

st.subheader("영상 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric("해상도", f"{width} × {height}")
col2.metric("FPS", f"{fps:.2f}")
col3.metric("프레임 수", f"{frame_count}")
col4.metric("영상 길이", f"{duration:.2f}초")


st.video(video_bytes)


# ============================================================
# 7. 첫 프레임 표시 + P1 클릭 설정
# ============================================================

st.subheader("1단계 — P1 설정")

st.write(
    """
    **P1은 분석 대상 잎의 운동을 측정하기 위한 기준점입니다.**

    아래 이미지를 **클릭**해서 P1 좌표를 설정하세요.
    """
)


# 세션에 좌표 저장 (처음엔 이미지 중앙)
if "p1_x" not in st.session_state:
    st.session_state.p1_x = width // 2

if "p1_y" not in st.session_state:
    st.session_state.p1_y = height // 2


# 클릭 가능한 이미지 (미리보기용으로 P1 표시 반영해서 그림)
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
    (
        st.session_state.p1_x + 10,
        st.session_state.p1_y - 10
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

p1_preview_rgb = cv2.cvtColor(
    p1_preview,
    cv2.COLOR_BGR2RGB
)

p1_click_coords = streamlit_image_coordinates(
    p1_preview_rgb,
    key="p1_picker"
)

# 새로 클릭했으면 좌표 갱신
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


# 미세 조정용 숫자 입력 (선택 사항)
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


p1_x = st.session_state.p1_x
p1_y = st.session_state.p1_y

BASE_POINT = (
    int(p1_x),
    int(p1_y)
)


# ============================================================
# 8. Keypoint 추출 / 최근접점 탐색 함수
# ============================================================

def extract_all_points(result):
    """
    Roboflow 결과에서 감지된 '모든' keypoint를 리스트로 반환한다.
    반환 형식: [(x, y, confidence), (x, y, confidence), ...]
    """

    points = []

    if not result:
        return points

    predictions = result.get("predictions", [])

    if not predictions:
        return points

    # 가장 높은 confidence의 prediction(=잎 하나)을 대상으로 함
    best_prediction = max(
        predictions,
        key=lambda x: x.get("confidence", 0)
    )

    keypoints = best_prediction.get("keypoints")

    if keypoints is None:
        return points

    # 형태 1: keypoints = [{"x":.., "y":.., "confidence":..}, ...]
    if isinstance(keypoints, list):

        for kp in keypoints:

            if not isinstance(kp, dict):
                continue

            x = kp.get("x")
            y = kp.get("y")

            if x is None or y is None:
                continue

            confidence = kp.get("confidence", 1.0)

            points.append(
                (float(x), float(y), float(confidence))
            )

    # 형태 2: keypoints = {"tip": {"x":.., "y":..}, "base": {...}, ...}
    elif isinstance(keypoints, dict):

        for name, kp in keypoints.items():

            if not isinstance(kp, dict):
                continue

            x = kp.get("x")
            y = kp.get("y")

            if x is None or y is None:
                continue

            confidence = kp.get("confidence", 1.0)

            points.append(
                (float(x), float(y), float(confidence))
            )

    return points


def find_nearest_point(target_xy, points):
    """
    points: [(x, y, confidence), ...] 중에서
    target_xy = (x, y) 와 가장 가까운 점을 찾아 (x, y, confidence)로 반환.
    points가 비어있으면 None.
    """

    if not points:
        return None

    tx, ty = target_xy

    best_point = None
    best_dist = None

    for (x, y, conf) in points:

        d = math.hypot(x - tx, y - ty)

        if best_dist is None or d < best_dist:
            best_dist = d
            best_point = (x, y, conf)

    return best_point


# ============================================================
# 9. 2단계 — 추적할 점 선택
# ============================================================

st.subheader("2단계 — 추적할 점 선택")

st.write(
    """
    AI 모델이 첫 프레임에서 감지한 **모든 점**을 초록색으로 표시합니다.
    이 중에서 **추적하고 싶은 점을 클릭**하세요 (클릭한 위치에서 가장 가까운 점이 선택됩니다).
    선택된 점은 **주황색**으로 표시되고, 그 점과 P1 사이의 거리를 기준으로 분석이 시작됩니다.
    """
)

if "detected_points" not in st.session_state:
    st.session_state.detected_points = None

if "selected_point" not in st.session_state:
    st.session_state.selected_point = None


detect_button = st.button("🔍 첫 프레임에서 점 감지하기")

if detect_button:

    temp_first = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_first_path = temp_first.name
    temp_first.close()

    cv2.imwrite(temp_first_path, first_frame)

    try:
        first_result = CLIENT.infer(temp_first_path, model_id=MODEL_ID)

        st.session_state.detected_points = extract_all_points(first_result)

        if st.session_state.detected_points:

            # 기본값: P1에서 가장 먼 점을 자동 선택 (보통 잎 끝일 확률이 높음)
            # 마음에 안 들면 아래 이미지에서 다시 클릭해서 바꿀 수 있음
            farthest = max(
                st.session_state.detected_points,
                key=lambda p: math.hypot(p[0] - BASE_POINT[0], p[1] - BASE_POINT[1])
            )
            st.session_state.selected_point = (farthest[0], farthest[1])

        else:
            st.session_state.selected_point = None
            st.warning("이 프레임에서 점을 하나도 감지하지 못했습니다. AI 모델이나 이미지를 확인해보세요.")

    except Exception as e:
        st.error(f"API 에러: {e}")

    finally:
        if os.path.exists(temp_first_path):
            os.remove(temp_first_path)

    st.rerun()


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

    for (x, y, conf) in st.session_state.detected_points:

        xi, yi = int(round(x)), int(round(y))

        is_selected = (
            st.session_state.selected_point is not None
            and abs(xi - int(round(st.session_state.selected_point[0]))) <= 1
            and abs(yi - int(round(st.session_state.selected_point[1]))) <= 1
        )

        if is_selected:
            color = (0, 165, 255)  # 주황 (선택됨)
            radius = 11
        else:
            color = (0, 255, 0)  # 초록 (미선택)
            radius = 6

        cv2.circle(points_preview, (xi, yi), radius, color, -1)

    points_preview_rgb = cv2.cvtColor(points_preview, cv2.COLOR_BGR2RGB)

    point_click = streamlit_image_coordinates(points_preview_rgb, key="point_picker")

    if point_click is not None:

        cx = int(point_click["x"])
        cy = int(point_click["y"])

        nearest = find_nearest_point((cx, cy), st.session_state.detected_points)

        if nearest is not None:

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

elif detect_button:
    pass

else:
    st.info("먼저 위의 '🔍 첫 프레임에서 점 감지하기' 버튼을 눌러주세요.")


# ============================================================
# 10. 반응 감지 임계값 + 디버그 옵션
# ============================================================

st.subheader("옵션")

default_threshold = max(3, round(width * 0.02))

response_threshold_px = st.number_input(
    "반응 감지 임계값 (px) — P1과 추적점 사이 거리가 초기값보다 이만큼 변하면 '반응 시작'으로 판단",
    min_value=1,
    value=default_threshold,
    step=1
)

debug_mode = st.checkbox(
    "🔍 디버그 모드 (첫 프레임의 원본 API 응답을 화면에 표시)",
    value=True,
    help="Roboflow API가 실제로 어떤 데이터 구조를 돌려주는지 확인할 때 켜세요."
)


# ============================================================
# 11. 분석 시작
# ============================================================

st.subheader("3단계 — AI 영상 분석")

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

    # 디버그 모드용 표시 영역 (첫 프레임 결과만 여기에 찍힘)
    debug_container = st.container()
    debug_shown = False

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        st.error(
            "영상을 열 수 없습니다."
        )

        st.stop()


    # 결과 영상
    output_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_video.close()


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_video.name,
        fourcc,
        fps,
        (width, height)
    )


    records = []

    frame_index = 0

    api_error_count = 0
    last_api_error = None

    # 프레임별로 "이전 프레임에서 선택된 점"을 갱신해가며 추적
    tracked_point = st.session_state.selected_point


    while True:

        ok, frame = cap.read()

        if not ok:
            break


        timestamp = frame_index / fps


        # ----------------------------------------------------
        # 임시 이미지 생성
        # ----------------------------------------------------

        temp_image = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        temp_image_path = temp_image.name

        temp_image.close()


        cv2.imwrite(
            temp_image_path,
            frame
        )


        # ----------------------------------------------------
        # Roboflow 추론
        # ----------------------------------------------------

        all_points = []

        try:

            result = CLIENT.infer(
                temp_image_path,
                model_id=MODEL_ID
            )

            all_points = extract_all_points(result)

            # 디버그 모드: 첫 프레임의 원본 응답을 화면에 출력
            if debug_mode and not debug_shown:
                with debug_container:
                    st.markdown("### 🔍 디버그: 첫 프레임 API 원본 응답")
                    st.json(result)
                    st.write(f"감지된 점 개수: {len(all_points)}")
                    st.write(f"감지된 점 좌표: {all_points}")
                debug_shown = True

        except Exception as e:

            api_error_count += 1
            last_api_error = str(e)

            if debug_mode and api_error_count == 1:
                with debug_container:
                    st.markdown("### 🔍 디버그: API 호출 에러")
                    st.error(f"에러 내용: {e}")

        finally:

            if os.path.exists(
                temp_image_path
            ):
                os.remove(
                    temp_image_path
                )


        # ----------------------------------------------------
        # 최근접점 추적: 이전 프레임에서 선택했던 점과
        # 가장 가까운 점을 이번 프레임의 추적점으로 삼는다
        # ----------------------------------------------------

        nearest = find_nearest_point(
            (tracked_point[0], tracked_point[1]),
            all_points
        )


        # ----------------------------------------------------
        # 결과 초기화
        # ----------------------------------------------------

        tip_x = np.nan
        tip_y = np.nan
        distance_px = np.nan
        tip_confidence = np.nan


        # ----------------------------------------------------
        # 추적점 검출 성공
        # ----------------------------------------------------

        if nearest is not None:

            nx, ny, nconf = nearest

            tip = (int(round(nx)), int(round(ny)))

            tip_x, tip_y = tip
            tip_confidence = nconf

            distance_px = math.hypot(
                tip_x - BASE_POINT[0],
                tip_y - BASE_POINT[1]
            )

            # 다음 프레임에서는 이번에 찾은 점을 기준으로 최근접점을 다시 탐색
            tracked_point = (nx, ny)


            # AI 결과 표시
            cv2.circle(
                frame,
                BASE_POINT,
                8,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                tip,
                8,
                (0, 165, 255),
                -1
            )

            cv2.line(
                frame,
                BASE_POINT,
                tip,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "P1",
                (BASE_POINT[0] + 10, BASE_POINT[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "TRACK",
                (tip_x + 10, tip_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )


        # ----------------------------------------------------
        # 영상에 시간/거리 표시
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Time: {timestamp:.2f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if not math.isnan(distance_px):

            cv2.putText(
                frame,
                f"Distance: {distance_px:.1f}px",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
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
            "tip_x": tip_x,
            "tip_y": tip_y,
            "confidence": tip_confidence,
            "distance_px": distance_px
        })


        frame_index += 1


        progress_bar.progress(
            min(frame_index / frame_count, 1.0)
        )

        status.text(
            f"{frame_index} / {frame_count} 프레임 분석 중..."
        )


    cap.release()
    writer.release()


    status.text(
        f"{frame_index}개 프레임 분석 완료"
        + (f" (API 에러 {api_error_count}회 발생)" if api_error_count > 0 else "")
    )

    if api_error_count > 0:
        st.warning(
            f"⚠️ 총 {frame_index}개 프레임 중 {api_error_count}개 프레임에서 API 호출이 실패했습니다. "
            f"마지막 에러: {last_api_error}"
        )


    # ========================================================
    # 12. 데이터프레임
    # ========================================================

    df = pd.DataFrame(records)

    # 검출 실패 구간 보간
    df["tip_x"] = df["tip_x"].interpolate(limit_direction="both")
    df["tip_y"] = df["tip_y"].interpolate(limit_direction="both")
    df["distance_px"] = df["distance_px"].interpolate(limit_direction="both")


    if df["distance_px"].isna().all():

        st.error(
            "AI가 추적점을 한 번도 검출하지 못했습니다. "
            "위쪽의 '디버그 모드' 결과를 확인해서 API가 점을 비어있게 주는지 확인해보세요."
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

    # 이상값 제거 (프레임 간 튀는 값)
    velocity_outlier_limit = max(50, float(df["distance_velocity_px_s"].abs().quantile(0.99)) * 5) \
        if df["distance_velocity_px_s"].notna().any() else 1e9

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

    if len(response_candidates) > 0:
        response_time = float(response_candidates.iloc[0])
    else:
        response_time = np.nan


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


    # ========================================================
    # 18. 그래프
    # ========================================================

    st.subheader("움직임 그래프")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        df["time_s"],
        df["relative_distance_px"],
        label="Relative Distance (P1 - Tracked Point)"
    )

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Relative Distance (px)")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()

    ax2.plot(
        df["time_s"],
        df["distance_velocity_px_s"],
        alpha=0.65,
        label="Distance Change Rate"
    )

    ax2.set_ylabel("Distance Change Rate (px/s)")

    if not math.isnan(response_time):
        ax1.axvline(response_time, linestyle="--", alpha=0.7)

    fig.suptitle("Mimosa Motion Analysis (Point-Tracking Based)")

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

    st.subheader("AI 추적 결과 영상")

    with open(output_video.name, "rb") as f:
        output_video_bytes = f.read()

    st.video(output_video_bytes)

    st.download_button(
        "🎥 분석 영상 다운로드",
        data=output_video_bytes,
        file_name="mimosa_tracking_result.mp4",
        mime="video/mp4"
    )
