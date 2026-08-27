ㅍimport streamlit as st
import cv2
import numpy as np
import pandas as pd
import math
import tempfile
import os
import matplotlib.pyplot as plt

from inference_sdk import InferenceHTTPClient


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

MODEL_ID = "leaf-keypoint-clear/7"


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
    api_url="https://serverless.roboflow.com",
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
# 7. 첫 프레임 표시
# ============================================================

st.subheader("1단계 — P1 설정")

first_rgb = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2RGB
)

st.image(
    first_rgb,
    caption="첫 번째 프레임",
    use_container_width=True
)


st.write(
    """
    **P1은 분석 대상 잎의 운동을 측정하기 위한 기준점입니다.**

    첫 번째 프레임에서 P1의 픽셀 좌표를 입력하세요.
    좌표의 원점은 이미지의 왼쪽 위입니다.
    """
)


# ============================================================
# 8. P1 입력
# ============================================================

col1, col2 = st.columns(2)

with col1:
    p1_x = st.number_input(
        "P1 X 좌표",
        min_value=0,
        max_value=width - 1,
        value=width // 2,
        step=1
    )

with col2:
    p1_y = st.number_input(
        "P1 Y 좌표",
        min_value=0,
        max_value=height - 1,
        value=height // 2,
        step=1
    )

BASE_POINT = (
    int(p1_x),
    int(p1_y)
)


# ============================================================
# 9. P1 확인 이미지
# ============================================================

preview = first_frame.copy()

cv2.circle(
    preview,
    BASE_POINT,
    10,
    (0, 0, 255),
    -1
)

cv2.putText(
    preview,
    "P1",
    (
        BASE_POINT[0] + 10,
        BASE_POINT[1] - 10
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

preview_rgb = cv2.cvtColor(
    preview,
    cv2.COLOR_BGR2RGB
)

st.image(
    preview_rgb,
    caption=f"P1 = ({p1_x}, {p1_y})",
    use_container_width=True
)


# ============================================================
# 9.5 디버그 모드 옵션 (신규 추가)
# ============================================================

st.subheader("디버그 옵션")

debug_mode = st.checkbox(
    "🔍 디버그 모드 (첫 프레임의 원본 API 응답을 화면에 표시)",
    value=True,
    help="Roboflow API가 실제로 어떤 데이터 구조를 돌려주는지 확인할 때 켜세요. "
         "잎 끝점을 못 찾는 문제를 진단하는 데 도움이 됩니다."
)


# ============================================================
# 10. Keypoint 결과에서 좌표 찾기
# ============================================================

def extract_tip_from_result(result):
    """
    Roboflow Keypoint Detection 결과에서
    가장 적절한 keypoint 좌표를 추출한다.

    주의:
    leaf-keypoint-clear/7의 실제 keypoint 이름과
    결과 구조를 확인하기 전까지는
    첫 번째 유효 keypoint를 사용한다.
    """

    if not result:
        return None

    predictions = result.get("predictions", [])

    if not predictions:
        return None

    # 가장 높은 confidence의 prediction 선택
    best_prediction = max(
        predictions,
        key=lambda x: x.get("confidence", 0)
    )

    keypoints = best_prediction.get("keypoints")

    if keypoints is None:
        return None

    # ----------------------------------------
    # 형태 1:
    # keypoints = [
    #     {"x": ..., "y": ..., "confidence": ...}
    # ]
    # ----------------------------------------

    if isinstance(keypoints, list):

        valid_points = []

        for kp in keypoints:

            if not isinstance(kp, dict):
                continue

            x = kp.get("x")
            y = kp.get("y")

            if x is None or y is None:
                continue

            confidence = kp.get(
                "confidence",
                1.0
            )

            valid_points.append(
                (
                    float(confidence),
                    float(x),
                    float(y)
                )
            )

        if valid_points:

            valid_points.sort(
                reverse=True
            )

            _, x, y = valid_points[0]

            return (
                int(round(x)),
                int(round(y))
            )

    # ----------------------------------------
    # 형태 2:
    # keypoints = {
    #     "tip": {
    #         "x": ...,
    #         "y": ...
    #     }
    # }
    # ----------------------------------------

    if isinstance(keypoints, dict):

        candidate_points = []

        for name, kp in keypoints.items():

            if not isinstance(kp, dict):
                continue

            x = kp.get("x")
            y = kp.get("y")

            if x is None or y is None:
                continue

            confidence = kp.get(
                "confidence",
                1.0
            )

            candidate_points.append(
                (
                    float(confidence),
                    float(x),
                    float(y)
                )
            )

        if candidate_points:

            candidate_points.sort(
                reverse=True
            )

            _, x, y = candidate_points[0]

            return (
                int(round(x)),
                int(round(y))
            )

    return None


# ============================================================
# 11. 각도 계산
# ============================================================

def calculate_angle(
    base_point,
    tip_point
):

    bx, by = base_point
    tx, ty = tip_point

    dx = tx - bx

    # 영상 좌표의 y축은 아래 방향이므로
    # 수학 좌표계와 맞추기 위해 반전
    dy = by - ty

    angle = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    return angle


# ============================================================
# 12. 분석 시작
# ============================================================

st.subheader("2단계 — AI 영상 분석")

analyze_button = st.button(
    "🌱 분석 시작",
    type="primary"
)


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


    while True:

        ok, frame = cap.read()

        if not ok:
            break


        timestamp = frame_index / fps


        # ----------------------------------------------------
        # 프레임 저장
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # 임시 이미지 생성
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

        try:

            result = CLIENT.infer(
                temp_image_path,
                model_id=MODEL_ID
            )

            # 디버그 모드: 첫 프레임의 원본 응답을 화면에 출력
            if debug_mode and not debug_shown:
                with debug_container:
                    st.markdown("### 🔍 디버그: 첫 프레임 API 원본 응답")
                    st.json(result)
                    predictions_dbg = result.get("predictions", []) if result else []
                    if predictions_dbg:
                        st.write(
                            f"prediction 개수: {len(predictions_dbg)}, "
                            f"첫 prediction의 keys: {list(predictions_dbg[0].keys())}"
                        )
                        kp_dbg = predictions_dbg[0].get("keypoints")
                        st.write(f"keypoints 타입: {type(kp_dbg)}")
                        st.write(f"keypoints 내용: {kp_dbg}")
                    else:
                        st.warning("predictions가 비어 있습니다. 모델이 이 프레임에서 아무것도 검출하지 못했습니다.")
                debug_shown = True

            tip = extract_tip_from_result(
                result
            )

        except Exception as e:

            tip = None
            api_error_count += 1
            last_api_error = str(e)

            # 디버그 모드: 첫 에러의 상세 내용을 화면에 출력
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
        # 결과 초기화
        # ----------------------------------------------------

        tip_x = np.nan
        tip_y = np.nan
        angle = np.nan
        confidence = np.nan


        # ----------------------------------------------------
        # Tip 검출 성공
        # ----------------------------------------------------

        if tip is not None:

            tip_x, tip_y = tip

            angle = calculate_angle(
                BASE_POINT,
                tip
            )


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
                (255, 0, 0),
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
                (
                    BASE_POINT[0] + 10,
                    BASE_POINT[1]
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "TIP",
                (
                    tip_x + 10,
                    tip_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )


        # ----------------------------------------------------
        # 영상에 시간/각도 표시
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


        if not math.isnan(angle):

            cv2.putText(
                frame,
                f"Angle: {angle:.2f} deg",
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

            "frame":
                frame_index,

            "time_s":
                timestamp,

            "P1_x":
                BASE_POINT[0],

            "P1_y":
                BASE_POINT[1],

            "tip_x":
                tip_x,

            "tip_y":
                tip_y,

            "angle_deg":
                angle,

            "confidence":
                confidence

        })


        frame_index += 1


        progress_bar.progress(
            min(
                frame_index / frame_count,
                1.0
            )
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
    # 13. 데이터프레임
    # ========================================================

    df = pd.DataFrame(
        records
    )


    # Tip 검출 실패 구간 보간
    df["tip_x"] = (
        df["tip_x"]
        .interpolate(
            limit_direction="both"
        )
    )

    df["tip_y"] = (
        df["tip_y"]
        .interpolate(
            limit_direction="both"
        )
    )

    df["angle_deg"] = (
        df["angle_deg"]
        .interpolate(
            limit_direction="both"
        )
    )


    if df["angle_deg"].isna().all():

        st.error(
            "AI가 잎 끝점을 한 번도 검출하지 못했습니다. "
            "위쪽의 '디버그 모드' 결과를 확인해서 API가 predictions를 비어있게 주는지, "
            "아니면 keypoints 구조가 코드가 가정한 것과 다른지 확인해보세요."
        )

        st.stop()


    # ========================================================
    # 14. 상대각
    # ========================================================

    angle_rad = np.deg2rad(
        df["angle_deg"]
    )

    df["angle_unwrapped_deg"] = (
        np.rad2deg(
            np.unwrap(
                angle_rad
            )
        )
    )


    initial_angle = float(
        df[
            "angle_unwrapped_deg"
        ].iloc[0]
    )


    df["relative_angle_deg"] = (
        df[
            "angle_unwrapped_deg"
        ]
        - initial_angle
    )


    # ========================================================
    # 15. 각속도
    # ========================================================

    df["dt"] = (
        df["time_s"].diff()
    )

    df["dtheta"] = (
        df["relative_angle_deg"]
        .diff()
    )

    df[
        "angular_velocity_deg_s"
    ] = (
        df["dtheta"]
        / df["dt"]
    )


    # 이상값 제거
    df.loc[
        df[
            "angular_velocity_deg_s"
        ].abs() > 1000,
        "angular_velocity_deg_s"
    ] = np.nan


    df[
        "angular_velocity_deg_s"
    ] = (
        df[
            "angular_velocity_deg_s"
        ]
        .interpolate(
            limit_direction="both"
        )
    )


    # ========================================================
    # 16. 반응 시작 시간
    # ========================================================

    RESPONSE_THRESHOLD = 3.0

    response_candidates = df.loc[
        df[
            "relative_angle_deg"
        ].abs()
        >= RESPONSE_THRESHOLD,
        "time_s"
    ]


    if len(response_candidates) > 0:

        response_time = float(
            response_candidates.iloc[0]
        )

    else:

        response_time = np.nan


    # ========================================================
    # 17. 주요 결과
    # ========================================================

    max_angle_change = float(
        df[
            "relative_angle_deg"
        ]
        .abs()
        .max()
    )


    max_angular_velocity = float(
        df[
            "angular_velocity_deg_s"
        ]
        .abs()
        .max()
    )


    max_index = int(
        df[
            "relative_angle_deg"
        ]
        .abs()
        .idxmax()
    )


    max_change_time = float(
        df.loc[
            max_index,
            "time_s"
        ]
    )


    # ========================================================
    # 18. 결과 표시
    # ========================================================

    st.success(
        "분석이 완료되었습니다."
    )


    st.subheader(
        "3단계 — 분석 결과"
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "최대 각도 변화",
        f"{max_angle_change:.2f}°"
    )


    col2.metric(
        "최대 각속도",
        f"{max_angular_velocity:.2f}°/s"
    )


    if math.isnan(response_time):

        col3.metric(
            "반응 시작",
            "검출되지 않음"
        )

    else:

        col3.metric(
            "반응 시작",
            f"{response_time:.2f}s"
        )


    st.write(
        f"최대 변화 시점: "
        f"**{max_change_time:.2f}초**"
    )


    # ========================================================
    # 19. 그래프
    # ========================================================

    st.subheader(
        "움직임 그래프"
    )


    fig, ax1 = plt.subplots(
        figsize=(12, 6)
    )


    ax1.plot(
        df["time_s"],
        df["relative_angle_deg"],
        label="Relative Angle"
    )


    ax1.set_xlabel(
        "Time (s)"
    )

    ax1.set_ylabel(
        "Relative Angle (deg)"
    )

    ax1.grid(
        True,
        alpha=0.25
    )


    ax2 = ax1.twinx()


    ax2.plot(
        df["time_s"],
        df[
            "angular_velocity_deg_s"
        ],
        alpha=0.65,
        label="Angular Velocity"
    )


    ax2.set_ylabel(
        "Angular Velocity (deg/s)"
    )


    if not math.isnan(
        response_time
    ):

        ax1.axvline(
            response_time,
            linestyle="--",
            alpha=0.7
        )


    fig.suptitle(
        "Mimosa Motion Analysis"
    )


    fig.tight_layout()


    st.pyplot(fig)


    # ========================================================
    # 20. 데이터 테이블
    # ========================================================

    st.subheader(
        "프레임별 데이터"
    )

    st.dataframe(
        df,
        use_container_width=True
    )


    # ========================================================
    # 21. CSV 다운로드
    # ========================================================

    csv_data = df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


    st.download_button(
        "📊 CSV 다운로드",
        data=csv_data,
        file_name="mimosa_analysis.csv",
        mime="text/csv"
    )


    # ========================================================
    # 22. 분석 영상
    # ========================================================

    st.subheader(
        "AI 추적 결과 영상"
    )


    with open(
        output_video.name,
        "rb"
    ) as f:

        output_video_bytes = f.read()


    st.video(
        output_video_bytes
    )


    st.download_button(
        "🎥 분석 영상 다운로드",
        data=output_video_bytes,
        file_name="mimosa_tracking_result.mp4",
        mime="video/mp4"
    )
