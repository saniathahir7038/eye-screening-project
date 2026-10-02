"""Streamlit user interface for the pterygium screening research prototype."""
from pathlib import Path

import streamlit as st

from framing import auto_frame
from screening import (DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, ImageInputError,
                       assess_image_quality, decode_uploaded_image, load_runtime, run_screening)


st.set_page_config(page_title="Pterygium screening prototype", page_icon=None, layout="wide")


@st.cache_resource(show_spinner=False)
def cached_runtime(model_path: str, summary_path: str):
    return load_runtime(Path(model_path), Path(summary_path))


st.title("Pterygium screening prototype")
st.caption("Academic research interface for an external-eye photograph")
st.info(
    "Experimental model update: its output head was adapted with a small web-image dataset. "
    "It reduced false positives in exploratory testing but missed one labeled pterygium case. "
    "This is not a diagnosis and has not been clinically validated for smartphone photographs."
)

with st.expander("Photograph guidance", expanded=True):
    st.markdown(
        "1. Ask someone to use the rear camera if possible. Hold the phone level so the eye looks horizontal, not sideways.\n"
        "2. Look straight ahead. Keep the whole iris (coloured circle) and visible white on both sides in frame; let one eye fill most of the photo.\n"
        "3. Use bright, even light. Avoid direct flash, deep shadows and bright reflections over the eye.\n"
        "4. Tap the eye to focus, hold still and avoid beauty filters.\n"
        "5. Check the photo: retake it if the eye is sideways, cut off, blurry or covered by glare.\n\n"
        "Remove identifying text before uploading. PNG and JPEG files up to 15 MB are accepted."
    )

upload = st.file_uploader("Upload an external-eye photograph", type=("png", "jpg", "jpeg"))

if upload is None:
    st.write("Upload an image to begin the quality check.")
    st.stop()

try:
    uploaded_image = decode_uploaded_image(upload.getvalue())
except ImageInputError as error:
    st.error(str(error))
    st.stop()

framing = auto_frame(uploaded_image)
left, right = st.columns((1.05, 1), gap="large")
with left:
    st.subheader("Uploaded photograph")
    st.image(uploaded_image, width="stretch")
    if framing.adjusted:
        st.subheader("Automatically framed photograph")
        st.image(framing.image, width="stretch")
        st.caption(f"Removed {framing.removed_bottom_fraction:.0%} of a low-detail lower margin. The complete width is retained.")
    else:
        st.caption("Full photo used; no automatic crop was applied.")

quality = assess_image_quality(framing.image)
with right:
    st.subheader("Technical quality check")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Brightness", f"{quality.brightness:.1f}")
    metric_columns[1].metric("Contrast", f"{quality.contrast:.1f}")
    metric_columns[2].metric("Sharpness", f"{quality.sharpness:.1f}")
    st.caption(f"Screened area: {quality.width} × {quality.height} pixels")
    if quality.accepted:
        st.success("Basic technical checks passed. Confirm the eye is correctly framed in the preview.")
    else:
        st.error("Image quality insufficient — retake photograph")
        for issue in quality.issues:
            st.write(f"- {issue}")
        st.caption("No model result is produced when a technical quality check fails.")

if not quality.accepted:
    st.stop()

try:
    with st.spinner("Analysing the image..."):
        runtime = cached_runtime(str(DEFAULT_MODEL_PATH), str(DEFAULT_SUMMARY_PATH))
        result = run_screening(runtime, uploaded_image, framing)
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"The screening model could not be loaded or executed: {error}")
    st.stop()

st.divider()
st.subheader("Screening result")
if result.inconclusive:
    st.warning("Unable to screen this photo reliably — the decision changed after automatic framing. Retake a closer eye photograph.")
elif result.suspected:
    st.warning("Suspected pterygium — professional eye examination recommended")
else:
    st.success("No visible pterygium pattern")

score_columns = st.columns(3 if result.full_photo_score is not None else 2)
score_columns[0].metric("Framed model score" if result.framing.adjusted else "Model score",
                        f"{result.model_score:.3f}")
if result.full_photo_score is not None:
    score_columns[1].metric("Full-photo score", f"{result.full_photo_score:.3f}")
score_columns[-1].metric("Decision threshold", f"{result.threshold:.3f}")
st.caption("The model score is not a calibrated probability or a measure of disease severity.")

preview, explanation = st.columns(2, gap="large")
with preview:
    st.subheader("Model input")
    st.image(result.processed_image, width="stretch")
    st.caption("The upper-left corner is standardized to match model training.")
with explanation:
    st.subheader("Grad-CAM attention map")
    st.image(result.heatmap_overlay, width="stretch")
    st.caption("Warm colours show image regions that influenced the score. This is not a lesion boundary.")

with st.expander("Limitations"):
    st.markdown(
        "- The quality gate checks only resolution, aspect ratio, brightness, contrast and blur.\n"
        "- It cannot confirm that an eye is correctly centred or clinically suitable.\n"
        "- The model uses slit-lamp training and a small web-image adaptation set.\n"
        "- The adaptation test was exploratory and did not establish smartphone accuracy.\n"
        "- Smartphone performance requires separate data and independent validation.\n"
        "- Seek professional care for symptoms or concerns regardless of this result."
    )
