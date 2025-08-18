import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import plotly.express as px
import tensorflow as tf

from constants import *
from versions.ocr_standard import StandardDigitDrawGUI
from versions.ocr_optimized import OptimizedDigitDrawGUI

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'current_version' not in st.session_state:
        st.session_state.current_version = "optimized"
    
    if 'performance_stats' not in st.session_state:
        st.session_state.performance_stats = {"standard": [], "optimized": []}
    
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    
    if 'ocr_standard' not in st.session_state:
        st.session_state.ocr_standard = None
    
    if 'ocr_optimized' not in st.session_state:
        st.session_state.ocr_optimized = None

    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None

    if 'placeholders' not in st.session_state:
        st.session_state.placeholders = {
            "predictions": "🎯 Draw some digits on the canvas to see OCR in action!",
            "time": "⏱️ Time: --",
            "digits": "🔢 Digits: --"
        }

def initialize_ocr_processors(model):
    """Initialize OCR processors"""
    if st.session_state.ocr_standard is None:
        st.session_state.ocr_standard = StandardDigitDrawGUI(model, show_ui=False)
    
    if st.session_state.ocr_optimized is None:
        st.session_state.ocr_optimized = OptimizedDigitDrawGUI(model, show_ui=False)

def canvas_to_pil_image(canvas_data):
    """Convert Streamlit canvas data to PIL Image in the format expected by OCR classes"""
    if canvas_data is None or canvas_data.image_data is None:
        return None
    
    # Convert canvas data to PIL Image
    img_array = np.array(canvas_data.image_data)
    
    # Convert RGBA to grayscale
    if img_array.shape[2] == 4:  # RGBA
        # Create alpha mask for transparency
        alpha = img_array[:, :, 3]
        rgb = img_array[:, :, :3]
        
        # Convert to grayscale considering alpha
        gray = np.dot(rgb, [0.299, 0.587, 0.114])
        # Where alpha is 0 (transparent), set to black (0)
        # Where alpha > 0, keep the grayscale value
        gray = np.where(alpha > 0, gray, 0)
    else:
        gray = np.dot(img_array, [0.299, 0.587, 0.114])
    
    # Convert to PIL Image in 'L' (grayscale) mode
    pil_image = Image.fromarray(gray.astype(np.uint8), 'L')
    
    return pil_image

def predict_nums(canvas_data, version):
    """Predict digits on canvas image"""    
    # Convert canvas to PIL image
    pil_image = canvas_to_pil_image(canvas_data)
    
    # Get the appropriate OCR processor
    if version == "optimized":
        processor = st.session_state.ocr_optimized
    else:
        processor = st.session_state.ocr_standard
    
    # Set the image in the processor (simulating the drawing state)
    processor.image = pil_image
    
    # Time the processing using the existing predict_and_display method
    start_time = time.time()

    digit_preds, formatted_groups = processor.predict_nums()
        
    end_time = time.time()
    prediction_time = (end_time - start_time) * 1000  # Convert to ms
            
    return digit_preds, formatted_groups, prediction_time
    
def update_performance_stats(prediction_time):
    """Update performance statistics"""
    current_version = st.session_state.current_version
    stats = st.session_state.performance_stats[current_version]
    stats.append(prediction_time)
    
    # Keep only last 50 measurements
    if len(stats) > 50:
        stats.pop(0)

def create_performance_chart():
    """Create performance comparison chart"""
    stats = st.session_state.performance_stats
    
    if not any(stats.values()):
        st.info("No performance data available yet. Draw some digits to see comparisons!")
        return
    
    # Prepare data for plotting
    data = []
    for version, times in stats.items():
        for i, time_val in enumerate(times):
            data.append({
                'Version': version.title(),
                'Measurement': i + 1,
                'Time (ms)': time_val
            })
    
    if not data:
        st.info("No performance data available yet.")
        return
    
    df = pd.DataFrame(data)
    
    # Create performance comparison chart
    fig = px.line(df, x='Measurement', y='Time (ms)', color='Version',
                  title='Performance Comparison Over Time',
                  labels={'Time (ms)': 'Prediction Time (ms)'})
    
    fig.update_layout(
        yaxis_title="Prediction Time (ms)",
        xaxis_title="Measurement Number",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics table
    if len(df) > 0:
        stats_summary = df.groupby('Version')['Time (ms)'].agg([
            'count', 'mean', 'min', 'max', 'std'
        ]).round(2)
        
        stats_summary.columns = ['Count', 'Average', 'Min', 'Max', 'Std Dev']
        
        st.subheader("📊 Performance Statistics")
        st.dataframe(stats_summary)
        
        # Calculate speedup if both versions have data
        if 'Standard' in stats_summary.index and 'Optimized' in stats_summary.index:
            std_avg = stats_summary.loc['Standard', 'Average']
            opt_avg = stats_summary.loc['Optimized', 'Average']
            speedup = std_avg / opt_avg if opt_avg > 0 else 0
            
            if speedup > 1:
                st.success(f"🚀 **Optimized version is {speedup:.2f}x faster!**")
                st.info(f"💾 Time saved: {std_avg - opt_avg:.1f} ms per prediction")
            elif speedup < 1:
                st.warning(f"📊 Standard version is {1/speedup:.2f}x faster")
            else:
                st.info("⚖️ Performance is similar between versions")

def draw_bounding_boxes_on_canvas(canvas_data, digit_preds, formatted_groups):
    """Draw bounding boxes and labels on the canvas image"""
    if canvas_data is None or canvas_data.image_data is None or not digit_preds:
        return canvas_data.image_data
    
    # Convert canvas data to PIL Image for drawing
    img_array = np.array(canvas_data.image_data)
    pil_image = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil_image)

    # Draw grouped number bounding boxes (red) for multi-digit numbers
    for number_str, avg_conf, (gx0, gy0, gx1, gy1) in formatted_groups:
        if len(number_str) > 1:  # Only draw for multi-digit numbers
            # Draw bounding box
            draw.rectangle([gx0, gy0, gx1, gy1], outline=(255, 0, 0), width=3)
            
            # Draw number label
            label_y = gy0 - 10 if gy0 > 15 else gy1 + 10
            
            draw.text(((gx0 + gx1)/2, label_y), number_str, fill=(255, 0, 0))
    
    # Draw individual digit bounding boxes (green)
    for digit, conf, (x0, y0, x1, y1) in digit_preds:
        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
        
        # Draw digit label
        label = str(digit) if isinstance(digit, int) else digit
        label_y = y0 - 10 if y0 > 15 else y1 + 10
        
        draw.text(((x0 + x1)/2, label_y), label, fill=(0, 255, 0))
    
    return np.array(pil_image)

def main():
    st.set_page_config(
        page_title="OCR Performance Comparison",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("OCR Performance Comparison")
    st.markdown("**Compare Optimized vs Standard OCR Processing Performance**")

    with st.expander("🔍 Difference Between Optimized and Standard Implementations"):
        st.write(
            """
            The optimized version includes several key improvements:\n
            1. Batch processing: Multiple digits processed simultaneously instead of one-by-one\n
            2. Vectorized operations: NumPy-based calculations for faster preprocessing\n
            3. Optimized grouping: Graph-based digit grouping with vectorized distance calculations
            """
        )
    
    # Load model
    if 'model' not in st.session_state:
        with st.spinner("Loading model..."):
            model_path = 'src/model/CNN_model.keras'
            st.session_state.model = tf.keras.models.load_model(model_path)

    
    # Initialize OCR processors with the loaded model
    initialize_ocr_processors(st.session_state.model)
        
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Version selector
        current_version = st.selectbox(
            "OCR Version",
            ["optimized", "standard"],
            index=0 if st.session_state.current_version == "optimized" else 1,
            format_func=lambda x: f"🚀 Optimized" if x == "optimized" else "📊 Standard"
        )
        
        if current_version != st.session_state.current_version:
            st.session_state.current_version = current_version
            st.session_state.canvas_key += 1  # Force canvas refresh

        eraser_mode = st.radio("Mode", ["Draw", "Erase"], horizontal=True, index=0)

        if eraser_mode == "Draw":
            stroke_color = "white"
            stroke_width = STANDARD_STROKE_WIDTH
        elif eraser_mode == "Erase":
            stroke_color = "black"
            stroke_width = ERASER_MULTIPLIER * STANDARD_STROKE_WIDTH
        
        # Clear performance stats
        if st.button("🗑️ Clear Performance Data"):
            st.session_state.performance_stats = {"standard": [], "optimized": []}
            st.rerun()
        
        # Instructions
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Draw digits on the canvas below
        2. Switch between versions to compare performance
        3. View real-time metrics and detailed statistics
        """)
        
        # Performance info
        stats = st.session_state.performance_stats[current_version]
        if stats:
            recent_times = stats[-5:]
            avg_time = sum(recent_times) / len(recent_times)
            
            st.markdown("### ⚡ Recent Performance")
            st.metric("Average Time (last 5)", f"{avg_time:.1f} ms")
            
            if len(stats) >= 2:
                st.metric("Last Prediction", f"{stats[-1]:.1f} ms")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"✏️ Drawing Canvas - {current_version.title()} Version")

        # Placeholders for detected results (displayed above the canvas)
        res_col1, res_col2, res_col3 = st.columns([2, 1, 1])
        with res_col1:
            detected_placeholder = st.success(st.session_state.placeholders["predictions"])
        with res_col2:
            time_placeholder = st.info(st.session_state.placeholders["time"])
        with res_col3:
            digits_placeholder = st.info(st.session_state.placeholders["digits"])

        
        # Drawing canvas with reduced size for web
        canvas_size = min(CANVAS_SIZE, 512)  # Cap at 512 for web performance
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="black",
            width=canvas_size,
            height=canvas_size,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            display_toolbar=True
        )
        
        # Process canvas if there's drawing
        if canvas_result and canvas_result.image_data is not None:
            # Check if canvas has any drawing
            img_array = np.array(canvas_result.image_data)
            if img_array[:, :, 3].sum() > 0:  # Check alpha channel for any drawing
                
                with st.spinner(f"Processing with {current_version} version..."):
                    digit_preds, formatted_groups, prediction_time = predict_nums(canvas_result, current_version)
                
                # Update performance stats
                if prediction_time > 0:
                    update_performance_stats(prediction_time)

                st.session_state.annotated_image = draw_bounding_boxes_on_canvas(canvas_result, digit_preds, formatted_groups)
                
                # Display results
                if digit_preds and formatted_groups:
                    numbers = [group[0] for group in formatted_groups]

                    # Update the placeholders defined above the canvas
                    try:
                        detected_placeholder.success(f"🎯 **Detected Numbers:** {', '.join(numbers) if numbers else 'None'}")
                        time_placeholder.info(f"⏱️ **Time:** {prediction_time:.1f} ms")
                        digits_placeholder.info(f"🔢 **Digits:** {len(digit_preds)}")

                        st.session_state.placeholders["predictions"] = f"🎯 **Detected Numbers:** {', '.join(numbers) if numbers else 'None'}"
                        st.session_state.placeholders["time"] = f"⏱️ **Time:** {prediction_time:.1f} ms"
                        st.session_state.placeholders["digits"] = f"🔢 **Digits:** {len(digit_preds)}"
                    except NameError:
                        # Fallback if placeholders are not in scope for some reason
                        results_col1, results_col2, results_col3 = st.columns([2, 1, 1])
                        with results_col1:
                            st.success(f"🎯 **Detected Numbers:** {', '.join(numbers) if numbers else 'None'}")
                        with results_col2:
                            st.info(f"⏱️ **Time:** {prediction_time:.1f} ms")
                        with results_col3:
                            st.info(f"🔢 **Digits:** {len(digit_preds)}")
                                            
                    # Show individual digit predictions
                    with st.expander("🔍 Individual Digit Predictions"):
                        for i, (digit, conf, bbox) in enumerate(digit_preds):
                            st.write(f"Digit {i+1}: **{digit}** (confidence: {conf:.3f})")
                    
                    # Show detected groups
                    with st.expander("🔗 Grouped Numbers"):
                        for i, (number_str, avg_conf, bbox) in enumerate(formatted_groups):
                            st.write(f"Number {i+1}: **{number_str}** (avg confidence: {avg_conf:.3f})")
                else:
                    try:
                        detected_placeholder.warning("No digits detected. Try drawing larger, clearer digits.")
                        st.session_state.placeholders["predictions"] = "No digits detected. Try drawing larger, clearer digits."
                        if prediction_time > 0:
                            time_placeholder.info(f"⏱️ **Processing Time:** {prediction_time:.1f} ms")
                            st.session_state.placeholders["time"] = f"⏱️ **Processing Time:** {prediction_time:.1f} ms"
                            
                        digits_placeholder.info("🔢 Digits: 0")
                        st.session_state.placeholders["digits"] = "🔢 Digits: 0"
                    except NameError:
                        st.warning("No digits detected. Try drawing larger, clearer digits.")
                        if prediction_time > 0:
                            st.info(f"⏱️ **Processing Time:** {prediction_time:.1f} ms")
            else:
                st.info("👆 Draw some digits on the canvas to see OCR in action!")
        
    with col2:
        st.subheader("📈 Performance Stats")
        
        # Current performance display
        stats = st.session_state.performance_stats[current_version]
        if stats:
            latest_time = stats[-1]
            avg_time = sum(stats) / len(stats)
            
            # Color coding based on performance
            if latest_time < 100:
                color = "green"
            elif latest_time < 300:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"**Latest:** :{color}[{latest_time:.1f} ms]")
            st.markdown(f"**Average:** {avg_time:.1f} ms")
            
            if len(stats) >= 2:
                recent_trend = stats[-1] - stats[-2]
                trend_emoji = "📈" if recent_trend > 0 else "📉"
                st.markdown(f"**Trend:** {trend_emoji} {recent_trend:+.1f} ms")
        else:
            st.info("No data yet for this version")
        
        # Version comparison
        st.markdown("### 🆚 Version Comparison")
        std_stats = st.session_state.performance_stats["standard"]
        opt_stats = st.session_state.performance_stats["optimized"]
        
        if std_stats and opt_stats:
            std_avg = sum(std_stats) / len(std_stats)
            opt_avg = sum(opt_stats) / len(opt_stats)
            
            col_std, col_opt = st.columns(2)
            with col_std:
                st.metric("Standard", f"{std_avg:.1f} ms")
            with col_opt:
                st.metric("Optimized", f"{opt_avg:.1f} ms")
            
            speedup = std_avg / opt_avg if opt_avg > 0 else 0
            if speedup > 1:
                st.success(f"🚀 {speedup:.2f}x speedup!")
        else:
            st.info("Draw with both versions to compare")
        
        # Display annotated image in right column
        st.markdown("### 🎯 Computer Vision Visualization")
        if st.session_state.annotated_image is not None:
            # Scale the image to fit the column width better
            st.image(st.session_state.annotated_image, 
                    caption="Detected Digits with Bounding Boxes", 
                    use_container_width=True)
            
        else:
            st.info("Draw digits to see computer vision visualization")

    # Performance visualization (full width)
    st.markdown("---")
    st.subheader("📊 Detailed Performance Analysis")
    create_performance_chart()
    
if __name__ == "__main__":
    main()