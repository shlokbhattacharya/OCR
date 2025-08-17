from tkinter import messagebox
from tensorflow.keras.models import load_model
from ocr_version_switcher import OCRVersionSwitcher

def main():
    try:
        model = load_model('src/model/CNN_model.keras')
        app = OCRVersionSwitcher(model)
        app.run()
    except Exception as e:
        messagebox.showerror("Error", f"Could not load model: {e}\nPlease check the model path.")

if __name__ == "__main__":
    main()