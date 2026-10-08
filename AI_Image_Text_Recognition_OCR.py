import cv2
import pytesseract
import os
import sys
import shutil
# DECODE LABS - ARTIFICIAL INTELLIGENCE INTERNSHIP
# PROJECT 4 - IMAGE / TEXT RECOGNITION
# OCR USING OPENCV + TESSERACT
def setup_tesseract():

    tesseract_from_path = shutil.which("tesseract")
    if tesseract_from_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_from_path
        return True
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(
            r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"
        )
    ]
    for path in possible_paths:

        if os.path.exists(path):

            pytesseract.pytesseract.tesseract_cmd = path
            return True

    return False
def display_header():

    print("\n" + "=" * 75)
    print("                 AI IMAGE / TEXT RECOGNITION")
    print("                            PROJECT 4")
    print("                       Decode Labs Internship")
    print("=" * 75)
    print("\nMethod:")
    print("Optical Character Recognition (OCR)")
    print("\nTechnology:")
    print("Python + OpenCV + Tesseract OCR")
def load_image(image_path):

    print("\n" + "-" * 75)
    print("STEP 1 — IMAGE LOADING")
    print("-" * 75)
    if not os.path.exists(image_path):
        print("\n❌ ERROR: Image file not found.")
        print("Please check the path and try again.")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print("\n❌ ERROR: OpenCV could not read the image.")
        return None
    print("\n✓ Image loaded successfully.")
    print(f"File   : {image_path}")
    print(f"Width  : {image.shape[1]} pixels")
    print(f"Height : {image.shape[0]} pixels")
    return image
def preprocess_image(image):
    print("\n" + "-" * 75)
    print("STEP 2 — IMAGE PREPROCESSING")
    print("-" * 75)
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    print("✓ Grayscale conversion completed.")
    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )
    print("✓ Gaussian blur completed.")
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    print("✓ Adaptive thresholding completed.")

    return grayscale, thresholded

def perform_ocr(processed_image):

    print("\n" + "-" * 75)
    print("STEP 3 — OCR TEXT RECOGNITION")
    print("-" * 75)

    print("\nRunning Tesseract OCR...")
    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(
        processed_image,
        config=config
    )

    data = pytesseract.image_to_data(
        processed_image,
        config=config,
        output_type=pytesseract.Output.DICT
    )
    confidence_values = []

    for confidence in data["conf"]:

        try:

            confidence = float(confidence)

            if confidence >= 0:
                confidence_values.append(confidence)

        except ValueError:
            continue

    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        average_confidence = 0.0

    print("✓ OCR processing completed.")

    return (
        text.strip(),
        average_confidence,
        data
    )

def display_result(text, confidence):

    print("\n" + "=" * 75)
    print("                         OCR RESULT")
    print("=" * 75)

    if text:

        print("\nRecognized Text")
        print("-" * 75)

        print(text)

    else:

        print("\n❌ No readable text was detected.")

    print("\n" + "-" * 75)

    print(
        f"OCR Confidence : {confidence:.2f}%"
    )

    if confidence >= 80:

        print("Validation     : PASSED ✓")
        print(
            "The OCR confidence meets the 80% project threshold."
        )

    else:

        print("Validation     : BELOW 80%")
        print(
            "Try using a clearer image with better lighting."
        )

    print("=" * 75)

def create_visual_output(image, data):

    print("\n" + "-" * 75)
    print("STEP 4 — VISUAL CONFIRMATION")
    print("-" * 75)
    output = image.copy()

    detected_regions = 0

    for i in range(len(data["text"])):

        detected_text = data["text"][i].strip()

        try:

            confidence = float(
                data["conf"][i]
            )

        except ValueError:

            confidence = 0

        if detected_text and confidence >= 80:

            x = data["left"][i]
            y = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )
            cv2.putText(
                output,
                detected_text,
                (x, max(y - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detected_regions += 1

    print(
        f"\n✓ Visual confirmation completed."
    )

    print(
        f"Detected text regions : {detected_regions}"
    )

    return output

def save_results(
    text,
    grayscale,
    processed_image,
    visual_output
):
    print("\n" + "-" * 75)
    print("STEP 5 — SAVING RESULTS")
    print("-" * 75)

    output_folder = "output"

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    text_file = os.path.join(
        output_folder,
        "recognized_text.txt"
    )

    with open(
        text_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(text)

    print(
        f"\n✓ Recognized text:"
        f"\n  {text_file}"
    )

    grayscale_file = os.path.join(
        output_folder,
        "grayscale_image.png"
    )

    cv2.imwrite(
        grayscale_file,
        grayscale
    )

    print(
        f"\n✓ Grayscale image:"
        f"\n  {grayscale_file}"
    )

    processed_file = os.path.join(
        output_folder,
        "processed_image.png"
    )

    cv2.imwrite(
        processed_file,
        processed_image
    )

    print(
        f"\n✓ Processed image:"
        f"\n  {processed_file}"
    )

    visual_file = os.path.join(
        output_folder,
        "ocr_visual_result.png"
    )

    cv2.imwrite(
        visual_file,
        visual_output
    )

    print(
        f"\n✓ Visual OCR result:"
        f"\n  {visual_file}"
    )

def display_summary(
    text,
    confidence
):
    print("\n" + "=" * 75)
    print("                       PROJECT SUMMARY")
    print("=" * 75)

    print("\nProject:")
    print("AI Image / Text Recognition")

    print("\nRecognition Method:")
    print("Optical Character Recognition (OCR)")

    print("\nLibraries:")
    print("• OpenCV")
    print("• Pytesseract")
    print("• Tesseract OCR")

    print("\nProcessing Pipeline:")
    print(
        "Image → Grayscale → Gaussian Blur → "
        "Adaptive Threshold → OCR"
    )

    print(
        f"\nAverage OCR Confidence: {confidence:.2f}%"
    )

    if confidence >= 80:

        print("Project Validation: PASSED ✓")

    else:

        print("Project Validation: BELOW 80%")

    if text:

        print("Text Recognition: SUCCESS ✓")

    else:

        print("Text Recognition: NO TEXT DETECTED")

    print("\nGenerated Output:")
    print("• recognized_text.txt")
    print("• grayscale_image.png")
    print("• processed_image.png")
    print("• ocr_visual_result.png")

    print("\nAll output files are stored inside:")
    print("output\\")

    print("=" * 75)

def main():

    if not setup_tesseract():

        print("\n" + "=" * 75)
        print("              TESSERACT OCR NOT FOUND")
        print("=" * 75)

        print("\nTesseract OCR is required for Project 4.")

        print("\nInstall Tesseract OCR first, then run")
        print("this program again.")

        print("\nCommon installation path:")
        print(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        print("\n" + "=" * 75)

        sys.exit()

    display_header()

    print("\n" + "-" * 75)
    print("INPUT IMAGE")
    print("-" * 75)

    print(
        "\nPlace your image inside the Project 4 folder."
    )

    print("\nExample:")
    print(
        r"F:\DECODE LABS PROJECTS\project 4\sample.png"
    )

    image_path = input(
        "\nEnter image path: "
    ).strip()

    # Remove quotes if the user pasted a quoted path
    image_path = image_path.strip('"')

    image = load_image(
        image_path
    )

    if image is None:

        return

    grayscale, processed_image = preprocess_image(
        image
    )

    text, confidence, data = perform_ocr(
        processed_image
    )

    display_result(
        text,
        confidence
    )

    visual_output = create_visual_output(
        image,
        data
    )

    save_results(
        text,
        grayscale,
        processed_image,
        visual_output
    )

    display_summary(
        text,
        confidence
    )

    print(
        "\n✓ Project 4 execution completed successfully."
    )

    print("=" * 75)

if __name__ == "__main__":

    main()