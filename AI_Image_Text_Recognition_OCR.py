import cv2
import pytesseract
import os
import sys
import shutil

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
    print("\n" + "=" * 80)
    print("                  AI IMAGE / TEXT RECOGNITION")
    print("                             PROJECT 4")
    print("                        Decode Labs Internship")
    print("=" * 80)

    print("\nRecognition Method:")
    print("Optical Character Recognition (OCR)")

    print("\nTechnology:")
    print("Python + OpenCV + Tesseract OCR")

def load_image(image_path):

    print("\n" + "-" * 80)
    print("STEP 1 — IMAGE LOADING")
    print("-" * 80)

    if not os.path.exists(image_path):
        print("\n❌ ERROR: Image file not found.")
        print("Please check the image path.")
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

    print("\n" + "-" * 80)
    print("STEP 2 — IMAGE PREPROCESSING")
    print("-" * 80)

    scale_factor = 2

    resized = cv2.resize(
        image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )

    print("✓ Image upscaling completed.")

    grayscale = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    print("✓ Grayscale conversion completed.")

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )

    print("✓ Gaussian blur completed.")

    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print("✓ Otsu thresholding completed.")

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    print("✓ Adaptive thresholding completed.")

    inverted = cv2.bitwise_not(otsu)

    print("✓ Inverted threshold image generated.")

    sharpening_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    sharpened = cv2.morphologyEx(
        otsu,
        cv2.MORPH_CLOSE,
        sharpening_kernel
    )

    print("✓ Image sharpening completed.")

    processed_images = {
        "Otsu Threshold": otsu,
        "Adaptive Threshold": adaptive,
        "Inverted Threshold": inverted,
        "Sharpened": sharpened
    }

    return grayscale, processed_images

def perform_single_ocr(image, psm_mode):

    config = f"--oem 3 --psm {psm_mode}"

    text = pytesseract.image_to_string(
        image,
        config=config
    )

    data = pytesseract.image_to_data(
        image,
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

    return (
        text.strip(),
        average_confidence,
        data
    )
def perform_ocr(processed_images):

    print("\n" + "-" * 80)
    print("STEP 3 — OCR TEXT RECOGNITION")
    print("-" * 80)

    print("\nRunning multiple OCR configurations...")

    results = []

    psm_modes = [6, 7, 11]

    for image_name, image in processed_images.items():

        for psm_mode in psm_modes:

            text, confidence, data = perform_single_ocr(
                image,
                psm_mode
            )

            results.append({
                "method": image_name,
                "psm": psm_mode,
                "text": text,
                "confidence": confidence,
                "data": data,
                "image": image
            })

            print(
                f"✓ {image_name:<20} "
                f"PSM {psm_mode:<2} "
                f"Confidence: {confidence:.2f}%"
            )

    valid_results = [
        result
        for result in results
        if result["text"]
    ]

    if valid_results:

        best_result = max(
            valid_results,
            key=lambda result: result["confidence"]
        )

    else:

        best_result = {
            "method": "None",
            "psm": 6,
            "text": "",
            "confidence": 0.0,
            "data": {
                "text": [],
                "conf": [],
                "left": [],
                "top": [],
                "width": [],
                "height": []
            },
            "image": None
        }

    print("\n✓ OCR processing completed.")

    print(
        f"\nBest preprocessing method : "
        f"{best_result['method']}"
    )

    print(
        f"Best PSM mode             : "
        f"{best_result['psm']}"
    )

    print(
        f"Best confidence            : "
        f"{best_result['confidence']:.2f}%"
    )

    return best_result

def display_result(result):

    text = result["text"]
    confidence = result["confidence"]

    print("\n" + "=" * 80)
    print("                           OCR RESULT")
    print("=" * 80)

    if text:

        print("\nRecognized Text")
        print("-" * 80)
        print(text)

    else:

        print("\n❌ No readable text was detected.")

    print("\n" + "-" * 80)

    print(
        f"OCR Confidence : "
        f"{confidence:.2f}%"
    )

    if confidence >= 80:

        print("Validation     : PASSED ✓")
        print(
            "The OCR confidence meets the "
            "80% project threshold."
        )

    else:

        print("Validation     : BELOW 80%")
        print(
            "Try using a clearer image with "
            "better lighting and larger text."
        )

    print("=" * 80)

def create_visual_output(image, data):

    print("\n" + "-" * 80)
    print("STEP 4 — VISUAL CONFIRMATION")
    print("-" * 80)

    output = image.copy()

    detected_regions = 0

    for i in range(len(data["text"])):

        detected_text = data["text"][i].strip()

        try:
            confidence = float(data["conf"][i])

        except ValueError:
            confidence = 0

        if detected_text and confidence >= 80:

            x = data["left"][i]
            y = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]

            cv2.rectangle(
                output,
                (x // 2, y // 2),
                (
                    (x + width) // 2,
                    (y + height) // 2
                ),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                detected_text,
                (
                    x // 2,
                    max((y // 2) - 5, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detected_regions += 1

    print("\n✓ Visual confirmation completed.")

    print(
        f"Detected text regions : "
        f"{detected_regions}"
    )

    return output

def save_results(
    text,
    grayscale,
    processed_images,
    visual_output
):

    print("\n" + "-" * 80)
    print("STEP 5 — SAVING RESULTS")
    print("-" * 80)

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
        f"\n✓ Recognized text:\n  "
        f"{text_file}"
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
        f"\n✓ Grayscale image:\n  "
        f"{grayscale_file}"
    )

    for method_name, image in processed_images.items():

        safe_name = (
            method_name
            .lower()
            .replace(" ", "_")
        )

        processed_file = os.path.join(
            output_folder,
            f"{safe_name}.png"
        )

        cv2.imwrite(
            processed_file,
            image
        )

        print(
            f"\n✓ {method_name} image:\n  "
            f"{processed_file}"
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
        f"\n✓ Visual OCR result:\n  "
        f"{visual_file}"
    )

def display_summary(
    result
):

    text = result["text"]
    confidence = result["confidence"]

    print("\n" + "=" * 80)
    print("                         PROJECT SUMMARY")
    print("=" * 80)

    print("\nProject:")
    print("AI Image / Text Recognition")

    print("\nRecognition Method:")
    print("Optical Character Recognition (OCR)")

    print("\nLibraries:")
    print("• OpenCV")
    print("• Pytesseract")
    print("• Tesseract OCR")
    print("• Pillow")

    print("\nProcessing Pipeline:")
    print(
        "Image → Upscaling → Grayscale → "
        "Noise Reduction → Multiple Thresholding → "
        "OCR → Confidence Analysis"
    )

    print(
        f"\nBest OCR Confidence: "
        f"{confidence:.2f}%"
    )

    if confidence >= 80:

        print(
            "Project Validation: PASSED ✓"
        )

    else:

        print(
            "Project Validation: BELOW 80%"
        )

    if text:

        print(
            "Text Recognition: SUCCESS ✓"
        )

    else:

        print(
            "Text Recognition: "
            "NO TEXT DETECTED"
        )

    print("\nGenerated Output:")

    print("• recognized_text.txt")
    print("• grayscale_image.png")
    print("• otsu_threshold.png")
    print("• adaptive_threshold.png")
    print("• inverted_threshold.png")
    print("• sharpened.png")
    print("• ocr_visual_result.png")

    print("\nAll output files are stored inside:")

    print("output\\")

    print("=" * 80)

def main():
    if not setup_tesseract():

        print("\n" + "=" * 80)
        print("                  TESSERACT OCR NOT FOUND")
        print("=" * 80)

        print(
            "\nTesseract OCR is required "
            "for Project 4."
        )

        print(
            "\nInstall Tesseract OCR first, "
            "then run this program again."
        )

        print(
            "\nCommon installation path:"
        )

        print(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        print("\n" + "=" * 80)

        sys.exit()
    display_header()

    # Input image
    print("\n" + "-" * 80)
    print("INPUT IMAGE")
    print("-" * 80)

    print(
        "\nEnter the image filename "
        "or complete image path."
    )

    image_path = input(
        "\nEnter image path: "
    ).strip()

    image_path = image_path.strip('"')

    image = load_image(
        image_path
    )

    if image is None:
        return
    grayscale, processed_images = preprocess_image(
        image
    )
    best_result = perform_ocr(
        processed_images
    )
    display_result(
        best_result
    )
    visual_output = create_visual_output(
        image,
        best_result["data"]
    )
    save_results(
        best_result["text"],
        grayscale,
        processed_images,
        visual_output
    )
    display_summary(
        best_result
    )

    print(
        "\n✓ Project 4 execution completed successfully."
    )

    print("=" * 80)

if __name__ == "__main__":
    main()