# AI Image / Text Recognition — OCR

## Decode Labs Internship — Project 4

This project implements a basic **Optical Character Recognition (OCR)** system using Python, OpenCV, and Tesseract OCR.

The system takes an image containing text as input, preprocesses the image using computer vision techniques, extracts text using Tesseract OCR, calculates the average OCR confidence, and generates visual and text-based results.

## Project Objective

The objective of this project is to build a basic image/text recognition system capable of extracting text from images using OCR techniques.

The project focuses on:

- Image loading
- Image preprocessing
- Grayscale conversion
- Gaussian blur
- Adaptive thresholding
- Optical Character Recognition
- OCR confidence calculation
- Visual confirmation of detected text
- Saving processed results

## Technologies Used

- Python
- OpenCV
- Pytesseract
- Tesseract OCR
- Pillow

## OCR Processing Pipeline

```text
Input Image
     ↓
Image Loading
     ↓
Grayscale Conversion
     ↓
Gaussian Blur
     ↓
Adaptive Thresholding
     ↓
Tesseract OCR
     ↓
Confidence Calculation
     ↓
Visual Confirmation
     ↓
Output Generation