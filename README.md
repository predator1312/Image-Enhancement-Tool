# Image Enhancement Tool

This is a web-based application that improves image quality using wavelet transform techniques and contrast enhancement. It provides a simple interface where users can upload an image, process it, and compare the result instantly.

---

## What this project does

- Upload an image from your system  
- Enhance image using wavelet transform  
- Reduce noise using thresholding (hard/soft)  
- Optionally apply CLAHE for better contrast  
- View original and enhanced images side by side  
- Download the enhanced image  

---

## Tech stack

- Python (Flask)
- OpenCV
- NumPy
- PyWavelets
- SciPy
- HTML, CSS, JavaScript

---

## How to run the project

### 1. Clone the repository

```
git clone https://github.com/predator1312/Image-Enhancement-Tool.git
```

### 2. Go to project folder

```
cd Image-Enhancement-Tool
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the app

```
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## Project structure

```
Image-Enhancement-Tool/
│
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── .gitignore
└── README.md
```

---

## Future improvements

- Drag and drop image upload  
- Faster processing  
- Support for multiple images  
- Deploy as an online tool  

---

## Author

Kushal Sharma  
Rajarajeswari College of Engineering