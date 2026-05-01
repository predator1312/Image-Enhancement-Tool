import base64
import io
import os
import numpy as np
import cv2
import pywt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tempfile
from flask import Flask, request, render_template, jsonify, send_file
import threading
import webbrowser

# --- Cubic Spline Interpolation Functions ---
def build_spline_matrix(n, h):
    diagonals = [np.zeros(n-1), np.zeros(n), np.zeros(n-1)]
    diagonals[1][0] = diagonals[1][-1] = 1
    for i in range(1, n-1):
        diagonals[0][i-1] = h[i-1]
        diagonals[1][i] = 2 * (h[i-1] + h[i])
        diagonals[2][i] = h[i]
    return sp.diags(diagonals, offsets=[-1, 0, 1], format='csr')

def cubic_spline(n, x, y):
    h = np.diff(x)
    A = build_spline_matrix(n, h)
    rhs = np.zeros(n)
    for i in range(1, n-1):
        rhs[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    c = spla.spsolve(A, rhs)
    coeffs = np.zeros((n-1, 4))
    for i in range(n-1):
        coeffs[i, 0] = y[i]
        coeffs[i, 1] = (y[i+1] - y[i]) / h[i] - (h[i] * (2 * c[i] + c[i+1])) / 3
        coeffs[i, 2] = c[i]
        coeffs[i, 3] = (c[i+1] - c[i]) / (3 * h[i])
    return coeffs

def interpolate(xval, n, x, y, coeffs):
    if xval < x[0] or xval > x[-1]:
        return 0
    left, right = 0, n-2
    while left <= right:
        mid = (left + right) // 2
        if x[mid] <= xval < x[mid+1]:
            dx = xval - x[mid]
            return (coeffs[mid, 0] + coeffs[mid, 1] * dx +
                    coeffs[mid, 2] * dx**2 + coeffs[mid, 3] * dx**3)
        if xval < x[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return y[-1]

def interpolate_image(img, new_rows, new_cols):
    rows, cols = img.shape
    new_img = np.zeros((new_rows, new_cols), dtype=np.float32)
    for i in range(rows):
        x = np.linspace(0, cols-1, cols)
        y = img[i, :]
        coeffs = cubic_spline(cols, x, y)
        x_new = np.linspace(0, cols-1, new_cols)
        for j, x_val in enumerate(x_new):
            new_img[i, j] = interpolate(x_val, cols, x, y, coeffs)
    for j in range(new_cols):
        x = np.linspace(0, rows-1, rows)
        y = new_img[:rows, j]
        coeffs = cubic_spline(rows, x, y)
        x_new = np.linspace(0, rows-1, new_rows)
        for i, x_val in enumerate(x_new):
            new_img[i, j] = interpolate(x_val, rows, x, y, coeffs)
    return new_img

# --- Thresholding Functions ---
def hard_threshold(coeffs, threshold):
    return np.where(np.abs(coeffs) >= threshold, coeffs, 0)

def soft_threshold(coeffs, threshold):
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)

def estimate_noise_sigma(hh):
    return np.median(np.abs(hh)) / 0.6745

# --- Utility Functions ---
def pad_image(image):
    h, w = image.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return image, (pad_h, pad_w)

def unpad_image(image, pad):
    if pad[0] or pad[1]:
        image = image[:-pad[0], :-pad[1]]
    return image

def equalize_histogram_clahe(image):
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    y_clahe = clahe.apply(y)
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    return cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

# --- Enhanced Image Processing ---
def process_image(img_data, use_clahe, use_hard_threshold):
    # Decode image from base64
    img_bytes = base64.b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to RGB for processing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Split into channels
    channels = cv2.split(img)
    
    padded_channels = []
    swt_coeffs = []
    dwt_coeffs = []
    for ch in channels:
        ch_padded, pad = pad_image(ch)
        padded_channels.append((ch_padded, pad))
        
        # SWT with bior1.1
        swt = pywt.swt2(ch_padded, 'bior1.1', level=1)
        swt_coeffs.append(swt)
        
        # DWT with bior1.1
        dwt = pywt.dwt2(ch_padded, 'bior1.1')
        dwt_coeffs.append(dwt)
    
    difference_images = []
    for i, (ch_padded, _) in enumerate(padded_channels):
        LL_swt = swt_coeffs[i][0][0]
        diff = ch_padded - LL_swt
        difference_images.append(diff)
    
    interpolated_LH, interpolated_HL, interpolated_HH = [], [], []
    for i in range(3):
        LH_dwt = dwt_coeffs[i][1][0]
        HL_dwt = dwt_coeffs[i][1][1]
        HH_dwt = dwt_coeffs[i][1][2]
        h, w = LH_dwt.shape
        LH_interp = interpolate_image(LH_dwt, 2*h, 2*w)
        HL_interp = interpolate_image(HL_dwt, 2*h, 2*w)
        HH_interp = interpolate_image(HH_dwt, 2*h, 2*w)
        interpolated_LH.append(LH_interp)
        interpolated_HL.append(HL_interp)
        interpolated_HH.append(HH_interp)
    
    # Option 4: diff + LH_interp + LH_swt (using hard or soft threshold)
    final_LH, final_HL, final_HH = [], [], []
    for i in range(3):
        LH_swt = swt_coeffs[i][0][1][0]
        HL_swt = swt_coeffs[i][0][1][1]
        HH_swt = swt_coeffs[i][0][1][2]
        diff = difference_images[i]
        LH_interp = interpolated_LH[i]
        HL_interp = interpolated_HL[i]
        HH_interp = interpolated_HH[i]
        
        sigma = estimate_noise_sigma(HH_swt)
        N = HH_swt.size
        threshold = sigma * np.sqrt(2 * np.log(N))
        
        if use_hard_threshold:
            LH_interp_thresh = hard_threshold(LH_interp, threshold)
            HL_interp_thresh = hard_threshold(HL_interp, threshold)
            HH_interp_thresh = hard_threshold(HH_interp, threshold)
            LH_swt_thresh = hard_threshold(LH_swt, threshold)
            HL_swt_thresh = hard_threshold(HL_swt, threshold)
            HH_swt_thresh = hard_threshold(HH_swt, threshold)
        else:
            LH_interp_thresh = soft_threshold(LH_interp, threshold)
            HL_interp_thresh = soft_threshold(HL_interp, threshold)
            HH_interp_thresh = soft_threshold(HH_interp, threshold)
            LH_swt_thresh = soft_threshold(LH_swt, threshold)
            HL_swt_thresh = soft_threshold(HL_swt, threshold)
            HH_swt_thresh = soft_threshold(HH_swt, threshold)
        
        # Option 4: diff + LH_interp + LH_swt
        final_LH.append(diff + LH_interp_thresh * 1.2 + LH_swt_thresh * 1.2)
        final_HL.append(diff + HL_interp_thresh * 1.2 + HL_swt_thresh * 1.2)
        final_HH.append(diff + HH_interp_thresh * 1.2 + HH_swt_thresh * 1.2)
    
    reconstructed_channels = []
    for i, (ch_padded, pad) in enumerate(padded_channels):
        coeffs = [(ch_padded, (final_LH[i], final_HL[i], final_HH[i]))]
        recon = pywt.iswt2(coeffs, 'bior1.1')  # ISWT with bior1.1
        recon = np.clip(recon, 0, 255).astype(np.uint8)
        recon = unpad_image(recon, pad)
        reconstructed_channels.append(recon)
    
    final_img = cv2.merge(reconstructed_channels)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    
    if use_clahe:
        final_img = equalize_histogram_clahe(final_img)
    
    # Convert original image back to BGR for display
    original_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert images to base64 for web display
    _, buffer_original = cv2.imencode('.png', original_img)
    _, buffer_enhanced = cv2.imencode('.png', final_img)
    
    img_original_base64 = 'data:image/png;base64,' + base64.b64encode(buffer_original).decode('utf-8')
    img_enhanced_base64 = 'data:image/png;base64,' + base64.b64encode(buffer_enhanced).decode('utf-8')
    
    return {
        'original': img_original_base64,
        'enhanced': img_enhanced_base64
    }

# --- Flask Web Application ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    img_data = data['image']
    use_clahe = data['clahe'] == 'yes'
    use_hard_threshold = data['threshold'] == 'hard'
    
    try:
        result = process_image(img_data, use_clahe, use_hard_threshold)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def open_browser():
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start browser after a short delay
    threading.Timer(1.0, open_browser).start()
    
    # Run the Flask app
    app.run(debug=False)