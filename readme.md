# 🌱 Smart Weed Detection System

## 📌 Project Overview

The **Smart Weed Detection System** is an AI-based agriculture application designed to identify and detect weeds in field images using deep learning and image segmentation techniques.

The system uses a **U-Net deep learning architecture** to analyze uploaded field images and distinguish between crops, weeds, and background/soil. It helps farmers identify weed-affected areas and supports more efficient crop management.

## 🎯 Objectives

- Detect weeds in agricultural field images.
- Differentiate between crops, weeds, and soil.
- Calculate the percentage of weed coverage.
- Display segmentation results using visual overlays.
- Support faster and more accurate weed identification.
- Maintain a history of previous image analyses.

## ⚙️ Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | U-Net |
| Deep Learning Framework | TensorFlow / Keras |
| Backend | Flask |
| Database | SQLite |
| Image Processing | Pillow, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Development Environment | Visual Studio Code |

## 🧠 Model Architecture

The project uses the **U-Net architecture**, a convolutional neural network commonly used for image segmentation.

The model classifies image pixels into three categories:

1. Background / Soil
2. Crop
3. Weed

The segmentation output is processed to identify weed regions and estimate weed coverage.

## ✨ Key Features

- **Image Upload:** Upload agricultural field images for analysis.
- **Weed Detection:** Identify weed regions using the trained U-Net model.
- **Crop Segmentation:** Distinguish crop regions from weeds and soil.
- **Visual Results:** Display segmented output with different colors.
- **Weed Percentage:** Calculate the percentage of detected weed pixels.
- **Weed Severity:** Categorize weed coverage into priority levels.
- **Analysis History:** Store and review previous analysis results.
- **User Authentication:** Provide user registration and login functionality.
- **Database Management:** Store user details and analysis information using SQLite.

## 🔄 System Workflow

1. User uploads an agricultural field image.
2. The application validates the uploaded image.
3. Image preprocessing is performed.
4. The processed image is passed to the trained U-Net model.
5. The model performs pixel-level segmentation.
6. Crop, weed, and background regions are identified.
7. Weed coverage and severity are calculated.
8. The final prediction and segmentation overlay are displayed.
9. Analysis information is stored in the database.

## 📂 Project Structure

```text
Smart-Weed-Detection-System/
│
├── app.py
├── requirements.txt
│
├── models/
│   └── recovered_unet.keras
│
├── templates/
│   ├── login.html
│   ├── register.html
│   └── ...
│
├── statics/
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── ...
│
└── README.md
```

*Note: The structure above is illustrative. Your actual project may contain additional files and folders.*

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sravanalakshmi09/Smart-Weed-Detection-System..git
```

### 2. Navigate to the project folder

```bash
cd Smart-Weed-Detection-System
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

Open the local URL displayed in the terminal in your web browser.

## 📊 Expected Output

The application provides:

- Original uploaded field image.
- Segmented crop and weed regions.
- Weed detection percentage.
- Weed severity or priority level.
- Analysis results and history.

## 🚀 Future Enhancements

- Weed species classification.
- Real-time weed detection using a camera.
- Integration with precision spraying systems.
- Automated weed management recommendations.
- Deployment as a cloud-based application.
- Integration with agricultural robots.

## 👩‍💻 Project Status

The U-Net-based weed detection application has been developed using Flask and TensorFlow. Further testing, optimization, and deployment improvements are ongoing.

## 📜 License

This project is developed for academic and educational purposes.
