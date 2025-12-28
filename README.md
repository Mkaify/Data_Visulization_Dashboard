# 📊 Data Visualization Dashboard

A lightweight, powerful web-based tool built with **FastAPI** and **Python** that allows users to perform end-to-end data exploration. Upload any CSV, clean the data, filter it, and generate interactive visualizations in seconds.

## ✨ Features

-   **Dynamic Ingestion:** Upload any CSV dataset with instant row/column previews.
-   **Session Isolation:** Uses UUID-based session management to support multiple users simultaneously.
-   **Data Cleaning:** Handle missing values (Drop or Fill) with column-specific targeting.
-   **Advanced Filtering:** Query your data using logical operations (>, <, ==, Contains).
-   **Statistical Analysis:** Automated calculation of Mean, Median, Min, and Max for numerical data.
-   **Interactive Charts:** Render Bar, Line, Pie, and Scatter plots using Chart.js.
-   **Data Export:** Download your processed/cleaned dataset as a new CSV.

## 🛠️ Tech Stack

-   **Backend:** FastAPI (Python)
-   **Data Processing:** Pandas
-   **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
-   **Charting:** Chart.js
-   **Deployment Ready:** Configured for Render/Railway

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/data-viz-dashboard.git](https://github.com/YOUR_USERNAME/data-viz-dashboard.git)
cd data-viz-dashboard