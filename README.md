# Marketing Analytics Dashboard

A comprehensive, interactive dashboard for analyzing marketing campaign performance, tracking KPIs, and identifying optimization opportunities. Built with Streamlit and Plotly.

## 🚀 Features

The dashboard provides deep insights across multiple dimensions:

-   **Executive Overview**: High-level KPIs (Spend, Revenue, Profit, ROAS) and financial trend analysis.
-   **Traffic Analysis**: Breakdown of traffic sources, conversion rates, and cost efficiency.
-   **Campaign Performance**: Detailed metrics for individual campaigns, including a "Quadrant Analysis" to identify winners and losers.
-   **Geographic Insights**: Interactive map showing conversion rates by state.
-   **Funnel Analysis**: Visualization of the user journey from landing to policy sale, identifying drop-off points.
-   **Segments & Targeting**: Demographic breakdown (Device, OS, Browser, etc.) to refine targeting strategies.
-   **Time Series**: Custom trend analysis for any metric over time.
-   **Fraud Detection ("Sus Traffic")**: Identification of suspicious activity and potential click fraud.

## 🛠️ Technology Stack

-   **Python 3.10+**
-   **Streamlit**: For the interactive web application.
-   **Pandas**: For data manipulation and analysis.
-   **Plotly**: For interactive and dynamic visualizations.
-   **Scikit-learn**: For advanced analytics (Linear Regression, Clustering).

## 📦 Installation & Local Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/e-landeros/marketing_dash.git
    cd marketing-analytics-dashboard
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    The dashboard will open in your default browser at `http://localhost:8501`.

## 🐳 Deployment Options

### Docker

You can containerize the application for consistent deployment across environments.

1.  **Build the Docker image:**
    ```bash
    docker build -t marketing-dash .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 8501:8501 marketing-dash
    ```

### Hugging Face Spaces

This project is configured for easy deployment to Hugging Face Spaces.

1.  **Prerequisites:**
    -   A Hugging Face account.
    -   A created Space (SDK: Docker).
    -   Git LFS installed (`git lfs install`).

2.  **Configuration:**
    -   Open `deploy.sh` and update the `SPACE_REPO` variable with your Space's URL:
        ```bash
        SPACE_REPO="https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME"
        ```

3.  **Deploy:**
    Run the deployment script to automatically build, package, and push your application:
    ```bash
    ./deploy.sh
    ```
    This script handles:
    -   Cloning the remote Space.
    -   Copying the application code, `Dockerfile`, and data.
    -   Configuring Git LFS for large files.
    -   Pushing the changes to trigger a new build.

## 📂 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── data/
│   └── raw/                # Data directory
├── Dockerfile              # Docker configuration
├── deploy.sh               # Deployment automation script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
