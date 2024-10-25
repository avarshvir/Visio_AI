# Visio AI

![Visio AI Logo](../Visio_AI/Visio_AI/images/homepage_img.png)

**Visio AI** is an all-in-one machine learning and data analysis application designed to provide seamless dataset uploading, data visualization, and model prediction capabilities. It is built using Python, Streamlit, and scikit-learn, making it highly scalable and user-friendly for developers and analysts alike.

## Features

- **User Authentication**: Sign-up and login functionality.
- **Data Upload**: Supports uploading CSV, Excel, and text files.
- **Data Visualization**: Interactive charts (scatter, bar, line, etc.) using Matplotlib and Seaborn.
- **Missing Data Handling**: Automated imputation with options for mean, median, mode, and more.
- **Machine Learning Models**: Offers regression, classification, and clustering algorithms (e.g., decision trees, KNN, k-means).
- **Additional Tools**: Integrated tools such as a notepad, calculator, to-do list, and calendar.
  
## Technology Stack

- **Frontend**: Streamlit (for interactive UI)
- **Backend**: Python, Flask (for API integration)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Data Visualization**: Matplotlib, Seaborn
- **Database**: MySQL (XAMPP for local hosting)
  
## Getting Started

To get a copy of this project up and running on your local machine, follow the instructions below.

### Prerequisites

- Python 3.8+
- MySQL (XAMPP recommended)
- Streamlit
- scikit-learn
- pandas
- matplotlib
-  seaborn

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/avarshvir/Visio_AI.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Visio_AI
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

5. Set up the MySQL database by running the provided SQL schema (see `database/schema.sql`).

## Usage

1. Sign up or log in to access the application.
2. Upload your dataset (CSV, Excel, or text file).
3. Handle missing values, visualize the data, and select machine learning algorithms for predictions.
4. Access additional tools like the Notepad, Calculator, and more.

## Future Scope

- Integration of advanced AI models (deep learning, NLP).
- Real-time data analysis.
- Enhanced collaboration tools (file sharing, version control).
- Mobile application development.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Power BI](https://powerbi.microsoft.com)
- [Tableau](https://www.tableau.com)
- [IBM Watson AI](https://www.ibm.com/watson)
- [Streamlit](https://streamlit.io)
- [scikit-learn](https://scikit-learn.org)
