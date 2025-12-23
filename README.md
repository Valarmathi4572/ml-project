# 🩺 Diabetes Prediction Web App

A machine learning-powered web application for diabetes prediction using Flask and scikit-learn.

## 📋 Features

- **Interactive Web Interface**: Modern, responsive design with real-time form validation
- **Machine Learning Model**: Random Forest classifier trained with SMOTE for balanced predictions
- **Real-time Predictions**: Get instant diabetes risk assessment
- **Educational Content**: Information about diabetes types and risk factors
- **Progress Tracking**: Multi-step form with visual progress indicators

## 🚀 Live Demo

🌐 **View the app**: [https://your-app-url.com](https://your-app-url.com)

## 🛠️ Tech Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **ML**: scikit-learn, Random Forest, SMOTE
- **Data**: Pandas, NumPy
- **Deployment**: Ready for Heroku/AWS/GCP

## 📊 Dataset

The model is trained on the Pima Indians Diabetes Dataset with the following features:
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Valarmathi4572/ml-project.git
   cd ml-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

## 📈 Model Performance

- **Accuracy**: 75%
- **Precision**: 73%
- **Recall**: 75%
- **F1-Score**: 73%

## 🎯 Usage

1. Fill in your health metrics in the web form
2. Click "Get Prediction"
3. Receive instant diabetes risk assessment
4. View personalized recommendations

## 📁 Project Structure

```
ml-project/
├── app.py                      # Flask application
├── diabetes.csv               # Dataset
├── diabetes_rf_smote_model.pkl # Trained ML model
├── scaler.pkl                 # Data scaler
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html            # Main web interface
├── static/
│   └── style.css             # CSS styling
└── README.md                 # Project documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Pima Indians Diabetes Dataset
- scikit-learn documentation
- Flask framework
- Open source community

## 📞 Contact

**Valarmathi4572**
- GitHub: [@Valarmathi4572](https://github.com/Valarmathi4572)
- Project Link: [https://github.com/Valarmathi4572/ml-project](https://github.com/Valarmathi4572/ml-project)

---

⭐ **Star this repo if you found it helpful!**