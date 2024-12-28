# 🏥 Système de Médecine Prédictive

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)

</div>

## 🌟 Overview

A state-of-the-art medical prediction system leveraging advanced machine learning algorithms to predict potential medical conditions based on patient data. Built with precision and care for healthcare professionals.

## ✨ Key Features

- 🤖 **Advanced ML Models**: Utilizes Random Forest with optimized hyperparameters
- 📊 **Interactive Dashboard**: Real-time visualization of predictions and model metrics
- 🔍 **Comprehensive Analysis**: Feature importance and model performance insights
- 🛡️ **Robust Processing**: Advanced data preprocessing and feature engineering
- 📈 **Model Versioning**: Tracks model performance and metadata over time
- 🌐 **RESTful API**: Easy integration with existing healthcare systems

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip package manager

### Installation

1. Clone the repository:
```bash

https://github.com/eminemahjoub/systeme_medecine_predictive.git
cd systeme_medecine_predictive
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python -m flask --app src/app.py run
```

## 🏗️ Project Structure

```
systeme_medecine_predictive/
├── src/
│   ├── main.py              
│   ├── app.py             
│   └── utils/
│       ├── data_loader.py   
│       ├── data_processor.py 
│       └── model_evaluator.py 
├── models/                  
├── templates/            
├── tests/                
├── requirements.txt        
└── README.md             
```

## 💻 Usage

1. Start the web server:
```bash
python -m flask --app src/app.py run
```

2. Access the web interface at `http://localhost:5000`

3. Use the API endpoints:
- `POST /predict`: Make predictions
- `GET /model-info`: Get model information
- `POST /retrain`: Retrain the model

## 📊 Model Performance

- Accuracy: 96%
- Precision: 97%
- Recall: 96%
- F1 Score: 96%

## 🔧 Advanced Configuration

The system can be configured through environment variables:
- `MODEL_PATH`: Path to saved model
- `DATA_PATH`: Path to training data
- `DEBUG_MODE`: Enable debug logging

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Future Enhancements

- [ ] Integration with electronic health records
- [ ] Support for more medical conditions
- [ ] Advanced feature engineering
- [ ] Mobile application
- [ ] Multi-language support

## 👨‍💻 Author

**Amine Mahjoub**
- 📧 Email: [eminmahjoub@gmail.com]
- 💼 LinkedIn: [[amine mahjoub](https://www.linkedin.com/in/eminemahjoub/)]
- 🐱 GitHub: [eminemahjoub](https://github.com/eminemahjoub)

## 🙏 Acknowledgments

Special thanks to:
- The scikit-learn team for their excellent machine learning library
- The Flask team for the web framework
- The healthcare professionals who provided domain expertise

---

<div align="center">
Made with ❤️ by Amine Mahjoub
</div>
