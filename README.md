# 🏛️ ICEB Constitution Chatbot

An intelligent AI-powered chatbot for exploring and understanding the IQRA Constitution document. Built with Streamlit, LangChain, and OpenAI GPT.

![ICEB Logo](ICEB.png)

## ✨ Features

- 🎯 **Custom AI Instructions** - Customize how the AI responds to your questions
- 📚 **Context-aware Answers** - Intelligent responses based on the constitution content
- 🔍 **Source Reference Tracking** - See exactly which sections were used for answers
- 💡 **Sample Questions** - Quick-start with pre-written questions
- 🎨 **Professional Interface** - Clean, modern design with ICEB branding
- 📱 **Responsive Design** - Works on desktop and mobile devices

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ICEBLLM
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp env_example.txt .env
# Edit .env and add your OpenAI API key
```

4. **Run the application**
```bash
streamlit run iqra_constitution_qa.py
```

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Push to GitHub**
   - Create a new repository on GitHub
   - Push your code to the repository

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Add your OpenAI API key in the secrets section

3. **Configure Secrets**
   - In Streamlit Cloud, go to your app settings
   - Add secrets:
     ```toml
     [general]
     OPENAI_API_KEY = "your-actual-api-key"
     ```

### Option 2: Heroku

1. **Create Heroku app**
```bash
heroku create your-app-name
```

2. **Set environment variables**
```bash
heroku config:set OPENAI_API_KEY=your-api-key
```

3. **Deploy**
```bash
git push heroku main
```

### Option 3: Railway

1. **Connect GitHub repository**
2. **Set environment variables**
3. **Deploy automatically**

### Option 4: Google Cloud Platform

1. **Use Cloud Run**
2. **Deploy with Docker**
3. **Set environment variables**

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_ORGANIZATION` - Your OpenAI organization ID (optional)
- `TOKENIZERS_PARALLELISM` - Set to "false" to suppress warnings

### Custom Prompts

The application supports custom AI instructions:

1. **Strict IQRA Only** - Only answers IQRA Constitution questions
2. **Detailed Explanations** - Comprehensive answers with references
3. **Simple & Clear** - Easy-to-understand responses

## 📋 Requirements

- Python 3.9+
- OpenAI API key
- Internet connection for AI processing

## 🛠️ Dependencies

- `streamlit` - Web application framework
- `langchain` - LLM application framework
- `openai` - OpenAI API client
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embeddings
- `PyPDF2` - PDF text extraction

## 📄 File Structure

```
ICEBLLM/
├── iqra_constitution_qa.py    # Main application
├── requirements.txt           # Python dependencies
├── ICEB.png                  # Logo file
├── IQRAConstitution (1).pdf  # Constitution document
├── .env                      # Environment variables (local)
├── secrets.toml              # Streamlit secrets template
└── README.md                 # This file
```

## 🔒 Security

- Environment variables are used for API keys
- Sensitive files are excluded from version control
- API keys are never exposed in the frontend

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Contact the development team

## 📜 License

This project is licensed under the MIT License.

---

**Built with ❤️ for the ICEB Community** 