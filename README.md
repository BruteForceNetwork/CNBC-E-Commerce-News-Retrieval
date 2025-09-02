# CNBC & E-Commerce AI News Chatbot Co-Pilot
An advanced AI chatbot co-pilot with a **Flask web interface** for interactive conversation and real-time news retrieval.

This chatbot co-pilot leverages the **OpenAI API** for natural language understanding, complemented by **RAG (Retrieval-Augmented Generation)** to fetch up-to-date news and insights from **CNBC News** and top **e-commerce sources**.

---

## Tools Used

1. **LangChain Framework** – for user query synthesis and AI response generation  
2. **Flask** – for building the web interface  
3. **GPT-3.5** – for natural language understanding  
4. **CNBC & E-Commerce News Sources** – for current news and market updates  

![CNBC & E-Commerce](<CNBC & E-Commerce.JPG>)

### **Steps to run:**

1. Download this github repository
2. Create a virtual environment
```bash
python -m venv venv
```

3. Activate the virtual environment
```bash
`source venv/bin/activate` or `venv/Scripts/activate` [for Windows]
```

4. Install the requirements
```bash
`pip install -r requirements.txt`
```
5. Add your OpenAI API key in a .env file

6. On the terminal run the command below 
```bash
`python app.py`
```
7. App should not be running on localhost default port
