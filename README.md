# Streamlit App for AWS Bedrock foundation models  

### Prerequisite:
You need to get a valid authority for accessing Amazon Bedrock foundation models

### How to use:  
1. git clone the repo  

2. cd to the folder of repo  

3. pip install -r requirements.txt  

4. streamlit run bedrock_streamlit_converseAPI.py  

### Adding New Models:
To add a new model, simply edit the `models_config.json` file:

1. **Add model configuration** in the "models" section:
```json
"Your Model Name": {
  "model_id": "your.model.id",
  "max_tokens": 4096,
  "supports_multimodal": true,
  "supports_video": false,
  "supports_reasoning": false,
  "top_k_max": 500,
  "model_family": "your_family",
  "supports_top_k": true
}
```

2. **Model Parameters Explanation**:
   - `model_id`: AWS Bedrock model identifier
   - `max_tokens`: Maximum output tokens supported
   - `supports_multimodal`: Enable file upload (images/documents)
   - `supports_video`: Enable video file support
   - `supports_reasoning`: Enable reasoning mode toggle
   - `top_k_max`: Maximum top-k value for sampling
   - `model_family`: Model family name (nova, claude, deepseek, etc.)
   - `supports_top_k`: Whether model supports top-k parameter

3. **Automatic Integration**: After saving the config file, the new model will automatically appear in the model selection dropdown. No code changes required!

### Configuration Files:
- `models_config.json`: Model configurations and settings
- `config_manager.py`: Configuration management utilities

### Note:  
* Use Bedrock cross-region(US based regions) inference profile, you need to enable accessibility of models in US regions first.  

* Support image, document, video understanding based on model capabilities.  

* The default credentials for App will be [environment variables](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#environment-variables) or [shared credentials file](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#shared-credentials-file), which are same with credentials for boto3 sdk. Besides, you can also provide your own AKSK with region info via App.  

* "archived/bedrock_streamlit.py" was writen by InvokeModel API, "bedrock_streamlit_converseAPI.py" was writen by Converse API, and "archived/bedrock_streamlit_converseAPI_secret.py" was added with login function

### App screenshot:
Chat with image and chat with document
![screenshot](./utils/app-screenshot.png)
