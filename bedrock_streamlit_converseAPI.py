import os
import json
import base64
import logging
import boto3
import streamlit as st
import re

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError
from config_manager import ConfigManager, ModelInfo


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Initialize configuration manager
config_manager = ConfigManager()

@dataclass
class ModelConfig:
    model_id: str
    system_message: str
    messages: list
    max_tokens: int
    budget_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    enable_reasoning: bool = False

def initialize_session_state():
    # Initialize chat history for display 
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    # Initialize chat history for model interaction
    if "model_messages" not in st.session_state:
        st.session_state.model_messages = []

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    # Initialize image track recorder
    if "file_update" not in st.session_state:
        st.session_state.file_update = False

    if "allow_input" not in st.session_state:
        st.session_state.allow_input = True

    if "enable_reasoning" not in st.session_state:
        st.session_state.enable_reasoning = False

    if "current_model_id" not in st.session_state:
        st.session_state.current_model_id = config_manager.get_default_model()

def reset_app():
    st.session_state.display_messages = []
    st.session_state.model_messages = []
    st.session_state["file_uploader_key"] += 1
    st.session_state.file_update = False
    st.session_state.allow_input = True
    st.rerun()

def check_file_size(file, file_type):
    """
    Check if file size is within limits
    Returns (is_valid, message)
    """
    file_size = len(file.getvalue())
    file_types = config_manager.get_file_types_by_category()
    
    for category, types in file_types.items():
        if file_type in types:
            file_config = config_manager.get_file_config(category)
            if file_size > file_config.size_limit:
                size_mb = file_config.size_limit / (1024 * 1024)
                return False, f"{category.title()} file '{file.name}' exceeds {size_mb:.1f}MB limit"
    
    return True, ""

def file_update():
    st.session_state.file_update = True

def allow_input_disable():
    st.session_state.allow_input = False

def stream_multi_modal_prompt(bedrock_runtime, config: ModelConfig, model_info: ModelInfo):
    inference_config = {"maxTokens": config.max_tokens,}
    additional_model_fields = {}

    if config.enable_reasoning:
        if model_info.model_family == "claude":
            additional_model_fields = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": config.budget_tokens
                }
            }
        elif model_info.model_family == "qwen":
            additional_model_fields = {"reasoning_config": "high"}
    else:
        if config.temperature is not None:
            inference_config["temperature"] = config.temperature
        if config.top_p is not None and model_info.supports_top_p:
            inference_config["topP"] = config.top_p
        if model_info.model_family == "nova":
            additional_model_fields = {"inferenceConfig": {"top_k": config.top_k}} 
        elif model_info.supports_top_k:
            additional_model_fields = {"top_k": config.top_k}

    try:
        response = bedrock_runtime.converse_stream(
            modelId=config.model_id,
            messages=config.messages,
            system=[{"text": config.system_message}],
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )

        in_reasoning_block = False
        actual_response_text = ""  # actual response text (excluding reasoning part)
        reasoning_response_text= ""  # reasoning response text
        reasoning_redacted_content = b''
        signature_response_text = ""
        assistant_content = []


        for chunk in response["stream"]:
            if "contentBlockDelta" not in chunk:
                continue

            delta = chunk["contentBlockDelta"]["delta"]

            if "reasoningContent" in delta:
                if "text" in delta["reasoningContent"]:
                    reasoning_text = delta["reasoningContent"]["text"]
                    reasoning_response_text += reasoning_text  # Collect reasoning response text
                    if not in_reasoning_block:
                        yield "----------------\n"
                        in_reasoning_block = True
                    yield reasoning_text
                if "redactedContent" in delta["reasoningContent"]:
                    redacted_content = delta["reasoningContent"]["redactedContent"]
                    reasoning_redacted_content += redacted_content
                if "signature" in delta["reasoningContent"]:
                    signature = delta["reasoningContent"]["signature"]
                    signature_response_text += signature

            elif "text" in delta:
                text = delta["text"]
                actual_response_text += text  # Collect actual response text without reasoning part

                if in_reasoning_block:
                    yield "\n\n----------------\n"
                    in_reasoning_block = False
                yield text
        
        if reasoning_response_text and signature_response_text:
            assistant_content.append({"reasoningContent": {"reasoningText": {"text": reasoning_response_text, "signature": signature_response_text}}})
        if reasoning_redacted_content:
            assistant_content.append({"reasoningContent": {"redactedContent": reasoning_redacted_content}})
        if actual_response_text:
            assistant_content.append({"text": actual_response_text})

        st.session_state.model_messages.append({"role": "assistant", "content": assistant_content})

    except (ClientError, Exception) as e:
        logger.error(f"ERROR: Can't invoke '{config.model_id}'. Reason: {e}")
        st.error(f"ERROR: Can't invoke '{config.model_id}'. Reason: {e}")
        raise

def get_bedrock_runtime_client(aws_access_key=None, aws_secret_key=None, aws_region=None):
    try:
        if aws_access_key and aws_secret_key and aws_region:
            bedrock_runtime = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
        else:
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
    except ClientError as e:
        # Handle errors returned by the AWS service
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS service returned an error: {error_code} - {error_message}")
        st.error(f"AWS service returned an error: {error_code} - {error_message}")
        raise
    except NoCredentialsError:
        # Handle the case where credentials are missing
        logger.error("Unable to retrieve AWS credentials, please check your credentials configuration.")
        st.error("Unable to retrieve AWS credentials, please check your credentials configuration.")
        raise
    except Exception as e:
        # Handle any other unknown exceptions
        logger.error(f"An unknown error occurred: {str(e)}")
        st.error(f"An unknown error occurred: {str(e)}")
        raise
    return bedrock_runtime

def main():
    initialize_session_state()

    # App title
    st.set_page_config(page_title="Bedrock-Streamlit-Chat 💬", page_icon='./utils/logo.png')

    with st.sidebar:
        col1, col2 = st.columns([1,3.5])
        with col1:
            st.image('./utils/logo.png')
        with col2:
            st.title("Bedrock-Streamlit-Chat")

        model_names = config_manager.get_all_model_names()
        default_model = config_manager.get_default_model()
        default_index = model_names.index(default_model) if default_model in model_names else 0
        
        new_model_id = st.selectbox(
            'Choose a Model',
            model_names,
            index=default_index,
            label_visibility="collapsed"
        )

        if new_model_id != st.session_state.current_model_id:
            st.session_state.current_model_id = new_model_id
            reset_app()

        model_info = config_manager.get_model_info(new_model_id)
        model_id = model_info.model_id

        if model_info.supports_reasoning:
            st.session_state.enable_reasoning = st.toggle("Reasoning Mode", value=st.session_state.enable_reasoning, 
                                                          help="Enable model's reasoning capability")
        else:
            st.session_state.enable_reasoning = False

        regions = config_manager.get_regions()
        default_region = config_manager.get_default_region()
        default_region_index = regions.index(default_region) if default_region in regions else 0
        aws_region = st.selectbox('Choose a Region', regions, index=default_region_index, label_visibility="collapsed")
        if aws_region:
            os.environ['AWS_REGION'] = aws_region
        else:
            st.error("Please select a valid AWS region")
            return

        with st.expander('AWS Credentials', expanded=False):
            aws_access_key = st.text_input('AWS Access Key', os.environ.get('AWS_ACCESS_KEY_ID', ""), type="password")
            aws_secret_key = st.text_input('AWS Secret Key', os.environ.get('AWS_SECRET_ACCESS_KEY', ""), type="password")

            credentials_changed = (
                aws_access_key != os.environ.get('AWS_ACCESS_KEY_ID', "") or
                aws_secret_key != os.environ.get('AWS_SECRET_ACCESS_KEY', "")
            )

            if st.button('Update AWS Credentials', disabled=not credentials_changed):
                if aws_access_key == "" or aws_secret_key == "":
                    st.warning("Please fill out all the AWS credential fields.")
                else:
                    st.success("AWS credentials are updated successfully!")
                    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
                    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key

        with st.expander('System Prompt', expanded=False):
            system_prompt = st.text_area(
                "System prompt", 
                "You are a helpful, harmless, and honest AI assistant. "
                "Your goal is to provide informative and substantive responses to queries while avoiding potential harms.", 
                label_visibility="collapsed"
            )

        with st.expander('Model Parameters', expanded=False):
            params_disabled = st.session_state.enable_reasoning

            max_new_tokens = st.number_input(
                min_value=100,
                max_value=model_info.max_tokens,
                step=10,
                value=min(16384, model_info.max_tokens),
                label="Number of tokens to output",
                key="max_new_token"
            )

            if model_info.supports_reasoning and model_info.model_family == "claude":
                budget_tokens = st.number_input(
                    min_value=1024,
                    max_value=65536,
                    step=10,
                    value=4096,
                    label="Number of tokens to think" + (" (disabled in non-reasoning mode)" if not params_disabled else ""),
                    key="budget_tokens",
                    disabled=not params_disabled
                )
            else:
                budget_tokens = None

            col1, col2 = st.columns([4,1])
            with col1:
                temperature = st.slider(
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    value=0.5,
                    label="Temperature" + (" (disabled in reasoning mode)" if params_disabled else ""),
                    key="temperature",
                    disabled=params_disabled
                )
                top_p_disabled = params_disabled or not model_info.supports_top_p
                top_p_label = "Top P"
                if params_disabled:
                    top_p_label += " (disabled in reasoning mode)"
                elif not model_info.supports_top_p:
                    top_p_label += " (not supported by this model)"
                top_p = st.slider(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    value=1.0,
                    label=top_p_label,
                    key="top_p",
                    disabled=top_p_disabled
                )
                if model_info.supports_top_k:
                    top_k = st.slider(
                        min_value=0,
                        max_value=model_info.top_k_max,
                        step=1,
                        value=model_info.top_k_max // 2,
                        label="Top K" + (" (disabled in reasoning mode)" if params_disabled else ""),
                        key="top_k",
                        disabled=params_disabled
                    )
                else:
                    top_k = None

        if model_info.supports_multimodal:
            file = st.file_uploader("File Query", accept_multiple_files=True, key=st.session_state["file_uploader_key"], on_change=file_update, help='Support multimodal models', disabled=False)
            file_list = []
            file_types = config_manager.get_file_types_by_category()

            for item in file:
                item_type = item.name.split('.')[-1].lower()

                # Check file size
                is_valid_size, error_message = check_file_size(item, item_type)
                if not is_valid_size:
                    st.error(error_message)
                    return None

                if item_type in file_types["image"]:
                    item_type = 'jpeg' if item_type == 'jpg' else item_type
                    st.image(item, caption=item.name)
                    file_list.append({"image": {"format": item_type, "source": {"bytes": item.getvalue()}}})
                elif item_type in file_types["document"]:
                    file_list.append({"document": {"format": item_type, "name": item.name.split(".")[0], "source": {"bytes": item.getvalue()},  "citations": {"enabled": True}}})
                elif item_type in file_types["video"]:
                    if model_info.supports_video:
                        st.video(item)
                        file_list.append({"video": {"format": item_type, "source": {"bytes": item.getvalue()}}})
                    else:
                        st.error(f"Video files are only supported by models that support video. Please remove {item.name}")
                        return None
                else:
                    st.write(f"Unsupported file type: {item_type}, please remove the file!")
                    return None
        else:
            file = st.file_uploader("File Query", help='Multimodal models only', disabled=True)

        # Clear messages, including uploaded images
        if st.sidebar.button("New Conversation", type="primary"):
            reset_app()

    with st.chat_message("assistant", avatar="./utils/assistant.png"):
        st.write("I am an AI chatbot powered by Amazon Bedrock, what can I do for you？💬")

    # Display chat messages from history on app rerun
    for message in st.session_state.display_messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="./utils/assistant.png"):
                st.markdown(message["content"][0]["text"])
        else:
            with st.chat_message(message["role"], avatar="./utils/user.png"):
                for item in message["content"]:
                    if "image" in item:
                        st.image(item["image"]["source"]["bytes"], width=50)
                    elif "document" in item:
                        col1, col2 = st.columns([0.45,8])
                        with col1:
                            st.image('./utils/file.png')
                        with col2:
                            document_full = item["document"]["name"]+"."+item["document"]["format"]
                            st.markdown(document_full)
                    elif "video" in item:
                        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1,1,1,1,1,1,1,1])
                        with col1:
                            st.video(item["video"]["source"]["bytes"])
                    else:
                        st.markdown(item["text"])

    if query := st.chat_input("Input your message...", disabled=not st.session_state.allow_input, on_submit=allow_input_disable):
        # Display user message in chat message container
        with st.chat_message("user", avatar="./utils/user.png"):
            user_content = []
            if st.session_state.file_update:
                file_types = config_manager.get_file_types_by_category()
                for item in file:
                    item_type = item.name.split('.')[-1].lower()
                    if item_type in file_types["image"]:
                        st.image(item, width=50)
                    elif item_type in file_types["video"]:
                        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1,1,1,1,1,1,1,1])
                        with col1:
                            st.video(item)
                    else:
                        col1, col2 = st.columns([0.45,8])
                        with col1:
                            st.image('./utils/file.png')
                        with col2:
                            st.markdown(item.name)
                user_content = file_list
            st.session_state.file_update = False
            st.markdown(query)
            
        # Add user message to chat history
        user_content.append({"text": query})
        user_message = {"role": "user", "content": user_content}
        st.session_state.display_messages.append(user_message)
        st.session_state.model_messages.append(user_message)

        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="./utils/assistant.png"):
            system_message = system_prompt

            model_messages = st.session_state.model_messages

            model_config = ModelConfig(
                model_id=model_id,
                system_message=system_message,
                messages=model_messages,  # Use model_messages without reasoning part
                max_tokens=max_new_tokens,
                budget_tokens=budget_tokens,
                temperature=temperature if not st.session_state.enable_reasoning else None,
                top_p=top_p if not st.session_state.enable_reasoning and model_info.supports_top_p else None,
                top_k=top_k if not st.session_state.enable_reasoning and model_info.supports_top_k else None,
                enable_reasoning=st.session_state.enable_reasoning
            )

            bedrock_runtime = get_bedrock_runtime_client(
                aws_access_key=os.environ.get('AWS_ACCESS_KEY_ID', ""), 
                aws_secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY', ""), 
                aws_region=os.environ.get('AWS_REGION', ""))

            with st.spinner('Thinking...'):
                try:

                    response = st.write_stream(
                        stream_multi_modal_prompt(
                            bedrock_runtime, model_config, model_info
                        )
                    )

                    if not response:
                         st.error("No response received from the model")
                         st.stop()

                    assistant_content = [{"text": response}]
                    st.session_state.display_messages.append({"role": "assistant", "content": assistant_content})

                except ClientError as err:
                    message = err.response["Error"]["Message"]
                    logger.error("A client error occurred: %s", message)
                    st.error(f"A client error occurred: {message}")
                    st.stop()

                except Exception as e:
                    logger.error(f"An unknown error occurred: {str(e)}")
                    st.error(f"An unknown error occurred: {str(e)}")
                    st.stop()

                finally:
                    st.session_state.allow_input = True
                    st.rerun()

if __name__ == "__main__":
    main()
