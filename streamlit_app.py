import sys
import os
import streamlit as st
from urllib.parse import urlparse
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from PIL import Image
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# cd /Users/macmini/Documents/Python/Image-Captioning; streamlit run streamlit_app.py
# pipreqs /Users/macmini/Documents/Python/Image-Captioning

def is_url_image(image_url):
   image_formats = ("image/png", "image/jpeg", "image/jpg")
   r = requests.head(image_url)
   if r.headers["content-type"] in image_formats:
      return True
   return False

def is_valid_url(url):
    parsed_url = urlparse(url)
    return bool(parsed_url.scheme and parsed_url.netloc)

image_url = st.text_input('Image URL', '')

try:
    if st.button('Submit Image URL'):
        st.write('\n\n')
        image_url = image_url.strip()
        # st.text(f'image_url: {image_url}')

        if image_url == "":
            st.error("There is no image URL. Please try again.")
        else:
            if not is_valid_url(image_url):
                st.error("Your given Image URL is invalid. Please try again with a valid image URL.")
            elif not is_url_image(image_url):
                st.error("Your given Image URL is invalid. Please try again with a valid image URL.")
            else:
                hf_model = "Salesforce/blip-image-captioning-large"
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                processor = BlipProcessor.from_pretrained(hf_model)
                model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)

                # image_url = 'https://images.unsplash.com/photo-1570042707390-2e011141ab78?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1332&q=80'
                image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
                # image
                st.image(image)

                # unconditional image captioning
                inputs = processor(image, return_tensors="pt").to(device)

                out = model.generate(**inputs, max_new_tokens=20)
                image_caption = processor.decode(out[0], skip_special_tokens=True)

                st.title('Image Caption')
                # st.header('Image Caption')
                st.success(image_caption)
except Exception as e:
    error_message = ''
    # st.text('Hello World')
    st.error('An error has occurred. Please try again.', icon="🚨")
    # Just print(e) is cleaner and more likely what you want,
    # but if you insist on printing message specifically whenever possible...
    if hasattr(e, 'message'):
        error_message = e.message
    else:
        error_message = e
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    st.error('ERROR MESSAGE: {}'.format(error_message))
