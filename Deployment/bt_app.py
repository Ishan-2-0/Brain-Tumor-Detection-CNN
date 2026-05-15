import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import json
import os
import time

#setup
CLASSES=['glioma','meningioma','notumor','pituitary']
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
#model: ImprovedCNN
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN,self).__init__()
        self.conv1=nn.Conv2d(3, 32,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32, 64,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64, 128,kernel_size=3,padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.pool=nn.MaxPool2d(2, 2)
        self.fc1=nn.Linear(128*16*16,256)
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(256,4)
 
    def forward(self,x):
        x=self.pool(torch.relu(self.bn1(self.conv1(x))))
        x=self.pool(torch.relu(self.bn2(self.conv2(x))))
        x=self.pool(torch.relu(self.bn3(self.conv3(x))))
        x=x.view(x.size(0),-1)
        x=self.dropout(torch.relu(self.fc1(x)))
        x=self.fc2(x)
        return x
 
#page setup
st.set_page_config(page_title="Brain Tumor Detection",layout="wide")
st.title("Brain Tumor Detection CNN")
 
#description
st.markdown("""
### About this App
This model classifies brain MRI scans into 4 tumor types:
- **Glioma** : Most common primary brain tumor
- **Meningioma** : Tumor of brain membranes  
- **Pituitary** : Tumor of pituitary gland
- **No Tumor** : Normal scan
 
The CNN was trained on 5,600+ MRI images with 87% accuracy on test data.
 
### How to Use
1. Upload a brain MRI scan (JPG or PNG)
2. The model will analyze and classify the tumor
3. You'll see the predicted type and confidence score
 
**Disclaimer:** This is a demo model for research only. Always consult medical professionals for actual diagnosis.
""")
MAX_SIZE_MB=200
MAX_SIZE_BYTES=MAX_SIZE_MB*1024*1024
#file uploader
uploaded_file=st.file_uploader(f"Choose MRI scan (JPG/PNG,max {MAX_SIZE_MB}MB)",type=['jpg','jpeg','png'])
 
#loading model
@st.cache_resource
def load_model():
    try:
        script_dir=os.path.dirname(os.path.abspath(__file__))
        model_path=os.path.join(script_dir,"model_best.pth")
        
        if not os.path.exists(model_path):
            return None,f"Model file not found at {model_path}"
        model=ImprovedCNN()
        model.load_state_dict(torch.load(model_path,map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model,None
    except Exception as e:
        return None,str(e)
 
#validating image
def validate_image(file):
    try:
        #size check
        if file.size > MAX_SIZE_BYTES:
            return False, f"File too large (max {MAX_SIZE_MB}MB)", None
        #MIME type check
        if file.type not in ['image/jpeg','image/png']:
            return False, "Only JPG/PNG allowed", None
        #opening image
        img = Image.open(file)
        
        #convert to rpg for compactiblity
        if img.mode!='RGB':
            if img.mode=='RGBA':
                background=Image.new('RGB',img.size,(255,255,255))
                background.paste(img,mask=img.split()[3])
                img=background
            else:
                img=img.convert('RGB')
        
        #dimension check
        if img.width<32 or img.height<32 or img.width>4096 or img.height>4096:
            return False,"Image dimensions invalid (32-4096px)", None
        return True, None, img
    except Exception as e:
        return False, f"Corrupted image: {str(e)}", None
 
#preprocessing image
def preprocess(image):
    img=image.resize((128,128),Image.Resampling.LANCZOS)
    arr=np.array(img, dtype=np.float32)
    arr=arr/255.0
    arr=(arr-0.5)/0.5
    tensor=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return tensor.to(DEVICE)
 
#predict
def predict(model,tensor):
    try:
        with torch.no_grad():
            output=model(tensor)
        if torch.isnan(output).any() or torch.isinf(output).any():
            return None, None, "Model error: invalid output"
        probs=torch.softmax(output,dim=1)
        conf,pred_idx=torch.max(probs,dim=1)
        return CLASSES[pred_idx.item()],conf.item(),None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None, None, "Out of memory-try smaller image"
        return None, None, f"Model error:{str(e)}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

#log pred
def log_pred(pred_class,conf,latency,error=None):
    try:
        entry={
            'timestamp':time.time(),
            'prediction':pred_class,
            'confidence':float(conf) if conf else None,
            'latency_ms':float(latency),
            'error': error
        }
        with open('predictions.jsonl','a') as f:
            f.write(json.dumps(entry)+'\n')
    except:
        pass

#main flow
if uploaded_file:
    #validate
    is_valid,err_msg,image=validate_image(uploaded_file)
    if not is_valid:
        st.error(err_msg)
        st.stop()
    st.success("Image validated")
    #load model
    model,model_err=load_model()
    if model_err:
        st.error(f"Model error:{model_err}")
        st.stop()
    #predict
    with st.spinner("analyzing..."):
        start=time.time()
        tensor=preprocess(image)
        pred_class,confidence,pred_err=predict(model,tensor)
        latency=(time.time()-start)*1000
    if pred_err:
        st.error(pred_err)
        log_pred(None, None,latency,error=pred_err)
        st.stop()

    #results
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(image)
    with col2:
        st.subheader("Result")
        st.metric("Tumor Type", pred_class.upper())
        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Latency", f"{latency:.0f}ms")
        if confidence < 0.7:
            st.warning(f"Low confidence:{confidence:.1%}")
    st.info("Disclaimer: 87% accuracy. Always consult medical professionals")
    log_pred(pred_class, confidence, latency)
#sidebar
with st.sidebar:
    st.divider()
    st.subheader("About")
    st.write("""
    **Model:** ImprovedCNN with Batch Norm & Dropout
    **Classes:** 4 tumor types
    **Accuracy:** 87% on test data
    **Framework:** PyTorch + Streamlit
    """)
    st.divider()
    st.write("**GitHub:** [Brain-Tumor-Detection-CNN](https://github.com/Ishan-2-0/Brain-Tumor-Detection-CNN)")