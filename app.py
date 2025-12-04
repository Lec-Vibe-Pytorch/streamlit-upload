####
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# 1. 모델 정의 (Notebook에서 가져옴)
class ASLLinearNet(nn.Module):
    """nn.Linear 기반 ASL 분류 모델 (28x28 Grayscale)"""
    
    def __init__(self, input_size=784, num_classes=24):
        super(ASLLinearNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x

# 2. 설정 및 클래스 이름
MODEL_PATH = './model/best_nnLinear_model.pth'
# A-I (0-8), K-Y (9-23) - J와 Z 제외
CLASS_NAMES = [chr(65 + i) if i < 9 else chr(65 + i + 1) for i in range(24)]

# 3. 모델 로드 함수
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASLLinearNet(num_classes=24)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model, device
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {e}")
            return None, device
    else:
        st.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return None, device

# 4. 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),  # Grayscale 변환
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 1채널 정규화
    ])
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# 5. 메인 UI
def main():
    st.set_page_config(page_title="ASL 수어 분류기", page_icon="✋")
    
    st.title("🤟 AI ASL Classifier")
    st.write("이미지를 업로드하거나 샘플 이미지를 선택하면 어떤 알파벳 수어인지 알려줍니다!")
    
    # 사이드바
    st.sidebar.header("📌 정보")
    st.sidebar.info("이 앱은 PyTorch로 학습된 다층 신경망 모델(nn.Linear)을 사용합니다.")
    st.sidebar.write("**모델 구조:**")
    st.sidebar.write("- 입력: 28×28 Grayscale")
    st.sidebar.write("- 레이어: 784→512→256→128→24")
    st.sidebar.write("- BatchNorm + Dropout 적용")
    
    # 모델 로드
    model, device = load_model()
    
    if model is None:
        return

    # 이미지 업로드
    st.subheader("📤 업로드")
    
    uploaded_file = st.file_uploader("수어 이미지를 선택하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    image = None
    image_source = ""
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_source = uploaded_file.name
    
    # 이미지가 선택되었을 때 분석 수행
    if image is not None:
        try:
            # 1. 선택된 이미지 표시
            st.subheader("📷 선택된 이미지")
            st.image(image, caption=f'{image_source}', use_container_width=True)
            
            st.write("---")  # 구분선
            
            # 2. 분석 및 결과 표시
            st.subheader("📊 분석 결과")
            
            with st.spinner('분석 중...'):
                # 전처리 및 예측
                input_tensor = preprocess_image(image).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    predicted_idx = predicted.item()
                    predicted_class = CLASS_NAMES[predicted_idx]
                    confidence_score = confidence.item() * 100
                
                # 결과 표시
                st.success(f"### 예측: **{predicted_class}**")
                st.metric(label="신뢰도", value=f"{confidence_score:.2f}%")
                
                # Top 3 확률 표시
                st.write("---")
                st.write("**상위 3개 예측:**")
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                for i in range(3):
                    cls = CLASS_NAMES[top3_idx[0][i].item()]
                    prob = top3_prob[0][i].item() * 100
                    st.write(f"{i+1}. **{cls}**: {prob:.2f}%")
                    st.progress(int(prob))
                                
        except Exception as e:
            st.error(f"이미지 처리 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
