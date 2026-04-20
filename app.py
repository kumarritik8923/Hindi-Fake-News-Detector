import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import re
from gensim.models import Word2Vec

hindi_stopwords = set(["है", "और", "कि", "का", "की", "के", "में", "से", "को", "पर", "यह", "वह", "जो", "तो", "भी", "ने", "एक", "हो", "कर", "साथ"])

def custom_hindi_tokenizer(text):
    text = str(text)
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~।]', ' ', text)
    return [word for word in text.split() if word not in hindi_stopwords]

class FakeNewsLSTM(nn.Module):
    def __init__(self, input_size=50, hidden_size=64):
        super(FakeNewsLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

@st.cache_resource
def load_models():
    w2v = Word2Vec.load("models/hindi_word2vec.model")
    lstm = FakeNewsLSTM()
    lstm.load_state_dict(torch.load('models/lstm_fake_news.pth', map_location=torch.device('cpu')))
    lstm.eval()
    return w2v, lstm

w2v_model, lstm_model = load_models()

def text_to_sequence(tokens, w2v, max_len=20):
    seq = [w2v.wv[word] for word in tokens if word in w2v.wv]
    if len(seq) < max_len: seq.extend([np.zeros(50)] * (max_len - len(seq)))
    else: seq = seq[:max_len]
    return np.array(seq)

# --- WEB DESIGN ---

st.title("🕵️‍♂️ Hindi Fake News Detector")

st.write("Welcome to our core NLP project. Paste a suspicious Hindi WhatsApp forward below to check its authenticity based on our custom LSTM model.")

user_input = st.text_area("✍️ Enter Hindi News Snippet Here:", height=150)

if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Processing the input
        tokens = custom_hindi_tokenizer(user_input)
        sequence = text_to_sequence(tokens, w2v_model)
        tensor_input = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = lstm_model(tensor_input).item()
        
        st.markdown("---")
        
        if prediction >= 0.5:
            st.error(f"🚨 **FAKE NEWS DETECTED** (The model is {prediction*100:.1f}% confident this is fake.)")
        else:
            st.success(f"✅ **LIKELY REAL NEWS** (The model is {(1-prediction)*100:.1f}% confident this is real.)")
        
        st.info(f"**Words identified after processing:** {', '.join(tokens)}")