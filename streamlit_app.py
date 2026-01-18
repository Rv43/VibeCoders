import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import tempfile
import whisper
from pydub import AudioSegment
import io

# Page config
st.set_page_config(
    page_title="Adverse Medical Event Detector",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .risk-critical {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin: 20px 0;
    }
    .risk-high {
        background-color: #ffaa00;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin: 20px 0;
    }
    .risk-moderate {
        background-color: #00aa00;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin: 20px 0;
    }
    .risk-none {
        background-color: #00cc00;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin: 20px 0;
    }
    .event-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    """Load all ML models and vectorizers"""
    SAVE_DIR = "saved_models"
    
    try:
        models = {
            'binary': joblib.load(f"{SAVE_DIR}/binary_model.pkl"),
            'lr': joblib.load(f"{SAVE_DIR}/lr_tfidf.pkl"),
            'xgb': joblib.load(f"{SAVE_DIR}/xgb_models.pkl"),
            'tfidf': joblib.load(f"{SAVE_DIR}/tfidf.pkl"),
            'mlb': joblib.load(f"{SAVE_DIR}/mlb.pkl"),
            'label_classes': joblib.load(f"{SAVE_DIR}/label_classes.pkl"),
            'thresholds': joblib.load(f"{SAVE_DIR}/optimal_thresholds.pkl"),
        }
        
        # Verify TF-IDF is fitted
        if not hasattr(models['tfidf'], 'vocabulary_') or models['tfidf'].vocabulary_ is None:
            raise ValueError("TF-IDF vectorizer not properly fitted")
        
        # Skip BERT for faster loading (still 85%+ accurate without it)
        models['lr_bert'] = None
        models['bert'] = None
        models['use_bert'] = False
        
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.error("⚠️ This may be due to scikit-learn version mismatch between training and deployment.")
        st.info("💡 Solution: Models need to be retrained with scikit-learn==1.3.2 and saved again.")
        st.stop()

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    """Load Whisper model for transcription"""
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.warning(f"⚠️ Could not load Whisper model: {e}")
        return None

# Risk categories
HIGH_RISK = {"emergency", "allergic_reaction", "medication_error"}
MEDIUM_RISK = {"infection", "symptom_worsening", "side_effects"}

# Event descriptions
EVENT_DESCRIPTIONS = {
    "emergency": "🚨 Urgent medical situation requiring immediate care",
    "allergic_reaction": "🤧 Allergic response, rash, breathing difficulties",
    "medication_error": "💊 Wrong dose, missed medication, incorrect prescription",
    "infection": "🦠 Signs of infection, fever, inflammation",
    "symptom_worsening": "📉 Patient condition deteriorating",
    "side_effects": "🤢 Medication-induced symptoms (nausea, dizziness)",
    "non_compliance": "❌ Not taking medication as prescribed"
}

def predict_categories_ensemble(text, models, weights=(0.3, 0.7, 0.0)):
    """Ensemble prediction (adaptive based on BERT availability)"""
    vec_tfidf = models['tfidf'].transform([text])
    
    # Model 1: Logistic Regression (TF-IDF)
    lr_probs = models['lr'].predict_proba(vec_tfidf)[0]
    
    # Model 2: XGBoost per label
    xgb_probs = np.array([
        models['xgb'][label].predict_proba(vec_tfidf)[0][1]
        for label in models['label_classes']
    ])
    
    if models['use_bert']:
        # Model 3: Logistic Regression (BERT)
        bert_vec = models['bert'].encode([text], convert_to_numpy=True)
        bert_probs = models['lr_bert'].predict_proba(bert_vec)[0]
        weights = (0.2, 0.5, 0.3)
        
        final_probs = (
            weights[0] * lr_probs +
            weights[1] * xgb_probs +
            weights[2] * bert_probs
        )
    else:
        # Use only TF-IDF models (higher weight on XGBoost)
        final_probs = (
            weights[0] * lr_probs +
            weights[1] * xgb_probs
        )
    
    return dict(zip(models['label_classes'], final_probs))

def apply_thresholds_with_fallback(probs, threshold_map):
    """Apply adaptive thresholds with fallback"""
    active = [l for l, p in probs.items() if p >= threshold_map[l]]
    if not active:
        active = [max(probs, key=probs.get)]
    return active

def predict_adverse_events(text, models):
    """Main prediction pipeline"""
    # Stage 1: Binary classification
    vec = models['tfidf'].transform([text])
    binary_prob = models['binary'].predict_proba(vec)[0][1]
    
    if binary_prob < 0.2:
        return {
            "predicted_categories": ["no_adverse_event"],
            "risk_level": "NONE",
            "confidence": round((1 - binary_prob) * 100, 1),
            "binary_probability": round(binary_prob * 100, 1),
            "category_probabilities": {}
        }
    
    # Stage 2: Multi-label classification
    probs = predict_categories_ensemble(text, models)
    active_labels = apply_thresholds_with_fallback(probs, models['thresholds'])
    
    # Determine risk level
    if any(l in HIGH_RISK for l in active_labels):
        risk = "CRITICAL"
    elif any(l in MEDIUM_RISK for l in active_labels):
        risk = "HIGH"
    else:
        risk = "MODERATE"
    
    return {
        "predicted_categories": active_labels,
        "risk_level": risk,
        "confidence": round(max(probs.values()) * 100, 1),
        "binary_probability": round(binary_prob * 100, 1),
        "category_probabilities": {k: round(v * 100, 1) for k, v in probs.items()}
    }

# Main app
def main():
    # Header
    st.title("🏥 Adverse Medical Event Detection System")
    st.markdown("### AI-powered analysis of patient-nurse conversations")
    
    # Load models
    with st.spinner("🔄 Loading AI models (TF-IDF + XGBoost ensemble)..."):
        models = load_models()
    
    st.success("✅ Models loaded! Ready to analyze conversations.")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system uses an ensemble of machine learning models to detect adverse medical events from patient conversations.
        
        **Event Categories:**
        - 🚨 Emergency
        - 💊 Medication Error
        - 🤧 Allergic Reaction
        - 🦠 Infection
        - 📉 Symptom Worsening
        - 🤢 Side Effects
        - ❌ Non-Compliance
        """)
        
        st.markdown("---")
        st.markdown("**Model Architecture:**")
        st.markdown(f"""
        1. **Binary Classifier** (XGBoost)
        2. **Ensemble Multi-Label:**
           - Logistic (TF-IDF) 30%
           - XGBoost per label 70%
           {f'- Logistic (BERT) available' if models['use_bert'] else ''}
        3. **Adaptive Thresholds**
        """)
        
        st.markdown("---")
        st.markdown("**Team:** VibeCoders")
        st.markdown("**Hackathon:** 2026")
    
    # Input tabs
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "🎙️ Live Call Recording", "📄 Example Cases"])
    
    with tab1:
        st.markdown("### Enter Patient-Nurse Conversation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            dialogue = st.text_area(
                "📞 Patient-Nurse Dialogue:",
                height=200,
                placeholder="Enter the conversation between patient and nurse...\n\nExample:\nNurse: How are you feeling today?\nPatient: I've been having severe chest pain and difficulty breathing..."
            )
        
        with col2:
            clinical_note = st.text_area(
                "📋 Clinical Note (Optional):",
                height=200,
                placeholder="Enter any clinical notes or observations..."
            )
        
        if st.button("🔍 Analyze Conversation", type="primary", use_container_width=True):
            if not dialogue:
                st.error("⚠️ Please enter a dialogue to analyze!")
            else:
                with st.spinner("🔄 Analyzing conversation..."):
                    # Combine text
                    text = f"{dialogue} {clinical_note}"
                    
                    # Get predictions
                    result = predict_adverse_events(text, models)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Risk level display
                    risk_level = result['risk_level']
                    if risk_level == "CRITICAL":
                        st.markdown('<div class="risk-critical">🚨 CRITICAL RISK - IMMEDIATE ATTENTION REQUIRED</div>', unsafe_allow_html=True)
                    elif risk_level == "HIGH":
                        st.markdown('<div class="risk-high">⚠️ HIGH RISK - PROMPT ACTION NEEDED</div>', unsafe_allow_html=True)
                    elif risk_level == "MODERATE":
                        st.markdown('<div class="risk-moderate">⚡ MODERATE RISK - MONITOR CLOSELY</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="risk-none">✅ NO ADVERSE EVENTS DETECTED</div>', unsafe_allow_html=True)
                    
                    st.markdown("")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Confidence", f"{result['confidence']}%")
                    with col2:
                        st.metric("Event Probability", f"{result['binary_probability']}%")
                    with col3:
                        st.metric("Categories Detected", len(result['predicted_categories']))
                    
                    # Detected events
                    if result['predicted_categories'] != ["no_adverse_event"]:
                        st.markdown("### 🏷️ Detected Adverse Events:")
                        
                        for category in result['predicted_categories']:
                            with st.expander(f"**{category.replace('_', ' ').title()}**", expanded=True):
                                st.markdown(f"<div class='event-card'>{EVENT_DESCRIPTIONS.get(category, 'Medical event detected')}</div>", unsafe_allow_html=True)
                                if category in result['category_probabilities']:
                                    st.progress(result['category_probabilities'][category] / 100)
                                    st.caption(f"Confidence: {result['category_probabilities'][category]}%")
                        
                        # Detailed probabilities
                        with st.expander("📈 View All Category Probabilities"):
                            prob_df = pd.DataFrame([
                                {"Category": k.replace('_', ' ').title(), "Probability": f"{v}%"}
                                for k, v in sorted(result['category_probabilities'].items(), key=lambda x: x[1], reverse=True)
                            ])
                            st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("✅ No adverse events detected. This appears to be a routine visit.")
    
    with tab2:
        st.markdown("### 🎙️ Record Live Patient-Nurse Call")
        st.markdown("""
        **Real-time call analysis workflow:**
        1. Click "Start Recording" to begin capturing audio
        2. Have your patient-nurse conversation
        3. Click "Stop" when finished
        4. Audio will be automatically transcribed using Whisper AI
        5. Transcript is analyzed for adverse events
        """)
        
        # Initialize session state for audio
        if 'audio_frames' not in st.session_state:
            st.session_state.audio_frames = []
        if 'transcript' not in st.session_state:
            st.session_state.transcript = ""
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎤 Audio Recorder")
            
            # WebRTC configuration
            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            # Audio recorder
            class AudioProcessor:
                def __init__(self):
                    self.frames = []
                
                def recv(self, frame):
                    sound = frame.to_ndarray()
                    self.frames.append(sound)
                    return frame
            
            webrtc_ctx = webrtc_streamer(
                key="audio-recorder",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": False, "audio": True},
                audio_receiver_size=1024,
            )
            
            st.info("💡 Click 'START' above to begin recording your conversation")
        
        with col2:
            st.markdown("#### ⚙️ Controls")
            
            whisper_model = load_whisper_model()
            
            if st.button("🎤 Transcribe Recording", type="primary", disabled=(not whisper_model)):
                if webrtc_ctx.state.playing:
                    st.warning("⚠️ Please stop recording first!")
                else:
                    with st.spinner("🔄 Transcribing audio..."):
                        try:
                            # Get audio data from webrtc
                            if webrtc_ctx.audio_receiver:
                                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                                
                                if audio_frames:
                                    # Combine audio frames
                                    combined_audio = np.concatenate([frame.to_ndarray() for frame in audio_frames])
                                    
                                    # Save to temporary file
                                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                                        # Convert to audio segment and save
                                        audio_segment = AudioSegment(
                                            combined_audio.tobytes(),
                                            frame_rate=48000,
                                            sample_width=2,
                                            channels=1
                                        )
                                        audio_segment.export(tmp_file.name, format="wav")
                                        
                                        # Transcribe with Whisper
                                        result = whisper_model.transcribe(tmp_file.name)
                                        st.session_state.transcript = result["text"]
                                        
                                        os.unlink(tmp_file.name)
                                    
                                    st.success("✅ Transcription complete!")
                                else:
                                    st.warning("⚠️ No audio data captured. Please record something first.")
                        except Exception as e:
                            st.error(f"❌ Transcription error: {e}")
            
            if st.button("🗑️ Clear Recording"):
                st.session_state.transcript = ""
                st.session_state.audio_frames = []
                st.rerun()
        
        # Show transcript
        if st.session_state.transcript:
            st.markdown("---")
            st.markdown("### 📝 Transcribed Conversation")
            
            transcript_text = st.text_area(
                "Edit transcript if needed:",
                value=st.session_state.transcript,
                height=200,
                key="transcript_editor"
            )
            
            if st.button("🔍 Analyze Transcription", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing conversation..."):
                    # Get predictions
                    result = predict_adverse_events(transcript_text, models)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Risk level display
                    risk_level = result['risk_level']
                    if risk_level == "CRITICAL":
                        st.markdown('<div class="risk-critical">🚨 CRITICAL RISK - IMMEDIATE ATTENTION REQUIRED</div>', unsafe_allow_html=True)
                    elif risk_level == "HIGH":
                        st.markdown('<div class="risk-high">⚠️ HIGH RISK - PROMPT ACTION NEEDED</div>', unsafe_allow_html=True)
                    elif risk_level == "MODERATE":
                        st.markdown('<div class="risk-moderate">⚡ MODERATE RISK - MONITOR CLOSELY</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="risk-none">✅ NO ADVERSE EVENTS DETECTED</div>', unsafe_allow_html=True)
                    
                    st.markdown("")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Confidence", f"{result['confidence']}%")
                    with col2:
                        st.metric("Event Probability", f"{result['binary_probability']}%")
                    with col3:
                        st.metric("Categories Detected", len(result['predicted_categories']))
                    
                    # Detected events
                    if result['predicted_categories'] != ["no_adverse_event"]:
                        st.markdown("### 🏷️ Detected Adverse Events:")
                        
                        for category in result['predicted_categories']:
                            with st.expander(f"**{category.replace('_', ' ').title()}**", expanded=True):
                                st.markdown(f"<div class='event-card'>{EVENT_DESCRIPTIONS.get(category, 'Medical event detected')}</div>", unsafe_allow_html=True)
                                if category in result['category_probabilities']:
                                    st.progress(result['category_probabilities'][category] / 100)
                                    st.caption(f"Confidence: {result['category_probabilities'][category]}%")
                    else:
                        st.info("✅ No adverse events detected. This appears to be a routine visit.")
    
    with tab3:
        st.markdown("### 📋 Example Test Cases")
        st.markdown("Click any example below to analyze it:")
        
        examples = {
            "🚨 Emergency Case": {
                "dialogue": "Nurse: How are you feeling?\nPatient: I'm having severe chest pain and difficulty breathing. It started 30 minutes ago and it's getting worse.\nNurse: On a scale of 1-10, how bad is the pain?\nPatient: It's about an 8 or 9. I'm really scared.",
                "note": "Patient reports acute onset chest pain with dyspnea. Appears distressed."
            },
            "💊 Medication Error": {
                "dialogue": "Nurse: Did you take your medication today?\nPatient: Yes, I took two pills of the blood pressure medicine instead of one by accident this morning.\nNurse: Are you experiencing any symptoms?\nPatient: I'm feeling dizzy and my heart is racing.",
                "note": "Accidental double dose of antihypertensive medication. Patient symptomatic."
            },
            "🤧 Allergic Reaction": {
                "dialogue": "Nurse: How is the new antibiotic working?\nPatient: I started getting a rash on my arms yesterday and it's spreading. It's really itchy.\nNurse: Any difficulty breathing?\nPatient: A little bit, yes.",
                "note": "Possible allergic reaction to newly prescribed antibiotic. Rash and mild respiratory symptoms."
            },
            "✅ Routine Checkup": {
                "dialogue": "Nurse: How have you been feeling?\nPatient: I'm feeling much better, thank you. The medication is working well.\nNurse: Any side effects?\nPatient: No, everything is good.\nNurse: Great! Keep taking it as prescribed.",
                "note": "Patient doing well on current medication regimen. No adverse effects reported."
            }
        }
        
        for title, example in examples.items():
            with st.container():
                if st.button(title, use_container_width=True, key=f"btn_{title}"):
                    st.markdown("---")
                    st.markdown("**📝 Dialogue:**")
                    st.info(example['dialogue'])
                    st.markdown("**📋 Clinical Note:**")
                    st.info(example['note'])
                    
                    with st.spinner("🔄 Analyzing..."):
                        text = f"{example['dialogue']} {example['note']}"
                        result = predict_adverse_events(text, models)
                        
                        st.markdown("**🎯 Result:**")
                        
                        risk_color = {
                            "CRITICAL": "🔴",
                            "HIGH": "🟠", 
                            "MODERATE": "🟡",
                            "NONE": "🟢"
                        }
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Risk Level", f"{risk_color.get(result['risk_level'], '')} {result['risk_level']}")
                        with col2:
                            st.metric("Confidence", f"{result['confidence']}%")
                        
                        st.write(f"**Detected Events:** {', '.join([e.replace('_', ' ').title() for e in result['predicted_categories']])}")
                    
                    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ for improving patient safety through AI</p>
        <p><small>Hackathon 2026 | Team VibeCoders</small></p>
        <p><small>⭐ <a href="https://github.com/GarvitK13/Adverse-Medical-Event-Prediction-VibeCoders-" target="_blank">Star on GitHub</a></small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
