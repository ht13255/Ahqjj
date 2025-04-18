import streamlit as st import os import tempfile import torch from pydub import AudioSegment from spleeter.separator import Separator from pathlib import Path import tensorflow as tf import concurrent.futures import traceback

Real-Time-Voice-Cloning imports

from encoder import inference as encoder from synthesizer.inference import Synthesizer from vocoder import inference as vocoder

--- Configuration ---

ENCODER_MODEL = 'models/encoder.pt' SYNTHESIZER_MODEL = 'models/synthesizer.pt' VOCODER_MODEL = 'models/vocoder.pt' DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' NUM_THREADS = os.cpu_count() or 1

--- Threading & Performance ---

torch.set_num_threads(NUM_THREADS) torch.set_num_interop_threads(NUM_THREADS) tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS) tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

--- Utility Functions with Error Handling ---

@st.cache_resource(show_spinner=False) def load_models(enc_path, syn_path, voc_path): try: st.sidebar.info(f"Loading models on {DEVICE}...") if not all(os.path.exists(p) for p in [enc_path, syn_path, voc_path]): raise FileNotFoundError("One or more model files not found. Ensure encoder.pt, synthesizer.pt, vocoder.pt are in the models/ folder.") encoder.load_model(enc_path) synthesizer = Synthesizer(syn_path) vocoder.load_model(voc_path) return synthesizer except Exception as e: st.sidebar.error(f"Model loading failed: {str(e)}") st.stop()

@st.cache_data def separate_audio(input_path): try: out_dir = tempfile.mkdtemp() Separator('spleeter:2stems').separate_to_file(input_path, out_dir) base = Path(input_path).stem vocals = os.path.join(out_dir, base, 'vocals.wav') accomp = os.path.join(out_dir, base, 'accompaniment.wav') if not os.path.exists(vocals) or not os.path.exists(accomp): raise RuntimeError("Separation failed: output files not found.") return vocals, accomp except Exception as e: st.error(f"Audio separation error: {e}") raise

def fine_tune_synthesizer(samples, epochs=100, batch_size=4, lr=1e-4): try: if len(samples) < 10: raise ValueError("At least 10 samples required for fine-tuning.") device = torch.device(DEVICE) # Preprocess in parallel with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor: futures = [executor.submit(encoder.preprocess_wav, path) for path in samples] wavs = [f.result() for f in futures] specs = synthesizer.melspectrograms(wavs) # Dataset class SampleDataset(torch.utils.data.Dataset): def init(self, specs): self.specs = specs def len(self): return len(self.specs) def getitem(self, idx): return torch.tensor(self.specs[idx]) dataset = SampleDataset(specs) loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_THREADS) optimizer = torch.optim.Adam(synthesizer.model.parameters(), lr=lr) scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) for epoch in range(epochs): for batch in loader: optimizer.zero_grad() loss = synthesizer.compute_loss(batch.to(device)) loss.backward() optimizer.step() scheduler.step() synth_path = 'models/synthesizer_finetuned.pt' torch.save(synthesizer.model.state_dict(), synth_path) return synth_path except Exception as e: st.error(f"Fine-tuning failed: {e}") traceback.print_exc() return None

--- Streamlit App ---

st.set_page_config(page_title="AI Song Cover", layout="wide") st.title("🎤 AI Song Cover Generator — 안정화 버전 🎶")

Sidebar

with st.sidebar: st.header("🔧 Settings & Info") st.text(f"Device: {DEVICE}") st.text(f"Threads: {NUM_THREADS}") if st.button("Refresh Models"): load_models.clear() st.checkbox("Show requirements.txt", key='show_req') if st.session_state.show_req: st.code( """ streamlit torch numpy scipy spleeter pydub tensorflow git+https://github.com/CorentinJ/Real-Time-Voice-Cloning.git """ )

Load models

synthesizer = load_models(ENCODER_MODEL, SYNTHESIZER_MODEL, VOCODER_MODEL)

Tabs

tab1, tab2 = st.tabs(["🛠 Fine-Tune Synthesizer", "🎵 Generate AI Cover"])

with tab1: st.header("Fine-Tuning Settings") samples = st.file_uploader("Upload 10+ WAV samples", type='wav', accept_multiple_files=True) epochs = st.slider("Epochs", 10, 500, 100) lr = st.number_input("Learning Rate", 1e-5, 1e-3, 1e-4, format="%.5f") if st.button("Start Fine-Tuning"): if not samples: st.error("Please upload samples before fine-tuning.") else: tmp = [] for f in samples: path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name open(path, 'wb').write(f.getbuffer()) tmp.append(path) finetuned = fine_tune_synthesizer(tmp, epochs=epochs, lr=lr) if finetuned: st.success(f"Fine-tuning complete! Model saved to {finetuned}")

with tab2: st.header("Generate AI Cover") st.markdown("Upload 3+ WAV samples for embedding extraction.") voice_files = st.file_uploader("Voice Samples", type='wav', accept_multiple_files=True, key='conv_samples') if st.button("Generate Cover"): if not voice_files or len(voice_files) < 3: st.error("Please upload at least 3 voice samples.") else: try: # Save and preprocess samples paths = [] for vf in voice_files: p = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name open(p, 'wb').write(vf.getbuffer()) paths.append(p) embed = encoder.embed_utterance(encoder.preprocess_wav(paths)) st.info("Embedding created.") song = st.file_uploader("Song File (MP3/WAV)", type=['mp3', 'wav'], key='song') if not song: st.error("Please upload a song file.") else: song_path = tempfile.NamedTemporaryFile(delete=False, suffix=Path(song.name).suffix).name open(song_path, 'wb').write(song.getbuffer()) st.info("Separating vocals...") vocals, accomp = separate_audio(song_path) st.info("Converting vocals...") wav = encoder.preprocess_wav(vocals) specs = synthesizer.synthesize_spectrograms([wav], [embed]) generated = vocoder.infer_waveform(specs[0]) out_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name synthesizer.save_wav(generated, out_wav) st.info("Mixing audio...") conv = AudioSegment.from_wav(out_wav) acc = AudioSegment.from_wav(accomp) min_len = min(len(conv), len(acc)) mix = acc[:min_len].overlay(conv[:min_len]) out_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name mix.export(out_mp3, format='mp3') st.success("AI Cover Ready!") st.audio(out_mp3) with open(out_mp3, 'rb') as f: st.download_button("Download", f, file_name="ai_cover.mp3") except Exception as e: st.error(f"Generation failed: {e}") st.error(traceback.format_exc())
