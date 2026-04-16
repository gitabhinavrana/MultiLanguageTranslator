import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 🌍 Expanded language list (25+)
LANG_MAP = {
    "German": "deu_Latn",
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Hindi": "hin_Deva",
    "Romanian": "ron_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Dutch": "nld_Latn",
    "Russian": "rus_Cyrl",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Arabic": "arb_Arab",
    "Turkish": "tur_Latn",
    "Polish": "pol_Latn",
    "Swedish": "swe_Latn",
    "Danish": "dan_Latn",
    "Finnish": "fin_Latn",
    "Greek": "ell_Grek",
    "Czech": "ces_Latn",
    "Hungarian": "hun_Latn",
    "Thai": "tha_Thai",
    "Vietnamese": "vie_Latn",
    "Indonesian": "ind_Latn",
    "Ukrainian": "ukr_Cyrl"
}

# 🎤 Speech-to-text helper (Gradio handles mic input)
def speech_to_text(audio):
    if audio is None:
        return ""
    return audio  # Gradio already converts speech → text

# 🔁 Translation function
def translate_text(text, destination_language):
    if not text or not text.strip():
        return "Please provide input text or use the microphone."

    tgt_lang = LANG_MAP.get(destination_language)
    if tgt_lang is None:
        return "Invalid language selected."

    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(text, return_tensors="pt")

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)
    )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# 🎯 Combined function (voice OR text)
def process_input(text, audio, language):
    if audio:
        text = speech_to_text(audio)  # Transcribe audio to text
    return translate_text(text, language)

# 🎨 UI
demo = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(label="Type text (or use mic below)", lines=4),
        gr.Microphone(type="numpy", label="🎤 Speak here"), # Changed type to 'numpy'
        gr.Dropdown(list(LANG_MAP.keys()), label="Select Target Language")
    ],
    outputs=gr.Textbox(label="Translated Text"),
    title="🌍 Multi-language Translator with Voice Input",
    description=(
        "You can either type text or use your microphone. "
        "Speech will be converted to text automatically and translated "
        "into your selected language."
    )
)

demo.launch(share=True)
