import io
import streamlit as st
import torchaudio
from speechbrain.pretrained import EncoderClassifier

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

stt_button = Button(label="Speak", width=100)



def load_audio():
    """Создание формы для загрузки аудио"""

    #Форма для загрузки средствами Streamlit

    uploaded_file = st.file_uploader(
        label='Загрузите аудиофайл для распознавания')

    if uploaded_file is not None:
        #Получение загруженного аудио
        audio_data = uploaded_file.getvalue()
        #st.write(audio_data)
        #Вывод аудиоплеера
        st.audio(audio_data)
        # return io.BytesIO(audio_data)
        with open("temp.wav","wb") as f:
            f.write(uploaded_file.getbuffer())


@st.cache(allow_output_mutation=True)
def model_loading():
    return EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")    

language_id = model_loading()

audio_data = load_audio()

result = st.button('Распознать аудиофайл')
#Если кнопка нажата, то запускаем распознавание


stt_button.js_on_event("button_click", CustomJS(code="""
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
 
    recognition.onresult = function (e) {
        var value = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
            }
        }
        if ( value != "") {
            document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
        }
    }
    recognition.start();
    """))

result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)

if result:
    if "GET_TEXT" in result:
        st.write(result.get("GET_TEXT"))











if result:
    signal = language_id.load_audio("temp.wav")
    prediction =  language_id.classify_batch(signal)
    st.write(prediction[3][0] + ' with probability' + f" {prediction[1].exp().item()}")
