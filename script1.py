import io
import streamlit as st
import torchaudio
from speechbrain.pretrained import EncoderClassifier


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
        return audio_data
        №return io.BytesIO(audio_data)

@st.cache(allow_output_mutation=True)
def model_loading():
    return EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

language_id = model_loading()

audio_data = load_audio()

result = st.button('Распознать аудиофайл')
#Если кнопка нажата, то запускаем распознавание

if result:
    signal = language_id.load_audio(audio_data)
    prediction =  language_id.classify_batch(signal)
    print(prediction[3][0] + ' with probability' + f" {prediction[1].exp().item()}")
