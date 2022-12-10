import io
import streamlit as st
import torchaudio
from speechbrain.pretrained import EncoderClassifier

import wave

class SoundFile:

    def __init__(self, filename):
        self.filename = filename
        self.file = wave.open(filename, 'rb')
        self.frameCount = self.file.getnframes()

    def describe(self):
        return self.file.getparams()

    def close(self):
        return self.file.close()

    # Returns an array of 16 bit words
    # Joins bytes of the frame array
    def byteArray(self):
        curFrame = 0
        frameArray = []
        while curFrame < self.frameCount:
            frame = self.file.readframes(1)
            # Assumes 16 bit frame
            b1 = frame[0]  # right bit
            b2 = frame[1]  # left bit
            b = (b2 << 8) + b1
            # print('{0:08b} {1:08b} -> {2:016b}'.format(b2, b1, b))
            frameArray.append(b)
            curFrame += 1
        return frameArray

    def byteFormatedString(self, bArray):
        bStrings = []
        for idx, val in enumerate(bArray):
            hexStr = ''
            if ((idx % 30 == 0) and (idx > 0)):
                hexStr += '\n'
            hexStr += ('0x{:04x}'.format(val))
            bStrings.append(hexStr)
        return ",".join(bStrings)

    # Converts wave two's compliment to positive
    # integers and shifts existing positive integers up
    def twosToOnes(self, bArray):
        onesArray = []
        for b in bArray:
            if b > 0x8000:
                b2 = 0x8000 - (0xffff & (~b + 1))
                # print('{0:016b} NN {1:016b}'.format(b,b2))

            else:
                b2 = b + 0x8000
                # print('{0:016b} PP {1:016b}'.format(b,b2))
            # print('{0} -> {1}'.format(b,b2))
            onesArray.append(b2)

        return (onesArray)

    def toDAC12(self, bArray):
        dacArray = []
        for b in bArray:
            dacVal = b >> 4
            # print('0x{:04x}'.format(dacVal))
            # print(dacVal)
            if dacVal > 4095:
                print('ALERT - Out of band')
            dacArray.append(b >> 4)

        return dacArray

    def writeFile(self):
        txtName = self.filename.replace('.wav', '.h')
        txt = open(txtName, 'w')
        txt.truncate()
        txt.write('\n\n// Auto generated\nint frame_count = {};\n'.format(self.frameCount))
        txt.write('const uint16_t wave_data[{}] =\n'.format(self.frameCount))
        txt.write('{')
        bArray = self.byteArray()
        bArray = self.twosToOnes(bArray)
        bArray = self.toDAC12(bArray)
        txt.write(self.byteFormatedString(bArray))
        txt.write('};\n')
        txt.close()


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
        return SoundFile(audio_data)


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
