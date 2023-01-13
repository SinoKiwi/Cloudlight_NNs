'''dataproc.py
   数据预处理
   在模型中，音频数据将以Spectrogram的形式进行训练
   首先加载train.tsv，对应关系以文件名为键、文本为值的形式存储在字典中
   然后，定义音频处理，传入文件路径，使用torchaudio加载并处理
   再建立字典，把文本构建为tensor
   之后，处理为dataset（原计划使用TensorDataset,但是TensorDataset不支持变成tensor，最后仍然使用继承Dataset类）
   在train.py中，用dataloader从dataset加载batch
'''

import pandas as pd
import torchaudio
import torch
from collections import Counter

#继承Dataset类
class DATASET(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


#load tsv file as a dict
def load_tsv(filepath):
    data = pd.read_csv(filepath, sep='\t', header=None)
    data_dict = data.set_index(0).to_dict()[1]
    return data_dict

#将音频文件加载并转换为tensor
def audio_to_tensor(filepath):
    # 读取音频文件
    waveform, sample_rate = torchaudio.load(filepath)
    print("Waveform:", waveform.size())

    #重采样为16000Hz
    # 使用 Resample 类进行重采样
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

    # 转换成Mel Spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
    print("Mel Spectrogram:", mel_spectrogram.size())

    # 转化成Tensor
    mel_spectrogram_tensor = mel_spectrogram.transpose(0, 1)
    #print(type(mel_spectrogram_tensor))
    print("Mel Spectrogram Tensor:", mel_spectrogram_tensor.size())
    return mel_spectrogram_tensor

#文本转tensor
def text_to_tensor(text, vocabDict):
    #text_bytes = text.encode("utf-8")
    #print(f"text:{text}")
    #text_tensor = torch.tensor(list(text_bytes), dtype=torch.int8)
    tokens = list(text)
    ids = [vocabDict[char] for char in tokens]
    #print(f"ids:{ids}")
    text_tensor = torch.FloatTensor(ids)
    return text_tensor

'''#封装多个tensor为列表(Seq2Seq+Transformer可处理变长序列，不需要填充)
def pack_tensors(*args):
    #封装
    pack = []
    for content in args:
        pack.append(content)
    
    return pack '''#直接在all_steps内处理为列表

#封装tensor对为dataset
#def tensor_to_DATASET(data, labels):
    
#    return dataset

def build_vocab_dict(source_list):
    # 使用 join() 函数将所有句子连接起来
    all_chars = ''.join(source_list)

    # 使用 Counter 类统计字出现的次数
    char_freq = Counter(all_chars)

    # 使用 dict() 函数将 Counter 对象转换为字典，并按照 value 的值从大到小排序
    char_freq_sorted = dict(sorted(char_freq.items(), key=lambda x: x[1], reverse=True))

    #重新编号
    for index, (key, value) in enumerate(char_freq_sorted.items()):
        char_freq_sorted[key] = index + 1

    return char_freq_sorted

def all_steps(audio_path, file_dir):
    datadict = load_tsv(file_dir)
    audio_tensors = []
    sentence_list = []
    text_tensors = []
    #构建词典的句子
    for textContent in datadict.values():
        sentence_list.append(textContent)
    #建立字典
    vocab_dict = build_vocab_dict(sentence_list)
    #print(f"vocab_dict:{vocab_dict}")
    #音频的tensor
    for audio_name in datadict.keys():
        audio_tensors.append(audio_to_tensor(audio_path+audio_name))
    #文本的tensor
    for text_content in datadict.values():
        text_tensors.append(text_to_tensor(text_content, vocab_dict))
    #final_dataset = tensor_to_DATASET(audio_tensors, text_tensors)
    final_dataset = DATASET(audio_tensors, text_tensors)
    #print("vocab_dict:", vocab_dict)
    #print("audio_tensors:", audio_tensors)
    #print("text_tensors:", text_tensors)
    return final_dataset, vocab_dict


