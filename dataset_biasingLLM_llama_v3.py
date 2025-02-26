import torch
from transformers import AutoProcessor, AutoFeatureExtractor, AutoTokenizer
#from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch.utils.data as data_utils
import torchaudio
import pandas as pd
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

def collate_wrapper(batch):
    print(len(batch))
    uttnames = [i[0] for i in batch]
    fbank = torch.cat([i[1] for i in batch])
    tgt = [i[2] for i in batch]
    blist = [] # batch level biasing list
    for i in batch:
        for word in i[3]:
            if word not in blist:
                blist.append(word)
    return uttnames, fbank, tgt, blist

class MyCollator_biasingLLM:
    def __init__(self, audio_processor, tokenizer):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

    def __call__(self, batch):
        #uttname, waveform, bwords, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt
        uttnames = [ b[0] for b in batch]
        #mel_feats = torch.stack([ b[1] for b in batch])
        fbank = torch.cat([ 
            self.audio_processor(
                b[1], sampling_rate=16000, return_tensors="pt"
            ).input_features for b in batch]
        )
        blist = [ b[2] for b in batch] ### note in batch sample biasing words !!! extends to 30 or 50
        blist = list(set(sum(blist, []))) # batch level biasing list

        pre_speech_prompt = [ b[3] for b in batch]
        post_speech_prompt = [ b[4] for b in batch]
        output_prompt = [ f'{self.tokenizer.bos_token} '+ b[5] +f' {self.tokenizer.eos_token}' for b in batch]
        complete_prompt = [ b[6] for b in batch]

        # blist_tokenized contains (input_ids, attention_mask)
        blist_tokenized = self.tokenizer(blist, padding="longest", return_tensors='pt', truncation=False, add_special_tokens=False)
        pre_tokenized_ids = self.tokenizer(pre_speech_prompt, padding="longest", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        post_tokenized_ids = self.tokenizer(post_speech_prompt, padding="longest", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        output_tokenized_ids = self.tokenizer(output_prompt, padding="longest", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        
        return uttnames, fbank, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids


class BiasingASRDataset(Dataset):
    def __init__(self, json_file, mode='train'):
        self.data_frame = pd.read_json(json_file)
        #self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle dataset
        self.data_idx = list(self.data_frame.keys())
        self.mode = mode #mode = 'train' return target
        #self.tokenizer = tokenizer
        #self.audio_proc = audio_proc
        
    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        # Load audio
        uttname = self.data_idx[idx]
        data_row = self.data_frame[uttname]
        target = data_row["words"].lower()
        num_samples = np.random.randint(1, 5)
        rand_utt_idxs = np.random.choice(len(self.data_idx), num_samples)
        rand_bwords = [ self.data_frame[self.data_idx[rand_utt_idx]]["blist"] for rand_utt_idx in rand_utt_idxs]
        rand_bwords = sum(rand_bwords, [])

        # Load audio
        audio_path = data_row['audio']
        if pd.isna(audio_path):
            waveform = None
        else:
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.squeeze()
            #fbank = self.audio_proc(waveform, sampling_rate=16000, return_tensors="pt").input_features
        if self.mode == 'train':
            lower_bwords = [ w.lower() for w in data_row["blist"]]
            lower_rand_bwords = [ w.lower() for w in rand_bwords]
            bwords = lower_bwords + lower_rand_bwords + ['<unbiased>']
        
        elif self.mode == 'test_utt':
            lower_bwords = [ w.lower() for w in data_row["blist"]]
            bwords = lower_bwords + ['<unbiased>']
        
        elif self.mode == 'test_utt_rand3':
            num_samples = 3
            rand_utt_idxs = np.random.choice(len(self.data_idx), num_samples)
            rand_bwords = [ self.data_frame[self.data_idx[rand_utt_idx]]["blist"] for rand_utt_idx in rand_utt_idxs]
            rand_bwords = sum(rand_bwords, [])
            lower_bwords = [ w.lower() for w in data_row["blist"]]
            rand_bwords = [ w.lower() for w in rand_bwords]
            bwords = lower_bwords + rand_bwords + ['<unbiased>']
            
        
        return uttname, waveform, target, bwords


class InstructionalBiasingASRDataset(BiasingASRDataset):
    def __init__(self, json_file, mode='train'):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

        Returns:
            None
        """
        super().__init__(json_file, mode)
    
    def __getitem__(self, idx):
        uttname, waveform, target, bwords = super().__getitem__(idx)
        instruction_phrase = f'Transcribe the input audio'

        pre_speech_prompt = f"Instruction:\n{instruction_phrase}."
        pre_speech_prompt += "\n\nInput:\n<speech>"
        post_speech_prompt = f"</speech>\n\n" + \
             "Output:\n"
        output_prompt = "{"
        output_prompt += f'  "transcript": "{target}", '
        output_prompt = output_prompt.rstrip(',\n') + "}"

        complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt
        return uttname, waveform, bwords, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt


# Example usage
if __name__ == "__main__":

    #llm_tokenizer = AutoTokenizer.from_pretrained('../qwen2/1.5B/')
    llm_tokenizer = AutoTokenizer.from_pretrained('../llama3.1/8B/')
    #llm_tokenizer = AutoTokenizer.from_pretrained('../llama3.2/1B/')
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # for llama3.1-8b
    whisper_proc = WhisperProcessor.from_pretrained('../whisper_large_v2')
    dataset = InstructionalBiasingASRDataset(json_file='data/LibriSpeech/biasingASR_json/train_clean_100_full.json',
     mode='test_utt_rand3')
    
    my_collator = MyCollator_biasingLLM(whisper_proc, llm_tokenizer)
    
    train_loader = data_utils.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=my_collator, num_workers=1)
    uttnames, fbank, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = next(iter(train_loader))
    print(fbank.shape)
    print([llm_tokenizer.decode(utt) for utt in output_tokenized_ids])
    print([llm_tokenizer.decode(bwords) for bwords in blist_tokenized['input_ids']])

