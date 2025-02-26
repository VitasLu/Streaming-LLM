import torch
from trainer_speechLLM_llama_v3_1b_noquant_noFeatExt import SpeechLLM # no whisper as feature extractor
import torch.utils.data as data_utils
from dataset_biasingLLM_llama_noFeatExt import InstructionalBiasingASRDataset, MyCollator_biasingLLM
from jiwer import wer
import numpy as np
import re
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from whisper_normalizer.english import EnglishTextNormalizer
pattern = "[!\"#$%&'()*+,\-/:;<=>?@[\]^_`{|}~]"
if __name__ == "__main__":
    gen_config = {
        'max_new_tokens': 200,
        'num_beams': 4,
        'do_sample': False,
        'min_length': 1,
        'temperature': 0.0,
        'top_p': 1.0,
        'repetition_penalty': 1.0,
        'length_penalty': 1.0,
    }
    # load the model include whisper
    ckpt_path = './exp4/speechLLM_llama1b_CIFwithQuantityLoss_prompt/libri100_speechLLM_llama1b/epoch=1.ckpt' 
    ckpt = torch.load(ckpt_path, map_location='cpu')

    model = SpeechLLM(**ckpt['hyper_parameters'])
    model_dict = model.state_dict()
    # skip whisper, "audio_encoder"
    trained_model_dict = ckpt['state_dict']
    load_dict = {}
    for k in model_dict:
        if k in trained_model_dict:
            load_dict[k] = trained_model_dict[k]
        else:
            print(f'{k} not in load into dict') # for debug, k expect to be llm-layers
    model.load_state_dict(load_dict, strict=False) #<All keys matched successfully>
    #model = SpeechLLM.load_from_checkpoint(ckpt_path)
    model.cuda()
    model.eval()
    tokenizer = model.llm_tokenizer
    my_collator = MyCollator_biasingLLM(tokenizer)
    
    test_mode = 'test_utt'
    dev_clean = InstructionalBiasingASRDataset(
        json_file='./data/LibriSpeech/dev_clean_full.json',
        mode=test_mode)
    dev_clean_dl = DataLoader(
        dev_clean, batch_size=1, shuffle=False, collate_fn=my_collator)

    dev_other = InstructionalBiasingASRDataset(
        json_file='./data/LibriSpeech/dev_other_full.json',
        mode=test_mode)
    dev_other_dl = DataLoader(
        dev_other, batch_size=1, shuffle=False, collate_fn=my_collator)

    test_clean = InstructionalBiasingASRDataset(
        json_file='./data/LibriSpeech/test_clean_full.json',
        mode=test_mode)
    test_clean_dl = DataLoader(
        test_clean, batch_size=1, shuffle=False, collate_fn=my_collator)
    
    test_other = InstructionalBiasingASRDataset(
        json_file='./data/LibriSpeech/test_other_full.json',
        mode=test_mode)
    test_other_dl = DataLoader(
        test_other, batch_size=1, shuffle=False, collate_fn=my_collator)
    
    val_sets = {'dev_clean':dev_clean_dl, 'dev_other':dev_other_dl, 'test_clean':test_clean_dl, 'test_other':test_other_dl}

    
    en_norm = EnglishTextNormalizer()
    
    for dataset in val_sets:
        hyps, refs, tot_we, num_failed = [], [], 0.0, 0.0
        print(f"evaluate dataset: {dataset}") 
        for idx, batch in enumerate(tqdm(val_sets[dataset])):
            with torch.no_grad():
                uttnames, audio_feat, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
                generated_output_text = model.generate(batch, gen_config, model.device)
                target_text = model.llm_tokenizer.decode(output_tokenized_ids[0], skip_special_tokens=False)
                extracted_pred = model.extract_prediction_values(generated_output_text[0])
                extracted_target = model.extract_prediction_values(target_text)

                keys = extracted_target.keys()
                pred_keys = extracted_pred.keys()

                if 'transcript' in pred_keys:
                    transcription = en_norm(extracted_pred['transcript'].lower())
                    refwords = en_norm(extracted_target['transcript'].lower())

                else:
                    num_failed +=1

                    continue
                utt_wer = wer(transcription, refwords)
                tot_we += utt_wer
                utt_hyp = f"{transcription} ({batch[0]})"
                utt_ref = f"{refwords} ({batch[0]})"
                hyps.append(utt_hyp)
                refs.append(utt_ref)
        print(f"WER for {dataset}: {np.round(tot_we/len(val_sets[dataset]), 4):.4f}")
        print(f"Valied Rate for {dataset}: {np.round((len(val_sets[dataset])-(num_failed))/len(val_sets[dataset]), 4):.4f}")

        expdir = os.path.join(os.path.dirname(ckpt_path), f'gen_testGPU_{test_mode}', dataset)
                
        if not os.path.exists(expdir):
            os.makedirs(expdir)
        with open(os.path.join(expdir, f"valied.log"), "w") as fout:
            fout.write(f"WER for {dataset}: {np.round(tot_we/len(val_sets[dataset]), 4):.4f}" + '\n')
            fout.write(f"Valied Rate for {dataset}: {np.round((len(val_sets[dataset])-(num_failed))/len(val_sets[dataset]), 4):.4f}" + '\n')

        with open(os.path.join(expdir, f"hyp.wrd.trn"), "w") as fout:
            for line in hyps:
                fout.write(line + '\n')
        with open(os.path.join(expdir, f"ref.wrd.trn"), "w") as fout:
            for line in refs:
                fout.write(line + '\n')
