import torch
from trainer_speechLLM_llama_v3_1b_noquant_noFeatExt import SpeechLLM
import torch.utils.data as data_utils
from dataset_biasingLLM_llama_noFeatExt import InstructionalBiasingASRDataset, MyCollator_biasingLLM
from jiwer import wer
import numpy as np
import re
import os
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from whisper_normalizer.english import EnglishTextNormalizer
import argparse

# --mode streaming --dataset dev_clean --use_context

pattern = "[!\"#$%&'()*+,\-/:;<=>?@[\]^_`{|}~]"

def streaming_inference_with_features(model, feat_path, tokenizer, gen_config, device, segment_length=100.0, overlap_ratio=0.5, use_context=True):
    """使用預處理好的特徵文件進行流式推理，支持幀重疊和上下文傳遞"""
    # 記錄開始時間
    start_time = time.time()
    
    print("\n" + "="*80)
    print(f"開始處理特徵文件: {feat_path}")
    print(f"使用設備: {device}, 段落長度: {segment_length} 幀")
    print(f"重疊比例: {overlap_ratio*100}%, 使用上下文: {use_context}")
    print("="*80 + "\n")

    # 加載特徵
    load_start = time.time()
    print(f"正在加載特徵文件...")
    features = torch.load(feat_path, map_location='cpu')  # 先加載到CPU
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
        print(f"  已將NumPy數組轉換為張量")
    elif isinstance(features, torch.Tensor):
        features = features.float()
        print(f"  已將張量轉換為float類型")
    
    features = features.squeeze().to(device)  # 明確移動到指定設備
    print(f"  特徵加載完成: 形狀={features.shape}, 類型={features.dtype}")
    print(f"  特徵統計: 最小值={features.min().item():.4f}, 最大值={features.max().item():.4f}, 平均值={features.mean().item():.4f}")
    print(f"  特徵加載耗時: {time.time() - load_start:.2f}秒")
    print("-"*80)
    
    # 計算每個段落的大小和重疊
    total_frames = features.shape[0]
    frames_per_segment = int(segment_length)
    overlap_frames = int(frames_per_segment * overlap_ratio)
    effective_step = frames_per_segment - overlap_frames
    
    # 構建基礎提示詞
    prompt_start = time.time()
    print(f"構建提示詞...")
    instruction_phrase = f'Transcribe the input audio'
    base_pre_speech_prompt = f"Instruction:\n{instruction_phrase}." 
    base_pre_speech_prompt += "\n\nInput:\n<speech>"
    post_speech_prompt = f"</speech>\n\n" + "Output:\n" 
    
    # 空的biasing list
    blist = ['<unbiased>']
    blist_tokenized = tokenizer(blist, padding="longest", return_tensors='pt', 
                              truncation=False, add_special_tokens=False)
    blist_tokenized = {k: v.to(device) for k, v in blist_tokenized.items()}
    
    print(f"  提示詞構建完成:")
    print(f"  基礎前置提示詞: '{base_pre_speech_prompt}'")
    print(f"  後置提示詞: '{post_speech_prompt}'")
    print(f"  偏置列表: {blist}")
    print(f"  提示詞構建耗時: {time.time() - prompt_start:.2f}秒")
    print("-"*80)
    
    # 流式處理每個段落
    print(f"開始流式處理...")
    accumulated_text = ""
    previous_context = ""
    
    # 計算實際段數（考慮重疊）
    if effective_step > 0:
        num_segments = (total_frames - overlap_frames) // effective_step
        if (total_frames - overlap_frames) % effective_step > 0:
            num_segments += 1
    else:
        num_segments = 1  # 避免除零錯誤
    
    print(f"總特徵長度: {total_frames} 幀")
    print(f"分段大小: {frames_per_segment} 幀/段")
    print(f"重疊大小: {overlap_frames} 幀/段")
    print(f"有效步長: {effective_step} 幀")
    print(f"總段數: {num_segments} 段")
    print("-"*80)
    
    total_inference_time = 0
    audio_id = os.path.basename(feat_path)
    # 開始前重置CIF緩衝區
    model.cif_forward.reset_buffer(audio_id)
    
    for i in range(num_segments):
        seg_start_time = time.time()
        
        # 計算當前段落的開始和結束索引（考慮重疊）
        start_idx = i * effective_step
        end_idx = min(start_idx + frames_per_segment, total_frames)
        
        # 檢查是否需要調整開始索引（確保不超出範圍）
        if end_idx - start_idx < frames_per_segment and start_idx > 0:
            start_idx = max(0, end_idx - frames_per_segment)
        
        is_last = (i == num_segments - 1 or end_idx == total_frames)
        segment_id = i + 1
        
        print("\n" + "-"*50)
        print(f"處理段落 {segment_id}/{num_segments}")
        print(f"  幀範圍: {start_idx}-{end_idx} ({end_idx-start_idx} 幀)")
        print(f"  是否為最後段: {is_last}")
        
        # 提取當前段落特徵
        segment_features = features[start_idx:end_idx].unsqueeze(0).to(device)
        print("正在處理的特徵範圍:", start_idx, end_idx)   
        print(f"  段落特徵形狀: {segment_features.shape}")
        
        # 根據是否使用上下文構建提示詞
        current_pre_speech_prompt = pre_speech_prompt
        if use_context and previous_context:
            context_prefix = f"{previous_context}\n"
            current_pre_speech_prompt = context_prefix + pre_speech_prompt
            print(f"  使用上下文: '{previous_context}'")
        
        # 將當前提示詞轉換為token IDs
        pre_tokenized_ids = tokenizer(current_pre_speech_prompt, padding="longest", # Instruction:\n ... \n\nInput:\n<speech> 
                                   return_tensors='pt', truncation=False, 
                                   add_special_tokens=False)["input_ids"].to(device)
        post_tokenized_ids = tokenizer(post_speech_prompt, padding="longest", # ...</speech>\n\n" + "Output:\n
                                    return_tensors='pt', truncation=False, 
                                    add_special_tokens=False)["input_ids"].to(device)
        
        # 模擬batch結構 - 先構建batch再傳給generate
        batch = ([f"segment_{segment_id}"], segment_features, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, None)
        
        # 生成文本
        print(f"  開始生成文本...")
        inference_start = time.time()
        with torch.no_grad():
            generated_output_text = model.generate(
                batch, 
                gen_config, 
                device, 
                audio_id=audio_id,  # 傳遞音頻ID
                is_final_segment=is_last  # 傳遞是否為最後段
            )
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        
        print("  生成結果:", generated_output_text[0])
        print(f"  生成耗時: {inference_time:.2f}秒")
            
        # 提取預測文本
        extract_start = time.time()
        extracted_pred = model.extract_prediction_values(generated_output_text[0])
        
        if 'transcript' in extracted_pred:
            segment_text = extracted_pred['transcript']
            print("提取的文本:", segment_text)
            
            # 更新上下文（用於下一段）
            if use_context:
                previous_context += " " + segment_text
            # 添加到累積文本
            if i == 0:
                accumulated_text = segment_text
            else:
                accumulated_text += " | " + segment_text
        else:
            print(f"  警告: 無法從生成結果中提取轉錄文本")

        print(f"  提取耗時: {time.time() - extract_start:.2f}秒")
        print(f"  總段落處理耗時: {time.time() - seg_start_time:.2f}秒")
        print("-"*50)

    # 處理完後清理緩衝區
    model.cif_forward.clear_audio_buffer(audio_id)
    
    # 清理並返回完整文本
    final_text = accumulated_text.strip()
    
    print("\n" + "="*80)
    print(f"處理完成")
    print(f"  總處理時間: {time.time() - start_time:.2f}秒")
    print(f"  純推理時間: {total_inference_time:.2f}秒")
    print(f"  平均每段推理時間: {total_inference_time/num_segments:.2f}秒")
    print("="*80 + "\n")
    
    return final_text

if __name__ == "__main__":
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="流式語音辨識推理")
    parser.add_argument("--mode", type=str, default="dataset", choices=["dataset", "streaming"],
                      help="推理模式: dataset (使用數據集), streaming (流式處理)")
    parser.add_argument("--dataset", type=str, default="dev_clean",
                      choices=["dev_clean", "dev_other", "test_clean", "test_other"],
                      help="要評估的數據集")
    parser.add_argument("--ckpt_path", type=str, 
                      default='./exp4/streamingLLM_llama1b_CIFwithQuantityLoss_prompt/libri100_speechLLM_llama1b/epoch=10.ckpt',
                      help="模型檢查點路徑")
    parser.add_argument("--segment_length", type=float, default=50.0,
                      help="流式處理時的段落長度（特徵幀數）")
    parser.add_argument("--overlap_ratio", type=float, default=0.1,
                      help="相鄰段落的重疊比例（0-1之間）")
    parser.add_argument("--use_context", action="store_true",
                      help="是否使用先前生成的文字作為上下文")
    parser.add_argument("--beam_size", type=int, default=4,
                      help="波束搜索大小，較小的值可加快推理速度")
    args = parser.parse_args()
    
    # 生成配置
    gen_config = {
        'num_beams': args.beam_size,
        'do_sample': True,  # 改為True避免警告
        'min_length': 1,
        'max_length': 64,
        'temperature': 0.0,
        'top_p': 1.0,
        'repetition_penalty': 1.5,
        'no_repeat_ngram_size': 4,
        'length_penalty': 1.0,
    }
    
    # 加載模型
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')

    model = SpeechLLM(**ckpt['hyper_parameters'])
    model_dict = model.state_dict()
    # 跳過whisper, "audio_encoder"
    trained_model_dict = ckpt['state_dict']
    load_dict = {}
    for k in model_dict:
        if k in trained_model_dict:
            load_dict[k] = trained_model_dict[k]
        else:
            print(f'{k} not in load into dict') # 用於調試，k預期是llm-layers
    model.load_state_dict(load_dict, strict=False)
    model.cuda()
    model.eval()
    tokenizer = model.llm_tokenizer
    my_collator = MyCollator_biasingLLM(tokenizer)
    
    # 設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加載數據集
    test_mode = 'test_utt'
    datasets = {
        'dev_clean': './data/LibriSpeech/dev_clean_full.json',
        'dev_other': './data/LibriSpeech/dev_other_full.json',
        'test_clean': './data/LibriSpeech/test_clean_full.json',
        'test_other': './data/LibriSpeech/test_other_full.json'
    }
    
    # 根據模式選擇不同的推理方法
    if args.mode == "streaming":
        # 使用指定的數據集
        dataset_path = datasets[args.dataset]
        dataset = InstructionalBiasingASRDataset(json_file=dataset_path, mode=test_mode)
        
        # 流式處理數據集中的音頻
        en_norm = EnglishTextNormalizer()
        hyps, refs, tot_we, num_failed = [], [], 0.0, 0.0
        
        print(f"Streaming evaluation of dataset: {args.dataset}")
        
        for idx in tqdm(range(len(dataset))):
            uttname, audio_feat, bwords, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt = dataset[idx]
            
            # 獲取特徵文件路徑
            feat_path = dataset.data_frame[uttname]['feat']
            # 從output_prompt中提取目標文本
            target = dataset.data_frame[uttname]["words"].lower()
            
            # 流式處理特徵
            final_transcript = streaming_inference_with_features(
                model, 
                feat_path, 
                tokenizer, 
                gen_config, 
                device, 
                segment_length=args.segment_length,
                overlap_ratio=args.overlap_ratio,
                use_context=args.use_context
            )

            print("Final transcript:", final_transcript)
            print("Target transcript:", target)
            
            # 計算WER
            transcription = en_norm(final_transcript.lower())
            refwords = en_norm(target.lower())
            utt_wer = wer(refwords, transcription)
            tot_we += utt_wer
            
            utt_hyp = f"{transcription} ({uttname})"
            utt_ref = f"{refwords} ({uttname})"
            hyps.append(utt_hyp)
            refs.append(utt_ref)
            
            print(f"Utterance: {uttname}, WER: {utt_wer:.4f}")
        
        # 保存結果
        avg_wer = tot_we / len(dataset)
        print(f"Average WER for {args.dataset} (streaming): {avg_wer:.4f}")
        
        expdir = os.path.join(os.path.dirname(ckpt_path), f'streaming_{args.dataset}_overlap{int(args.overlap_ratio*100)}_context{1 if args.use_context else 0}')
        if not os.path.exists(expdir):
            os.makedirs(expdir)
            
        with open(os.path.join(expdir, "valied.log"), "w") as fout:
            fout.write(f"WER for {args.dataset} (streaming): {avg_wer:.4f}\n")
            fout.write(f"Segment length: {args.segment_length}, Overlap ratio: {args.overlap_ratio}, Use context: {args.use_context}\n")
            fout.write(f"Beam size: {args.beam_size}, Max tokens: {args.max_tokens}\n")
            
        with open(os.path.join(expdir, "hyp.wrd.trn"), "w") as fout:
            for line in hyps:
                fout.write(line + '\n')
                
        with open(os.path.join(expdir, "ref.wrd.trn"), "w") as fout:
            for line in refs:
                fout.write(line + '\n')
        
    else:  # dataset模式，維持原有邏輯
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