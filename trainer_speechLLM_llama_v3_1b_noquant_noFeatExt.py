import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import wandb
import pytorch_lightning as pl
from jiwer import wer
import torchmetrics
import random
import re
import json

from models.biasing_module import get_biasing_module
from models.encoder_audio import get_audio_encoder, TransformerAudioEnoder
from models.buffered_cif_module import get_buffered_cif_module
from models.connector import get_connector
from models.llm_llama8b import get_llm
from models.encoder_gop import get_gop_encoder

import numpy as np
import bitsandbytes as bnb
from transformers import StoppingCriteriaList, StoppingCriteria

from transformers import WhisperTokenizer

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if input_ids.size(-1) >= len(stop):
                if torch.all((stop == input_ids[0][-len(stop):])).item():
                    return True
        return False

class SpeechLLM(pl.LightningModule):
    def __init__(self, 
                 audio_dim=1280,
                 llm_mode="test",
                 hidden_dim=512,
                 use_quantity_loss=True,  # 是否使用 quantity loss
                 lambda2=1.0,  # quantity loss 權重
                 gop_encoder_name="conv-branchformer",
                 audio_encoder_name="whisper_large_v2",
                 biasing_module_name="transformer",
                 biasing_heads=6,
                 biasing_depths=2,
                 llm_dim=2048,
                 connector_name='linear-pool',
                 connector_k=5,
                 llm_name="../llama3.2/huggingface/Llama-3.2-1B", 
                 finetune_encoder=False,
                 use_lora=True,
                 lora_r=32,
                 lora_alpha=2,
                 max_lr=3e-4,
                 total_training_step=500000,
                 train_batch_per_epoch=1000,
                 warmup_steps=1000,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.llm_dim = llm_dim
        self.llm_name = llm_name
        self.finetune_encoder = finetune_encoder
        self.use_lora = use_lora

        #self.audio_proc, self.audio_encoder = get_audio_encoder(audio_encoder_name, finetune_encoder)
        self.audio_adapter = get_gop_encoder('conv-branchformer', finetune_encoder,
         audio_dim, 4, 2, hidden_dim)
        self.cif_forward = get_buffered_cif_module(
            input_dim=hidden_dim,
            cif_threshold=1.0,
            cif_conv_channels=256,
            cif_conv_kernel_size=5,
            cif_padding=2
        )
        self.connector = get_connector(connector_name, hidden_dim, llm_dim, connector_k)
        self.llm_tokenizer, self.llm_model = get_llm(llm_name, use_lora, lora_r, lora_alpha)
        
        self.max_lr = max_lr
        self.total_training_step = total_training_step
        self.warmup_steps = warmup_steps
        self.use_embedding_loss = False
        self.num_validation_samples = 1000
        self.max_ep = total_training_step//train_batch_per_epoch

    def configure_optimizers(self):
        opt = [
            #{"params": self.audio_encoder.parameters(), "lr": 1e-5},
            {"params": self.audio_adapter.parameters(), "lr": self.max_lr},
            {"params": self.connector.parameters(), "lr": self.max_lr},
            {"params": self.llm_model.parameters(), "lr": self.max_lr},
        ]
        #optimizer = Adam(opt, lr=self.max_lr)
        optimizer = AdamW(opt, lr=self.max_lr, weight_decay=0.05, betas=(0.9,0.999))
        #optimizer = AdamW(opt, lr=self.max_lr, weight_decay=0.1, betas=(0.9,0.999))
        #optimizer = bnb.optim.Adam8bit(opt, lr=self.max_lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_ep, eta_min=0, last_epoch=-1, verbose=True)
        return [optimizer], [lr_scheduler]

    def encode(self, audio_feat, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, return_embedding_loss=False):
        batch_size = audio_feat.shape[0]
        #audio_embeds = self.audio_encoder(fbank)
        audio_embeds = self.audio_adapter(audio_feat)
        audio_embeds, alphas = self.cif_forward(audio_embeds) # CIF module
        audio_embeds = self.connector(audio_embeds)

        #embedder = self.llm_model.model.model.embed_tokens
        embedder = self.llm_model.base_model.model.model.embed_tokens 

        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)
        output_prompt_embeds = embedder(output_tokenized_ids)

        combined_embeds = torch.cat([pre_prompt_embeds, audio_embeds, post_prompt_embeds, output_prompt_embeds], dim=1)
        atts = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        input_token_length = pre_tokenized_ids.shape[1] + audio_embeds.shape[1] + post_tokenized_ids.shape[1]
        label_ids = torch.cat([
            torch.ones([batch_size, input_token_length], device=combined_embeds.device)*-100,
            output_tokenized_ids
        ], 1).to(combined_embeds.device).to(torch.int64)

        return combined_embeds, atts, label_ids

    def forward(self, embeds, atts, label_ids):
        out = self.llm_model(
            inputs_embeds=embeds,
            attention_mask=atts,
            labels=label_ids,
        )
        return out
    
    def generate(self, batch, gen_dict_config, device, audio_id=None, is_final_segment=False):
        # 如果是新音檔，清空所有緩衝區
        if audio_id is not None and not is_final_segment:
            # 檢查是否是段落的起始（新音檔開始）
            if self.cif_forward.buffers.get(audio_id, None) is None:
                # 新音檔ID，清空所有緩衝區
                self.cif_forward.reset_buffer()
                print(f"Reset buffer for new audio: {audio_id}")

        #batch obtained from collector
        uttnames, audio_feat, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        batch_size = audio_feat.shape[0]
        # encode speech
        audio_embeds = self.audio_adapter(audio_feat)
        audio_embeds, alphas = self.cif_forward(audio_embeds, audio_id=audio_id, is_final_segment=is_final_segment)
        audio_embeds = self.connector(audio_embeds)
        
        # encode prompt
        embedder = self.llm_model.base_model.model.model.embed_tokens
        pre_prompt_embeds = embedder(pre_tokenized_ids.to(device))
        post_prompt_embeds = embedder(post_tokenized_ids.to(device))

        combined_embeds = torch.cat([pre_prompt_embeds, audio_embeds, post_prompt_embeds], dim=1)
        atts = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        # prepare begin of sentence
        bos = f'{self.llm_tokenizer.bos_token} '
        bos = self.llm_tokenizer(bos, return_tensors='pt')
        bos_emb = embedder(bos['input_ids'].to(device))
        bos_att = bos['attention_mask'].to(device)

        # combined_embeds = torch.cat([combined_embeds, bos_emb], dim=1)
        combined_embeds = torch.cat([combined_embeds], dim=1)
        # atts = torch.cat([atts, bos_att], dim=1)
        atts = torch.cat([atts], dim=1)

        stop_words = [f' {self.llm_tokenizer.eos_token}', f'{self.llm_tokenizer.eos_token}']
        stop_words_ids = [self.llm_tokenizer(stops, return_tensors='pt')['input_ids'].squeeze().to(device)
        for stops in stop_words]
        #stop_words_ids = [stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        # generate
        output = self.llm_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=atts,
            max_new_tokens=gen_dict_config["max_new_tokens"],
            stopping_criteria=stopping_criteria,
            num_beams=gen_dict_config["num_beams"],
            do_sample=gen_dict_config["do_sample"],
            min_length=gen_dict_config["min_length"],
            # temperature=gen_dict_config["temperature"],
            top_p=gen_dict_config["top_p"],
            repetition_penalty=gen_dict_config["repetition_penalty"],
            length_penalty=gen_dict_config["length_penalty"],
            pad_token_id=self.llm_tokenizer.eos_token_id,
            forced_eos_token_id = self.llm_tokenizer.convert_tokens_to_ids('<|end_of_text|>'),
        )
        # output_text = self.llm_tokenizer.batch_decode(torch.cat([bos['input_ids'].to(device), output], dim=-1))
        output_text = self.llm_tokenizer.batch_decode(output, dim=-1)
        return output_text

    def training_step(self, batch, batch_idx):
        uttnames, audio_feat, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(audio_feat, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
        outputs = self.forward(embeds, atts, label_ids)
        loss = outputs["loss"]

        # 只保留quantity loss
        if self.hparams.use_quantity_loss:
            target_length = torch.sum(label_ids != -100, dim=1).float()
            pred_length = torch.sum(atts, dim=1).float()
            quantity_loss = F.mse_loss(pred_length, target_length)
            loss = loss + self.hparams.lambda2 * quantity_loss
            self.log("train/quantity_loss", quantity_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
            wandb.log({"train/quantity_loss": quantity_loss})

        self.log("train/total_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        uttnames, audio_feat, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch

        embeds, atts, label_ids = self.encode(audio_feat, blist_tokenized, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
        outputs = self.forward(embeds, atts, label_ids)
        loss = outputs["loss"]
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])

        # Add validation階段的quantity loss監控
        if self.hparams.use_quantity_loss:
            target_length = torch.sum(label_ids != -100, dim=1).float()
            pred_length = torch.sum(atts, dim=1).float()
            quantity_loss = F.mse_loss(pred_length, target_length)
            self.log("val/quantity_loss", quantity_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
            
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1).cpu()

        generated_output_text = self.llm_tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
        target_text = self.llm_tokenizer.decode(output_tokenized_ids[0], skip_special_tokens=False)

        extracted_pred = self.extract_prediction_values(generated_output_text)
        extracted_target = self.extract_prediction_values(target_text)

        keys = extracted_target.keys()
        pred_keys = extracted_pred.keys()

        if 'transcript' in pred_keys:
            target_ref_txt = extracted_target['transcript']
            predicted_ref_txt = extracted_pred['transcript']
            wer_metric = wer(target_ref_txt.lower(), predicted_ref_txt.lower())
            self.log("val/ref_text_wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])


        if batch_idx in self.selected_samples_for_logging:
            sample_idx = self.selected_samples_for_logging.index(batch_idx)
            # Use wandb.log to log prediction and truth texts
            # wandb.log({
            #    f"val_sample_{sample_idx}_gen": wandb.Html(f"<pre>{str(generated_output_text)}</pre>"), 
            #    f"val_sample_{sample_idx}_pred": wandb.Html(f"<pre>{str(extracted_pred)}</pre>"), 
            #    f"val_sample_{sample_idx}_target": wandb.Html(f"<pre>{str(target_text).replace('<s>', '').replace('</s>', '')}</pre>"),
            # }, commit=False)

            print(f"val_sample_{sample_idx}_gen: {generated_output_text} ")
            print(f"val_sample_{sample_idx}_pred: {str(extracted_pred)} ")
            print(f"val_sample_{sample_idx}_target: {str(target_text)} ")

            return {"val_loss": loss}
    
    def on_validation_epoch_start(self):
        """Select two random validation samples to log for each epoch."""
        self.selected_samples_for_logging = random.sample(range(self.num_validation_samples), 1)
   
    def extract_dictionary(self, input_string):
        #pattern = r'<s>\s*(\{.*?\})\s*</s>'
        #pattern = r'<im_start>\s.*(\{.*?\})\s*<im_end>'
        pattern = r'<\|begin_of_text\|>\s.*(\{.*?\})\s*<\|end_of_text\|>'
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            dict_string = match.group(1)
            dict_string = re.sub(r',\s*}', '}', dict_string)
            try:
                return json.loads(dict_string)
            except json.JSONDecodeError as e:
                return {}
        else:
            return {}
    
    def extract_prediction_values(self, input_string):
    # 嘗試使用標準JSON格式提取
        json_str_match = re.search(r'<\|begin_of_text\|>\s*(.*?)\s*(?:<\|end_of_text\|>|$)', input_string, re.DOTALL)
        
        try:
            if json_str_match:
                content = json_str_match.group(1).strip()
                
                # 嘗試解析JSON
                if content.startswith('{') and content.endswith('}'):
                    return json.loads(content)
                
                # 如果無法解析JSON，創建一個transcript字段
                return {"transcript": content}
            else:
                # 如果連標準格式都找不到，直接清理並返回文本
                cleaned_text = input_string.replace('<|begin_of_text|>', '').replace('<|end_of_text|>', '').strip()
                if cleaned_text:
                    return {"transcript": cleaned_text}
                return {}
        except json.JSONDecodeError:
            # JSON解析失敗，直接使用文本內容
            cleaned_text = json_str_match.group(1).strip() if json_str_match else ""
            if cleaned_text:
                return {"transcript": cleaned_text}
            return {}
        
