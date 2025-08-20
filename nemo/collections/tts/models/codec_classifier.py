
class T5TTS_Discriminator(ModelPT):
    # A model that classifies whether frames of audio codes are real or fake
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        self.cfg = cfg

        # CLS: special token that we will use to output the real/fake classification
        # 5 because in T5TTS the last 4 are already reserved for BOS, EOS, BOS_CONTEXT, EOS_CONTEXT
        self.audio_cls_id = cfg.num_audio_tokens_per_codebook - 5 # TODO: make this cleaner?

        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices    
        self._tb_logger = None

        super().__init__(cfg=cfg, trainer=trainer)

        # create and initialize the audio embeddings from pretrained T5TTS model; for now we will not freeze them
        # but it's worth experimenting with this
        audio_embeddings_pretrained = torch.load("audio_embeddings.pt", weights_only=False)
        freeze_audio_embeddings = self.cfg.get('freeze_audio_embeddings', False)
        self.audio_embeddings = self.create_audio_embeddings(self.cfg, audio_embeddings_pretrained, add_cls_token=True, freeze_audio_embeddings=freeze_audio_embeddings)
        
        d_audio_embeddings = self.audio_embeddings[0].weight.shape[1]
        d_model = self.cfg.encoder.d_model

        # Projection from audio codebook space to transformer dimensions
        self.audio_emb_to_model_proj = nn.Linear(d_audio_embeddings, d_model)

        # encoder-only transformer
        self.encoder = t5tts_transformer.Transformer(**self.cfg.encoder)                                                     

        # Project the encoder output to a single logit for real/fake classification
        self.final_proj = nn.Linear(d_model, 1)

        # Binary cross entropy loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        return state_dict

    def create_audio_embeddings(self, cfg, pretrained_audio_embeddings=None, add_cls_token=False, freeze_audio_embeddings=False):
        audio_embeddings = []
        if add_cls_token:
            # This is somewhat wasteful as we only need one embedding for the CLS token (can optimize if needed)
            audio_embeddings.append(nn.Embedding(cfg.num_audio_tokens_per_codebook, cfg.embedding_dim))
        for idx in range(cfg.num_audio_codebooks):
            audio_embeddings.append(nn.Embedding(cfg.num_audio_tokens_per_codebook, cfg.embedding_dim))
            if pretrained_audio_embeddings is not None:
                with torch.no_grad():
                    audio_embeddings[idx].weight.copy_(pretrained_audio_embeddings[idx].weight)
                    if freeze_audio_embeddings:
                        audio_embeddings[idx].weight.requires_grad = False
        return nn.ModuleList(audio_embeddings)
            
    
    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    tb_logger = logger.experiment
                    break
            self._tb_logger = tb_logger
        return self._tb_logger
        
    def training_step(self, batch, batch_idx):
        outputs = self.process_batch(batch)
        self.log('train_loss', outputs['loss'], prog_bar=True, sync_dist=True)
        return outputs['loss']
    
    def embed_audio_codes(self, audio_codes):
        # audio_codes: (B, C)
        # Unlike the T5TTS model, we don't average the embeddings across the codebooks
        audio_embedding_list = None
        for c in range(audio_codes.size(1)):
            embedding = self.audio_embeddings[c](audio_codes[:, c])
            if audio_embedding_list is None:
                audio_embedding_list = [embedding]
            else:
                audio_embedding_list.append(embedding)
        audio_embedding = torch.stack(audio_embedding_list, dim=1)        
        return audio_embedding # (B, C, E)

    def infer_batch(self, codes):
        B, C = codes.shape
        # Prepend with CLS token
        codes = torch.cat([torch.ones_like(codes[:, 0:1], device=self.device) * self.audio_cls_id, codes], dim=1)
        codes_lens = torch.ones(B, device=self.device, dtype=torch.long) * (C + 1) # +1 for the CLS token
        codes_embedded = self.embed_audio_codes(codes)
        codes_projected = self.audio_emb_to_model_proj(codes_embedded)
        mask = get_mask_from_lengths(codes_lens)
        logits, attn_info, dec_out = self.forward(codes_projected, mask)
        cls_logits = logits[:, 0].squeeze(1) # B
        preds_raw = cls_logits # torch.sigmoid(cls_logits)
        preds_post_sigmoid = torch.sigmoid(cls_logits)
        preds = preds_post_sigmoid > 0.5
        return preds, preds_raw, preds_post_sigmoid
    
    def process_batch(self, batch, mode="train"):
        audio_codes = batch['audio_codes'] # B, C
        audio_codes_lens = batch['audio_codes_lens'] # B
        labels = batch['labels'] if 'labels' in batch else None
        if labels is None:
            print("Warning: No labels provided for discriminator training. Setting all labels to 1 (real).")
            # set all labels to 1 (real) just for debugging
            labels = torch.ones_like(audio_codes, device=self.device)

        # Prepend with CLS token
        audio_codes = torch.cat([torch.ones_like(audio_codes[:, 0:1], device=self.device) * self.audio_cls_id, audio_codes], dim=1)
        audio_codes_lens += 1

        # Embed the audio codes
        audio_codes_embedded = self.embed_audio_codes(audio_codes) # B, C, E
        audio_codes_mask = get_mask_from_lengths(audio_codes_lens)

        # Project to transformer dimension
        audio_codes_projected = self.audio_emb_to_model_proj(audio_codes_embedded) # B, C, E'
       
        # Run the encoder
        logits, attn_info, dec_out = self.forward(
            dec_input_embedded=audio_codes_projected,
            dec_input_mask=audio_codes_mask,
        ) # logits: B, C, 1

        # Compute loss only on CLS token
        cls_logits = logits[:, 0].squeeze(1) # B
        loss = self.bce_loss(cls_logits, labels)



        return {
            'logits': logits,
            'loss': loss,
        }
    
    def forward(self, dec_input_embedded, dec_input_mask):
        encoder_out = self.encoder(
            dec_input_embedded,
            dec_input_mask,
        )
        attn_probabilities = encoder_out['attn_probabilities']
        all_code_logits = self.final_proj(encoder_out['output']) # (B, )
        return all_code_logits, attn_probabilities, encoder_out['output']    

    def validation_step(self, batch, batch_idx):
        outputs = self.process_batch(batch)

        # Compute accuracy
        cls_logits = outputs['logits'][:, 0].squeeze(1) # B
        preds = torch.sigmoid(cls_logits) > 0.5
        val_acc = (preds == batch['labels']).float().mean()
        
        val_loss = outputs['loss']
        
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
    
    def on_validation_epoch_end(self):
        def collect(key):
            values = []
            for x in self.validation_step_outputs:
                if x[key] is not None:
                    values.append(x[key])
                else:
                    values.append(torch.tensor(0.0, device=self.device))
            stacked_values = torch.stack(values)
            return stacked_values.mean()

        val_loss = collect("val_loss")
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        val_acc = collect("val_acc")
        self.log("val_acc", val_acc, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()


    def get_dataset(self, cfg, dataset_type):
        dataset_config = copy.deepcopy(cfg.dataset)
        dataset_config.dataset_type = dataset_type
        dataset = instantiate(
            dataset_config
        )
        return dataset

    def _setup_train_dataloader(self, cfg):
        dataset = self.get_dataset(cfg, dataset_type='train')
        sampler = dataset.get_sampler(cfg.dataloader_params.batch_size, world_size=self.trainer.world_size)
        persistent_workers = True
        if cfg.dataloader_params.num_workers == 0:
            persistent_workers = False
            # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **cfg.dataloader_params, worker_init_fn=worker_init_fn_simple, persistent_workers=persistent_workers
        )
        return data_loader

    def _setup_test_dataloader(self, cfg):
        dataset = self.get_dataset(cfg, dataset_type='test')
        persistent_workers = True
        if cfg.dataloader_params.num_workers == 0:
            persistent_workers = False
            data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params, worker_init_fn=worker_init_fn_simple, persistent_workers=persistent_workers)
        return data_loader

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_train_dataloader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(cfg)

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_test_dataloader(cfg)
