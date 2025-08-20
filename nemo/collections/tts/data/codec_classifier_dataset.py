
class AudioCodesRealFakeDataset(Dataset):
    def __init__(self, manifest_real=None, manifest_fake=None, max_samples=None, dataset_type='train'):
        super().__init__()
        # The codes are very compressed so for a reasonable number of samples, the memory footprint is manageable and we can just load them all into memory
        print(f"Loading real codes from manifest: {manifest_real}")
        self.codes_real = self.load_manifest_codes(manifest_real, max_samples=max_samples) # total_frames, C
        print(f"Loading fake codes from manifest: {manifest_fake}")
        self.codes_fake = self.load_manifest_codes(manifest_fake, max_samples=max_samples) # total_frames, C
        if dataset_type == 'train':
            self.reshuffle = True
        else:
            self.reshuffle = False

    
    def __len__(self):
        return min(len(self.codes_real), len(self.codes_fake))

    def get_sampler(self, batch_size: int, world_size: int) -> Optional[torch.utils.data.Sampler]:
        return None
    def load_manifest_codes(self, manifest_path, max_samples=None):
        records = read_manifest(manifest_path)
        codes = []
        for record in tqdm(records):
            codes.append(self.load_codes(record['target_audio_codes_path']))
            if max_samples is not None and len(codes) >= max_samples:
                break
        # stack codes
        codes = torch.cat(codes, dim=0) # total_frames, C
        return codes
    
    def load_codes(self, codes_path):
        codes = torch.load(codes_path).long() # C, T
        return codes.T # T, C
    
    def __getitem__(self, index):
        # get a random index, separate per subset so that we get a different real/fake pair each time
        if self.reshuffle:
            # Note: we are ignoring the index! 
            # not good for validation set where we might want to disable shuffling
            real_index = torch.randint(0, len(self.codes_real), (1,))
            fake_index = torch.randint(0, len(self.codes_fake), (1,))
            real_codes = self.codes_real[real_index]
            fake_codes = self.codes_fake[fake_index]
        else:
            real_index = index
            fake_index = index
            real_codes = self.codes_real[real_index:real_index+1]
            fake_codes = self.codes_fake[fake_index:fake_index+1]
        
        # Concat real and fake and label them
        codes = torch.cat([real_codes, fake_codes], dim=0)
        labels = torch.Tensor((1,0)) # 1=real, 0=fake
        return {
            "audio_codes": codes,
            "audio_codes_lens": torch.Tensor((codes.shape[1],codes.shape[1])), # one entry for real, one for fake;lengths are fixed at n_codebooks for now; 
            "labels": labels
        }
    
    def collate_fn(self, batch: List[dict]):
        audio_codes_list = []
        audio_codes_lens_list = []
        labels_list = []
        for example in batch:
            audio_codes_list.append(example["audio_codes"])
            audio_codes_lens_list.append(example["audio_codes_lens"])
            labels_list.append(example["labels"])
            
        audio_codes = torch.cat(audio_codes_list, dim=0)
        audio_codes_lens = torch.cat(audio_codes_lens_list, dim=0).int()
        labels = torch.cat(labels_list, dim=0)

        # shuffle batch to avoid real/fake always being consecutive (shouldn't really be necessary, but just to be safe)
        perm = torch.randperm(audio_codes.shape[0])
        audio_codes = audio_codes[perm]
        audio_codes_lens = audio_codes_lens[perm]
        labels = labels[perm]

        batch_dict = {
            "audio_codes": audio_codes,
            "audio_codes_lens": audio_codes_lens,
            "labels": labels
        }

        return batch_dict



if __name__ == "__main__":
    test_manifest = '/datap/misc/speechllm_codecdatasets/manifests/t5_exp/libri100__for_classifer.json'
    dataset = AudioCodesRealFakeDataset(manifest_real=test_manifest, manifest_fake=test_manifest, max_samples=30)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
    for i, batch in enumerate(dataloader):
        print(batch['audio_codes'].shape)
        print(batch['audio_codes_lens'])
        print(batch['labels'])
        if i > 10:
            break