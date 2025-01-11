import argparse
import json
import os
import pprint
import string

import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate_detail

LOCAL_EVALSETS = {
    'riva_challenging': {
        'manifest': '/datap/misc/Datasets/riva/riva_interspeech.json',
        'audio_dir': '/datap/misc/Datasets/riva'
    },
    'vctk': {
        'manifest': '/home/pneekhara/2023/SimpleT5NeMo/manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json',
        'audio_dir': '/datap/misc/Datasets/VCTK-Corpus'
    }
}


def find_sample_audios(audio_dir):
    file_list = []
    for f in os.listdir(audio_dir):
        if "predicted_audio" in f and f.endswith(".wav"):
            audio_number = int(f.split("_")[-1].split(".wav")[0])
            file_list.append((audio_number, os.path.join(audio_dir, f)))
    file_list.sort()
    file_list = [t[1] for t in file_list]
    return file_list
            
def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records

def process_text(input_text):
    # Convert text to lowercase
    lower_case_text = input_text.lower()
    
    # Remove commas from text
    no_comma_text = lower_case_text.replace(",", "")
    
    # Replace "-" with spaces
    no_dash_text = no_comma_text.replace("-", " ")
    
    # Replace double spaces with single space
    single_space_text = " ".join(no_dash_text.split())

    single_space_text = single_space_text.translate(str.maketrans('', '', string.punctuation))
    
    return single_space_text


def evaluate(manifest_path, audio_dir, generated_audio_dir, debug=False):
    audio_file_lists = find_sample_audios(generated_audio_dir)
    records = read_manifest(manifest_path)
    if not debug:
        assert len(audio_file_lists) == len(records)

    device = "cuda"
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    model_name="stt_en_conformer_transducer_large"
                )
    # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")
    asr_model = asr_model.to(device)
    asr_model.eval()

    speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large') 
    speaker_verification_model = speaker_verification_model.to(device)
    speaker_verification_model.eval()

    # HACK: work around massive label printout issue by removing titanet_small (and using titanet_large again)
    print("WARNING: *not* using a different model for the alternate SV since titanet_small has an issue of printing out its entire set of labels")
    speaker_verification_model_alternate = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large') 
    #speaker_verification_model_alternate = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_small') 
    speaker_verification_model_alternate = speaker_verification_model_alternate.to(device)
    speaker_verification_model_alternate.eval()

    filewise_metrics = []
    pred_texts = []
    gt_texts = []
    for ridx, record in enumerate(records):
        if debug and ridx >= len(audio_file_lists):
            break
        gt_audio_filepath = record['audio_filepath']
        context_audio_filepath = record.get('context_audio_filepath', None)
        if audio_dir is not None:
            gt_audio_filepath = os.path.join(audio_dir, gt_audio_filepath)
            if context_audio_filepath is not None:
                context_audio_filepath = os.path.join(audio_dir, context_audio_filepath)
        
        pred_audio_filepath = audio_file_lists[ridx]
        with torch.no_grad():
            pred_text = asr_model.transcribe([pred_audio_filepath])[0][0]
            pred_text = process_text(pred_text)

        if 'normalized_text' in record:
            gt_text = process_text(record['normalized_text'])
        else:
            gt_text = process_text(record['text'])
        
        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)

        print("{} GT Text:".format(ridx), gt_text)
        print("{} Pr Text:".format(ridx), pred_text)
        # Format cer and wer to 2 decimal places
        print("CER:", "{:.4f} | WER: {:.4f}".format(detailed_cer[0], detailed_wer[0]))
        pred_texts.append(pred_text)
        gt_texts.append(gt_text)

        pred_context_ssim = 0.0
        gt_context_ssim = 0.0
        with torch.no_grad():
            gt_speaker_embedding = speaker_verification_model.get_embedding(gt_audio_filepath).squeeze()
            pred_speaker_embedding = speaker_verification_model.get_embedding(pred_audio_filepath).squeeze()
            pred_gt_ssim = torch.nn.functional.cosine_similarity(gt_speaker_embedding, pred_speaker_embedding, dim=0).item()

            gt_speaker_embedding_alternate = speaker_verification_model_alternate.get_embedding(gt_audio_filepath).squeeze()
            pred_speaker_embedding_alternate = speaker_verification_model_alternate.get_embedding(pred_audio_filepath).squeeze()
            pred_gt_ssim_alternate = torch.nn.functional.cosine_similarity(gt_speaker_embedding_alternate, pred_speaker_embedding_alternate, dim=0).item()

            if context_audio_filepath is not None:
                context_speaker_embedding = speaker_verification_model.get_embedding(context_audio_filepath).squeeze()
                context_speaker_embedding_alternate = speaker_verification_model_alternate.get_embedding(context_audio_filepath).squeeze()

                pred_context_ssim = torch.nn.functional.cosine_similarity(pred_speaker_embedding, context_speaker_embedding, dim=0).item()
                gt_context_ssim = torch.nn.functional.cosine_similarity(gt_speaker_embedding, context_speaker_embedding, dim=0).item()

                pred_context_ssim_alternate = torch.nn.functional.cosine_similarity(pred_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0).item()
                gt_context_ssim_alternate = torch.nn.functional.cosine_similarity(gt_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0).item()
                
        
        filewise_metrics.append({
            'gt_text': gt_text,
            'pred_text': pred_text,
            'detailed_cer': detailed_cer,
            'detailed_wer': detailed_wer,
            'pred_gt_ssim': pred_gt_ssim,
            'pred_context_ssim': pred_context_ssim,
            'gt_context_ssim': gt_context_ssim,
            'pred_gt_ssim_alternate': pred_gt_ssim_alternate,
            'pred_context_ssim_alternate': pred_context_ssim_alternate,
            'gt_context_ssim_alternate': gt_context_ssim_alternate
        })
    
    avg_metrics = {}
    avg_metrics['cer_filewise_avg'] = sum([m['detailed_cer'][0] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['wer_filewise_avg'] = sum([m['detailed_wer'][0] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['cer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)[0]
    avg_metrics['wer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)[0]
    avg_metrics['ssim_pred_gt_avg'] = sum([m['pred_gt_ssim'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_pred_context_avg'] = sum([m['pred_context_ssim'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_gt_context_avg'] = sum([m['gt_context_ssim'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_pred_gt_avg_alternate'] = sum([m['pred_gt_ssim_alternate'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_pred_context_avg_alternate'] = sum([m['pred_context_ssim_alternate'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_gt_context_avg_alternate'] = sum([m['gt_context_ssim_alternate'] for m in filewise_metrics]) / len(filewise_metrics)

    pprint.pprint(avg_metrics)

    return avg_metrics

def main():
    # audio_dir="/datap/misc/Datasets/riva" \
    parser = argparse.ArgumentParser(description='Evaluate Generated Audio')
    parser.add_argument('--manifest_path', type=str, default=None)
    parser.add_argument('--audio_dir', type=str, default=None)
    parser.add_argument('--generated_audio_dir', type=str, default=None)
    parser.add_argument('--evalset', type=str, default=None)
    args = parser.parse_args()

    if args.evalset is not None:
        assert args.evalset in LOCAL_EVALSETS
        args.manifest_path = LOCAL_EVALSETS[args.evalset]['manifest']
        args.audio_dir = LOCAL_EVALSETS[args.evalset]['audio_dir']
    
    evaluate(args.manifest_path, args.audio_dir, args.generated_audio_dir)

    

if __name__ == "__main__":
    main()
