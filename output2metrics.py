from tts.metric.utils import calc_cer, calc_wer
import json


with open('output.json') as file:
    j = json.load(file)
wer_argmax = []
wer_bs = []
wer_lm = []
cer_argmax = []
cer_bs = []
cer_lm = []
for el in j:
    gt = el["ground_truth"]
    bs = el['pred_text_beam_search'][0][0]
    am = el['pred_text_argmax']
    lm = el['pred_text_beam_search_lm'][0][0]
    wer_argmax.append(calc_wer(gt, am))
    wer_bs.append(calc_wer(gt, bs))
    wer_lm.append(calc_wer(gt, lm))
    cer_argmax.append(calc_cer(gt, am))
    cer_bs.append(calc_cer(gt, bs))
    cer_lm.append(calc_cer(gt, lm))

def mean(arr):
    return sum(arr) / (1e-9 + len(arr))

print(f"WER (argmax):\t\t\t{mean(wer_argmax)*100:.2f}%")
print(f"WER (beam search):\t\t\t{mean(wer_bs)*100:.2f}%")
print(f"WER (beam search + LM):\t\t{mean(wer_lm)*100:.2f}%")
print(f"CER (argmax):\t\t{mean(cer_argmax)*100:.2f}%")
print(f"CER (beam search):\t\t\t{mean(cer_bs)*100:.2f}%")
print(f"CER (beam search + LM):\t\t{mean(cer_lm)*100:.2f}%")