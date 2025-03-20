# Multilingual ASR Model

`aihub_multi_lingual_en_jp_ko_zh_epoch_00011.pth` is trained on the following AIHUB datasets:

1. [다국어 통·번역 낭독체 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71524)
   - Multilingual speech and translation dataset
   - Contains Korean-English, Korean-Japanese, and Korean-Spanish parallel data
   - Total duration: 4,107 hours
   - Includes audio files and transcriptions

2. [감성 및 발화스타일 동시 고려 음성합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71349)
   - Speech synthesis dataset considering emotion and speaking style
   - Contains various emotional expressions and speaking styles
   - High-quality audio recordings with corresponding transcriptions

## Model Details

- Model Name: `aihub_multi_lingual_en_jp_ko_zh_epoch_00011.pth`
- Training Epoch: 11
- Supported Languages: English, Japanese, Korean, Chinese

## Vocabulary and Symbols

The model uses the vocabulary and symbols defined in `StyleTTS2-Vocos/symbols.py`, which includes:
