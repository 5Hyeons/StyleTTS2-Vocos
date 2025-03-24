# StyleTTS2-Vocos

StyleTTS2-Vocos is a modified version of StyleTTS2 that replaces the original HiFi-GAN decoder with the Vocos decoder. 
This project maintains the advantages of the original StyleTTS2 while providing improved inference speed and memory efficiency through the Vocos decoder.

## Acknowledgments

This project is built upon [StyleTTS2](https://github.com/yl4579/StyleTTS2.git) by [@yl4579](https://github.com/yl4579). I express my deepest gratitude to the original authors for their groundbreaking work.

## Key Changes

1. Vocos Decoder Integration
   - Replaced the original StyleTTS2's HiFi-GAN decoder with Vocos decoder
   - Significantly reduced and near-constant memory usage during inference
   - Faster inference speed with comparable quality - generation time increases only marginally with audio length, unlike proportional scaling in GAN-based models
   - For detailed implementation of the Vocos module, please refer to `Modules/vocos.py`
   - **Note**: While the Vocos decoder offers significant advantages in terms of memory usage and inference speed, the current implementation in `vocos.py` may not achieve the same audio quality as the original HiFi-GAN decoder. I believe this is due to the current implementation of the Vocos module, particularly in the source module and forward pass. I welcome contributions from the community to improve the quality while maintaining the efficiency benefits.

2. No-BERT Version
   - Provides a version without PL-BERT model
   - Optimized for languages not supported by multi-lingual PL-BERT model like Korean
   - Minimal performance degradation without PL-BERT
   
## Model Download

Pre-trained models can be downloaded from the following Hugging Face repository:
[https://huggingface.co/5Hyeons/StyleTTS2](https://huggingface.co/5Hyeons/StyleTTS2)

## Training

The training process consists of two stages, similar to the original StyleTTS2:

1. First stage training:
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```

2. Second stage training:
```bash
python train_second.py --config_path ./Configs/config.yml
```

For No-BERT version:
```bash
python train_second_no_bert.py --config_path ./Configs/config.yml
```

The model will be saved in the format "epoch_1st_%05d.pth" and "epoch_2nd_%05d.pth". Checkpoints and Tensorboard logs will be saved at `log_dir`.

### Key Differences from Original StyleTTS2

The configuration file has two main differences from the original StyleTTS2:

1. Decoder Type:
   ```yaml
   decoder:
     type: 'vocos'  # Changed from original 'hifigan' or 'istftnet'
   ```

2. Root Path:
   ```yaml
   data_params:
     root_path: "/data/LibriTTS"  # Update this path according to your dataset location
   ```

## Inference

Please refer to the Jupyter notebooks in the `Demo` folder for detailed inference examples:
- `Inference_stage1.ipynb`: For first stage inference
- `Inference_stage2.ipynb`: For second stage inference

## Contributing

I welcome contributions to improve the Vocos decoder implementation, particularly in the following areas:
- Source module implementation in `Modules/vocos.py`
- Forward pass optimization
- Quality improvements while maintaining the efficiency benefits

If you have ideas or implementations that could improve the audio quality while maintaining the efficiency benefits, please feel free to contribute.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
