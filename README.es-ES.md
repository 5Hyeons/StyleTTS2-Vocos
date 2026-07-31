# StyleTTS2-Vocos

StyleTTS2-Vocos es una versión modificada de StyleTTS2 que reemplaza el decodificador HiFi-GAN original por el decodificador Vocos. 
Este proyecto mantiene las ventajas del StyleTTS2 original mientras proporciona una velocidad de inferencia mejorada y mayor eficiencia de memoria a través del decodificador Vocos.

## Agradecimientos

Este proyecto está basado en [StyleTTS2](https://github.com/yl4579/StyleTTS2.git) de [@yl4579](https://github.com/yl4579). Expreso mi más profundo agradecimiento a los autores originales por su trabajo innovador.

## Cambios Clave

1. Integración del Decodificador Vocos
   - Se reemplazó el decodificador HiFi-GAN original de StyleTTS2 por el decodificador Vocos.
   - Reducción significativa y uso de memoria casi constante durante la inferencia.
   - Velocidad de inferencia más rápida con calidad comparable; el tiempo de generación aumenta solo marginalmente con la longitud del audio, a diferencia del escalado proporcional en los modelos basados en GAN.
   - Para una implementación detallada del módulo Vocos, consulte `Modules/vocos.py`.
   - **Nota**: Si bien el decodificador Vocos ofrece ventajas significativas en términos de uso de memoria y velocidad de inferencia, es posible que la implementación actual en `vocos.py` no alcance la misma calidad de audio que el decodificador HiFi-GAN original. Creo que esto se debe a la implementación actual del módulo Vocos, particularmente en el módulo de fuente y el pase hacia adelante (forward pass). Agradezco las contribuciones de la comunidad para mejorar la calidad manteniendo los beneficios de eficiencia.

2. Versión No-BERT
   - Proporciona una versión sin el modelo PL-BERT.
   - Optimizada para idiomas no soportados por el modelo PL-BERT multilingüe, como el coreano.
   - Degradación mínima del rendimiento sin PL-BERT.
   
## Descarga del Modelo

Los modelos pre-entrenados se pueden descargar desde el siguiente repositorio de Hugging Face:
[https://huggingface.co/5Hyeons/StyleTTS2](https://huggingface.co/5Hyeons/StyleTTS2)

## Entrenamiento

El proceso de entrenamiento consta de dos etapas, similar al StyleTTS2 original:

1. Entrenamiento de la primera etapa:
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```

2. Entrenamiento de la segunda etapa:
```bash
python train_second.py --config_path ./Configs/config.yml
```

Para la versión No-BERT:
```bash
python train_second_no_bert.py --config_path ./Configs/config.yml
```

El modelo se guardará en el formato "epoch_1st_%05d.pth" y "epoch_2nd_%05d.pth". Los checkpoints y los registros de Tensorboard se guardarán en `log_dir`.

### Diferencias Clave con el StyleTTS2 Original

El archivo de configuración tiene dos diferencias principales respecto al StyleTTS2 original:

1. Tipo de Decodificador:
   ```yaml
   decoder:
     type: 'vocos'  # Cambiado desde el original 'hifigan' o 'istftnet'
   ```

2. Ruta Raíz (Root Path):
   ```yaml
   data_params:
     root_path: "/data/LibriTTS"  # Actualice esta ruta según la ubicación de su conjunto de datos
   ```

## Inferencia

Consulte los cuadernos de Jupyter en la carpeta `Demo` para ver ejemplos detallados de inferencia:
- `Inference_stage1.ipynb`: Para la inferencia de la primera etapa.
- `Inference_stage2.ipynb`: Para la inferencia de la segunda etapa.

## Contribuciones

Agradezco las contribuciones para mejorar la implementación del decodificador Vocos, particularmente en las siguientes áreas:
- Implementación del módulo de fuente en `Modules/vocos.py`.
- Optimización del pase hacia adelante (forward pass).
- Mejoras de calidad manteniendo los beneficios de eficiencia.

Si tiene ideas o implementaciones que puedan mejorar la calidad del audio manteniendo los beneficios de eficiencia, no dude en contribuir.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulte el archivo LICENSE para más detalles.
