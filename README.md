# Senior Thesis

## Helpful Tutorials I referenced throughout project
- SpeechT5 Model Docs: https://huggingface.co/docs/transformers/en/model_doc/speecht5
    - If you are interested in Text-to-Speech see the model card for `SpeechT5ForTextToSpeech`.
    - If you click expand, this will explain parameters such as `input_ids` (obtained with `SpeechT5 Tokenizer`), `decoder_input_values` (float values of input mel-spectrogram), `speaker_embeddings` (tensor containing speaker embeddings), and `labels` (float values of target mel spectrogram, which can be obtained with `SpeechT5Processor`).