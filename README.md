# Senior Thesis

## Datasets
- Formality Datasets
    - `ACL-IWSLT 2022 Formality Track Datasets`
        - The ACL-IWSLT hosted the Formality Track in 2022 and 2023. They specialized in formality-sensitive translations and methods to improve them. The datasets are available through these links: 2022[https://github.com/amazon-science/contrastive-controlled-mt/tree/main/IWSLT2022] and 2023[https://github.com/amazon-science/contrastive-controlled-mt/tree/main/IWSLT2023]. This project focuses on English text to Spanish speech translation, so only the 2022 dataset was used.
    - `FAME-MT Dataset`
        - `FAME-MT Dataset` was made available in June 2024 through `EAMT 2024`. 


## Helpful Tutorials I referenced throughout project
- SpeechT5 Model Docs: https://huggingface.co/docs/transformers/en/model_doc/speecht5
    - If you are interested in Text-to-Speech see the model card for `SpeechT5ForTextToSpeech`.
    - If you click expand, this will explain parameters such as `input_ids` (obtained with `SpeechT5 Tokenizer`), `decoder_input_values` (float values of input mel-spectrogram), `speaker_embeddings` (tensor containing speaker embeddings), and `labels` (float values of target mel spectrogram, which can be obtained with `SpeechT5Processor`).
