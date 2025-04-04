# Controlling Formality in Text-to-Speech Translation

## Formality Datasets Used
- Formality Datasets (Text-to-Text Modality)
    - `ACL-IWSLT 2022 Formality Track Datasets`
        - The ACL-IWSLT hosted the Formality Track in 2022 and 2023. They specialized in formality-sensitive translations and methods to improve them. The datasets are available through these links: [2022](https://github.com/amazon-science/contrastive-controlled-mt/tree/main/IWSLT2022) and [2023](https://github.com/amazon-science/contrastive-controlled-mt/tree/main/IWSLT2023). This project focuses on English text to Spanish speech translation, so only the 2022 dataset was used.
    - `FAME-MT Dataset`
        - `FAME-MT Dataset` was made available in June 2024 through `EAMT 2024`. See the paper: "FAME-MT Dataset: Formality Awareness Made Easy for Machine Translation Purposes." The dataset is available here: https://github.com/laniqo-public/fame-mt.
- Formal and informal datasets are concatenated and shuffled. These amount to about 40K points of training data for each formality class.

## Tech Stack
- transformers: AutoTokenizer, AutoModelForSeq2SeqLM
- evaluate: sacrebleu
- `fairseq2`, `stopes`, `sonar-space (SONAR)`
- `soundfile`, `pyannote`
- `ParlerTTS`
- `coqui-ai/tts`

## Demo Page
Demo page is available at [this link](https://sophiayk20.github.io/tts-formality-translation).
I did not design HTML / CSS for the page and all rights for that portion of the code goes to their respectful authors.

## Helpful Tutorials I referenced throughout project
- [SpeechT5 Model Docs](https://huggingface.co/docs/transformers/en/model_doc/speecht5)
    - If you are interested in Text-to-Speech see the model card for `SpeechT5ForTextToSpeech`.
    - If you click expand, this will explain parameters such as `input_ids` (obtained with `SpeechT5 Tokenizer`), `decoder_input_values` (float values of input mel-spectrogram), `speaker_embeddings` (tensor containing speaker embeddings), and `labels` (float values of target mel spectrogram, which can be obtained with `SpeechT5Processor`).
