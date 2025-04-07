# ASR-BLEU transcriptions

import evaluate

def find_bleu_score(LANG_CODE, FORMALITY, MODE):
  opposite_formality = {'formal': 'informal', 'informal': 'formal'}
  FILENAME_PATH = f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.files.txt"
  # needs to be common in both formal and informal to use for %M-A
  TRANSCRIPTION_PATH = f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.common.transcriptions.txt"
  # ACL IWSLT FORMALITY SENSITIVE DETOKENIZED REFERENCE PATH
  CLEAN_REFERENCE_PATH=f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.detokenized.references.txt"
  OPPOSITE_REFERENCE_PATH=f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{opposite_formality[FORMALITY]}.detokenized.references.txt"

  with open(TRANSCRIPTION_PATH, "r") as f:
    transcriptions = f.readlines()
    transcriptions = [line.strip() for line in transcriptions]

  with open(CLEAN_REFERENCE_PATH, "r") as f:
    individual_references = f.readlines()
    individual_references = [line.strip() for line in individual_references]

  # with open(OPPOSITE_REFERENCE_PATH, "r") as f:
  #   opp_individual_references = f.readlines()
  #   opp_individual_references = [line.strip() for line in opp_individual_references]

  references = []
  opposite_references=[]
  for line in individual_references:
    references.append([line])
  # for line in opp_individual_references:
  #   opposite_references.append([line])

  assert(len(individual_references) == len(transcriptions))
  # assert(len(opp_individual_references) == len(transcriptions))

  bleu = evaluate.load("bleu")
  results = bleu.compute(predictions=transcriptions,references=references)
  print(f"LANG: {LANG_CODE} FORMALITY: {FORMALITY} BLEU SCORE: {round(100*results['bleu'], 2)}")
  # results = bleu.compute(predictions=transcriptions, references=opposite_references)
  # print(f"LANG: {LANG_CODE} OPP FORMALITY: {opposite_formality[FORMALITY]} BLEU_SCORE: {round(100*results['bleu'], 2)}")

LANG_CODES=['de', 'es', 'fr', 'it']
FORMALITIES=['formal']
MODES=['BB']

for MODE in MODES:
  for FORMALITY in FORMALITIES:
    for LANG_CODE in LANG_CODES:
      find_bleu_score(LANG_CODE, FORMALITY, MODE)