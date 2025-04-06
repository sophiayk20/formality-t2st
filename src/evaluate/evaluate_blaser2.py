"""
    Install: `fairseq2`, `stopes`, `sonar-space` (SONAR)
    fairseq2==v0.3.0rc1, depending on PyTorch version (below is example of 2.5.1+cu124)
    reference: # https://github.com/facebookresearch/large_concept_model?tab=readme-ov-file#installing
    pip install fairseq2==v0.3.0rc1 --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/cu124

    pip install -q stopes sonar-space
"""

from stopes.eval.blaser.blaser2 import compute_blaser2
import os

with open("/content/drive/MyDrive/formality/formal-outputs.txt", "r") as f:
  src_text = [line.rstrip() for line in f.readlines()]

tgt_paths = os.listdir('/content/drive/MyDrive/formality/formal-base')
tgt_paths = ['/content/drive/MyDrive/formality/formal-base/'+ path for path in tgt_paths]

(src_embs, ref_embs, tgt_embs), df_with_scores = compute_blaser2(
    src_column=None,
    ref_column=None,
    tgt_column=None,
    blaser_path="blaser_2_0_qe",  # or blaser_2_0_qe, if you don't use references
    src_lang="eng_Latn",  # lookup language codes on SONAR model cards
    tgt_lang="spa",
    src=src_text,
    tgt=tgt_paths,
    src_is_speech=False,
    tgt_is_speech=True,
)
mean_score = df_with_scores.mean(numeric_only=True)
print(mean_score.unsupervised_scores)  # a number usually between 0 and 1
print(mean_score.supervised_scores)    # a number usually between 1 and 5