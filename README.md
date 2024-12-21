# NTU_ADL_2023_Fall
- This repository contains the homework assignments and final project implementations in NTU 2023 Fall Applied Deep Learning
- The corresponding README file of each assignment is placed under the assignment folder
* * *

## HW1
### Chinese Extractive Question Answering (QA)
- 說明連結 : https://docs.google.com/presentation/d/1V5KE-AOTiVXWZjMVnr3GXyy_EjEMHeRKfeyudyeevl8/edit#slide=id.p
- Fine-tune `bert-base-chinese` model
   1. Paragraph Selection: Determine which paragraph is relevant
      - Implemented in `Multiple Choice` framework
   2. Span selection: Determine the start and end position of the answer span
      - Implemented in `Question Answering` framework
* * *
## HW2
### Chinese News Summarization (Title Generation)
- 說明連結 :  https://docs.google.com/presentation/d/1C9dhFQvz--9sDtjGukSL6Lmfp8CPQgmG/edit#slide=id.p1
- Fine-tune pre-trained `small-multilingual-T5` model
   1. Input: news content
   2. Output: news title
* * *
## HW3
### Instruction Tuning (Classical Chinese)
- 說明連結 : https://docs.google.com/presentation/d/1gbutje764HPndSCmS-I6TZuaGvJqPZwNQj-DR4x2H3o/edit#slide=id.g2976d025caf_0_126
- Fine-tune Taiwan-LLaMa
- Utilize QLoRA technique
